"""Local vLLM runner sharing the unchanged prompt builder and AST evaluator.

Only temporary loopback forwarding and inference requests are used. No Kubernetes
resource mutations, credential forwarding, or generated-code execution occur.
"""
import json
import re
import select
import subprocess
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from gorilla_eval import PROTOCOL_VERSION, make_prompt, _hash

MODEL_LABEL = 'Qwen3-8B'
ROUTES = {'/health','/version','/v1/models','/tokenize','/v1/chat/completions'}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

def http_json(base, endpoint, payload=None, timeout=120):
    if not re.fullmatch(r'http://127\.0\.0\.1:[0-9]{1,5}',base) or endpoint not in ROUTES:
        return {'ok':False,'error':'INVALID_LOCAL_ENDPOINT'}
    try:
        req = urllib.request.Request(base+endpoint,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type':'application/json'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
        with opener.open(req,timeout=timeout) as response:
            raw = response.read()
            return {'ok':True,'http_status':response.status,'data':json.loads(raw) if raw else {}}
    except urllib.error.HTTPError as error:
        return {'ok':False,'error':'HTTP_ERROR','http_status':error.code}
    except TimeoutError:
        return {'ok':False,'error':'TIMEOUT'}
    except urllib.error.URLError:
        return {'ok':False,'error':'NETWORK_ERROR'}
    except (ValueError,OSError):
        return {'ok':False,'error':'INVALID_RESPONSE_OR_TRANSPORT_ERROR'}

def start_output_drain(stream):
    """Discard diagnostics continuously so a tunnel cannot block on its log pipe."""
    def drain():
        while stream.read(8192):
            pass
    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return thread

class VllmEndpoint:
    """A temporary tunnel; sensitive endpoint coordinates stay in a local JSON file."""
    def __init__(self, connection_file):
        self.connection = json.loads(Path(connection_file).read_text())
        self.process = None
        self.log_thread = None

    def __enter__(self):
        c = self.connection
        if type(c.get('port')) is not int or not 0 < c['port'] < 65536:
            raise ValueError('Invalid local connection manifest')
        self.process = subprocess.Popen(
            ['kubectl','-n',c['namespace'],'port-forward','--address=127.0.0.1',
             'pod/'+c['pod'],':'+str(c['port'])],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            deadline = time.monotonic()+25
            while time.monotonic()<deadline:
                ready,_,_ = select.select([self.process.stdout],[],[],1)
                if ready:
                    match = re.search(r'Forwarding from 127\.0\.0\.1:(\d+)',self.process.stdout.readline())
                    if match:
                        self.base = 'http://127.0.0.1:'+match[1]
                        break
                if self.process.poll() is not None:
                    break
            else:
                raise RuntimeError('Local forwarding timed out; details suppressed')
            if not hasattr(self,'base'):
                raise RuntimeError('Local forwarding failed; details suppressed')
            self.log_thread = start_output_drain(self.process.stdout)
            health = self.request('/health')
            models = self.request('/v1/models')
            if not health['ok'] or not models['ok']:
                raise RuntimeError('vLLM health/model discovery failed; raw errors suppressed')
            matches = [m for m in models['data'].get('data',[]) if m.get('id')==c['model']]
            if len(matches)!=1 or 'qwen3' not in c['model'].lower() or '8b' not in c['model'].lower():
                raise RuntimeError('The configured Qwen3-8B model is not served')
            self.model_id = c['model']
            self.context_limit = matches[0].get('max_model_len')
            if type(self.context_limit) is not int:
                raise RuntimeError('No advertised model context limit; cannot safely budget requests')
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def request(self, endpoint, payload=None):
        return http_json(self.base,endpoint,payload)

    def __exit__(self,*args):
        if self.process is not None:
            self.process.terminate()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            if self.log_thread is not None:
                self.log_thread.join(timeout=2)
            self.process.stdout.close()
        return False

def output_budget(context_limit, prompt_counts, requested=4096):
    if (type(context_limit) is not int or not prompt_counts
            or any(type(n) is not int or n<1 for n in prompt_counts)):
        raise ValueError('Invalid context/token-count evidence')
    budget = min(requested,context_limit-max(prompt_counts))
    if budget<1:
        raise ValueError('An unchanged prompt does not fit the served context window')
    return budget

def normalize_completion(data):
    candidates = data.get('choices',[])
    candidate = candidates[0] if candidates else {}
    text = candidate.get('message',{}).get('content') or ''
    if not isinstance(text,str):text = ''
    reason = {'stop':'STOP','length':'MAX_TOKENS','content_filter':'SAFETY'}.get(candidate.get('finish_reason'),'UNKNOWN')
    # Separate provider reasoning exactly as Gemini thought parts are excluded.
    stripped = text.lstrip()
    had_reasoning = bool(candidate.get('message',{}).get('reasoning_content'))
    if stripped.startswith('<think>'):
        had_reasoning = True
        end = stripped.find('</think>')
        if end<0:
            text,reason = '', 'UNTERMINATED_REASONING'
        else:text = stripped[end+len('</think>'):].lstrip()
    usage = data.get('usage',{}) or {}
    return {'text':text,'finish_reason':reason,'reasoning_excluded':had_reasoning,
            'usage':{'promptTokenCount':usage.get('prompt_tokens'),
                     'candidatesTokenCount':usage.get('completion_tokens'),
                     'totalTokenCount':usage.get('total_tokens')}}

def verify_same_inputs(left,right):
    def mapping(run):
        rows = run['records']
        result = {r['request_id']:(r['prompt_sha256'],r['retrieved_apis']) for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate experiment IDs')
        return result
    if mapping(left)!=mapping(right):
        raise ValueError('Model runs do not use identical query prompts and retrieved API candidates')

def _save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)

def run_qwen(queries,top3,apis,connection_file,output_file,allow_requests=False):
    """One request per query, with saved responses; never accepts gold answers."""
    output_file = Path(output_file)
    if len(queries)!=len(top3):raise ValueError('Query/candidate alignment differs')
    prepared = [{'request_id':q['request_id'],'retrieved_apis':list(ids),
                 'prompt':make_prompt(q,list(ids),apis)} for q,ids in zip(queries,top3)]
    if len({p['request_id'] for p in prepared})!=len(prepared):raise ValueError('Duplicate query IDs')
    fingerprint = _hash({'model':MODEL_LABEL,'provider':'vllm','protocol':PROTOCOL_VERSION,
                         'temperature':0,'requested_max_tokens':4096,'enable_thinking':False,
                         'budget_policy':'min(4096, context_limit - largest_prompt_count)',
                         'prompts':prepared})
    if output_file.exists():
        run = json.loads(output_file.read_text())
        if run.get('fingerprint')!=fingerprint:raise ValueError('Saved Qwen run differs; use a new file for a new experiment')
        ids = [r['request_id'] for r in run['records']]
        if len(ids)!=len(set(ids)) or not set(ids)<={p['request_id'] for p in prepared}:
            raise ValueError('Invalid cached query IDs')
        if len(ids)==len(prepared):return run
    else:run = None
    if not allow_requests:
        raise RuntimeError('No complete cached Qwen run; set RUN_QWEN=True to send missing requests')
    with VllmEndpoint(connection_file) as endpoint:
        counts = {}
        for item in prepared:
            response = endpoint.request('/tokenize',{'model':endpoint.model_id,
                'messages':[{'role':'user','content':item['prompt']}],
                'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})
            if not response['ok']:raise RuntimeError('Prompt token counting failed; no benchmark request sent')
            counts[item['request_id']] = response['data'].get('count')
        budget = output_budget(endpoint.context_limit,list(counts.values()))
        config = {'temperature':0,'max_tokens':budget,'chat_template_kwargs':{'enable_thinking':False}}
        if run is None:
            run = {'model':MODEL_LABEL,'provider':'vllm','protocol':PROTOCOL_VERSION,
                   'fingerprint':fingerprint,'generation_config':config,
                   'requested_max_tokens':4096,'prompt_token_counts':counts,
                   'context_checks_passed':True,'model_id_sha256':_hash(endpoint.model_id),
                   'started_at':datetime.now(timezone.utc).isoformat(),'records':[]}
            _save(output_file,run)
        elif (run['generation_config']!=config or run['prompt_token_counts']!=counts
              or run['model_id_sha256']!=_hash(endpoint.model_id)):
            raise ValueError('Serving configuration changed; do not mix experiments')
        existing = {r['request_id'] for r in run['records']}
        print('All prompts fit; Qwen output token limit:',budget,flush=True)
        for item in prepared:
            if item['request_id'] in existing:continue
            started = time.monotonic()
            response = endpoint.request('/v1/chat/completions',{
                'model':endpoint.model_id,'messages':[{'role':'user','content':item['prompt']}],**config})
            record = {'request_id':item['request_id'],'retrieved_apis':item['retrieved_apis'],
                      'prompt_sha256':_hash(item['prompt']),'http_status':response.get('http_status'),
                      'api_status':'OK' if response['ok'] else 'API_ERROR','attempts':1,
                      'latency_seconds':round(time.monotonic()-started,3)}
            if response['ok']:
                record.update(normalize_completion(response['data']))
                record['model_version'] = MODEL_LABEL
            else:record['error_type'] = response.get('error','API_ERROR')
            run['records'].append(record)
            run['updated_at'] = datetime.now(timezone.utc).isoformat()
            _save(output_file,run)
            print(item['request_id'],record['api_status'],flush=True)
    return run
