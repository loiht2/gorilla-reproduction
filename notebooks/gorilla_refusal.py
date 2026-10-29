"""Independent API-withholding trials, static API taxonomy, and truthful artifacts."""
import ast
import copy
import hashlib
import json
import re
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from gorilla_eval import MEDIA, PROTOCOL_VERSION, _hash, make_prompt, parse_call, score_call, sdk_function
from gorilla_vllm import VllmEndpoint, normalize_completion


def write_json(path, value, forbidden_values=()):
    text=json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    if any(secret and secret in text for secret in forbidden_values):
        raise ValueError('Sensitive content detected; artifact was not written')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(text,encoding='utf-8');temp.replace(path)


def retrieve(apis, query_texts):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    matrix=vectorizer.fit_transform([a['description'] for a in apis])
    scores=cosine_similarity(vectorizer.transform(query_texts),matrix)
    order=np.argsort(-scores,axis=1,kind='stable')
    ids=[a['api_name'] for a in apis]
    rankings=[[{'api_name':ids[int(i)],'score':float(row[int(i)])} for i in indices]
              for row,indices in zip(scores,order)]
    metadata={'fitted_document_count':len(apis),'vocabulary_size':len(vectorizer.vocabulary_),
              'idf_sha256':_hash({'vocabulary':vectorizer.vocabulary_,'idf':vectorizer.idf_.tolist()}),
              'index_sha256':_hash([{'api_name':a['api_name'],'description':a['description']} for a in apis])}
    return rankings,metadata


def build_trials(apis,queries,answers,selected_ids):
    original_ids=[a['api_name'] for a in apis]
    if len(original_ids)!=len(set(original_ids)) or len(selected_ids)!=len(set(selected_ids)):
        raise ValueError('Duplicate API/query IDs')
    qmap={q['request_id']:q for q in queries};gold={a['request_id']:a['correct_api'] for a in answers}
    trials=[]
    for rid in selected_ids:
        removed=gold[rid]
        if original_ids.count(removed)!=1:raise ValueError('Gold API must occur exactly once')
        index=[a for a in apis if a['api_name']!=removed]  # Fresh original catalog EVERY time.
        ranking,metadata=retrieve(index,[qmap[rid]['input']])
        top3=[r['api_name'] for r in ranking[0][:3]]
        prompt=make_prompt(qmap[rid],top3,index)
        trials.append({'request_id':rid,'removed_api':removed,'index_api_ids':[a['api_name'] for a in index],
                       'index_metadata':metadata,'ranking':ranking[0],'retrieved_apis':top3,
                       'prompt':prompt,'prompt_sha256':_hash(prompt),
                       'prompt_utf8_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    return trials


def _expression(text):
    if not isinstance(text,str) or len(text)>100000:raise ValueError('Output size/type')
    text=text.strip();fence=re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```',text,re.S)
    return ast.parse(fence[1] if fence else text,mode='eval').body


def api_identity(text,registry):
    """Identify a named API without evaluating the expression or its arguments."""
    try:node=_expression(text)
    except (ValueError,SyntaxError,TypeError,RecursionError):return {'identity':'NO_CALL'}
    if not isinstance(node,ast.Call):return {'identity':'NO_CALL'}
    methods=registry['sdk_methods'];lookup={(r['alias'],r['method']):r for r in methods}
    aliases={r['alias'] for r in methods};classes={r['class']:r['alias'] for r in methods}
    alias=None;method=None;form=None
    if isinstance(node.func,ast.Name):
        name=node.func.id
        if name in registry['rest_operations']:
            return {'identity':'KNOWN','operation_id':name,'form':'canonical_rest_operation_id'}
        matches=[r for r in methods if r['method']==name]
        if len(matches)==1:
            return {'identity':'KNOWN',**matches[0],'form':'bare_sdk_method'}
        return {'identity':'UNRESOLVED' if matches else 'UNKNOWN','function_name':name,
                'known_method_name':bool(matches),'form':'bare_name'}
    if isinstance(node.func,ast.Attribute):
        method=node.func.attr;value=node.func.value
        if isinstance(value,ast.Name):
            alias=value.id;form='client_alias'
        elif isinstance(value,ast.Call):
            constructor=value.func
            classname=constructor.id if isinstance(constructor,ast.Name) else constructor.attr if isinstance(constructor,ast.Attribute) else None
            alias=classes.get(classname);form='sdk_constructor'
        if alias is not None:
            for suffix in MEDIA:
                if alias.endswith(suffix):alias=alias[:-len(suffix)];break
            alias=classes.get(alias,alias)
            method_base=method.removesuffix('_with_http_info')
            match=lookup.get((alias,method_base))
            if match:return {'identity':'KNOWN',**match,'form':form}
            known_name=any(r['method']==method_base for r in methods)
            if alias in aliases:return {'identity':'UNKNOWN','alias':alias,'function_name':method}
            return {'identity':'UNRESOLVED' if known_name else 'UNKNOWN','function_name':method,
                    'known_method_name':known_name,'form':'unknown_receiver'}
    return {'identity':'UNRESOLVED','form':'unsupported_call_interface'}


def classify_response(record,trial,apis,registry):
    result={'request_id':record.get('request_id'),'outcome':None,'exact_refusal':False,
            'normalized_refusal':False,'call_format_valid':None,'availability':None,
            'correct_call':False,'api_identity':None}
    if record['api_status']!='OK':result['outcome']='API_ERROR';return result
    if record.get('finish_reason')!='STOP':result['outcome']='INCOMPLETE_GENERATION';return result
    text=record.get('text','')
    result['exact_refusal']=text.strip()=='NO_MATCHING_API'
    try:
        parsed=parse_call(text)
        result['normalized_refusal']=parsed is None
        result['call_format_valid']=parsed is not None
    except (ValueError,SyntaxError,TypeError,RecursionError):result['call_format_valid']=False
    if result['normalized_refusal']:
        result['outcome']='REFUSAL' if result['exact_refusal'] else 'REFUSAL_FORMAT_VARIANT'
        return result
    identity=api_identity(text,registry);result['api_identity']=identity
    if identity['identity']!='KNOWN':
        result['outcome']={'NO_CALL':'INVALID_OUTPUT','UNKNOWN':'UNKNOWN_API','UNRESOLVED':'UNRESOLVED_API_BINDING'}[identity['identity']]
        return result
    op=identity.get('operation_id')
    if op is None:
        # Some SDK endpoints have no counterpart in the release OpenAPI; their methods still exist.
        match=[a['api_name'] for a in apis if sdk_function(a)==(identity.get('alias'),identity.get('method'))]
        op=match[0] if len(match)==1 else None
    if op is not None and op==trial.get('removed_api'):availability='REMOVED_CORRECT_API'
    elif op in trial['retrieved_apis']:availability='RETRIEVED'
    elif op in trial['index_api_ids']:availability='IN_INDEX_NOT_RETRIEVED'
    else:availability='OUTSIDE_ORIGINAL_INDEX'
    result.update(availability=availability,selected_api=op,
                  outcome=('AVAILABLE_WRONG_API' if trial.get('removed_api') is not None else 'AVAILABLE_API') if availability=='RETRIEVED' else 'REAL_BUT_UNAVAILABLE')
    return result


def refusal_summary(rows):
    n=len(rows);errors=sum(r['outcome']=='API_ERROR' for r in rows);success=n-errors
    count=sum(r['exact_refusal'] for r in rows);normalized=sum(r['normalized_refusal'] for r in rows)
    return {'queries':n,'api_errors':errors,'api_successes':success,'exact_refusals':count,
            'normalized_refusals':normalized,'refusal_rate_all_queries':count/n if n else None,
            'refusal_rate_api_successes':count/success if success else None,
            'normalized_refusal_rate_all_queries':normalized/n if n else None,
            'normalized_refusal_rate_api_successes':normalized/success if success else None,
            'correct_calls':sum(r.get('correct_call',False) for r in rows),
            'outcome_counts':dict(Counter(r['outcome'] for r in rows)),
            'real_but_unavailable_breakdown':dict(Counter(r['availability'] for r in rows if r['outcome']=='REAL_BUT_UNAVAILABLE'))}


def evaluate_refusal(run,trials,answers,apis,registry):
    tmap={t['request_id']:t for t in trials};amap={a['request_id']:a for a in answers}
    if len(run['records'])!=len(tmap) or {r['request_id'] for r in run['records']}!=set(tmap):
        raise ValueError('Incomplete or duplicate refusal results')
    rows=[]
    for record in run['records']:
        trial=tmap[record['request_id']]
        row=classify_response(record,trial,apis,registry)
        if record['api_status']=='OK' and record.get('finish_reason')=='STOP':
            old_score=score_call(record.get('text',''),amap[record['request_id']],apis,trial['retrieved_apis'])
            row['correct_call']=old_score['correct_call']
            row['baseline_scorer_status']=old_score['status']
        rows.append(row)
    return {'model':run['model'],'generation_config':run['generation_config'],
            'summary':refusal_summary(rows),'rows':rows}


def run_refusal(trials,model,settings,output_file,google_request=None,connection_file=None,
                allow_requests=False,forbidden_values=()):
    """Only trial.prompt is transmitted; exclusion metadata and labels stay local."""
    output_file=Path(output_file)
    fingerprint=_hash({'model':model,'settings':settings,'protocol':PROTOCOL_VERSION,
                       'prompt_sha256':[t['prompt_sha256'] for t in trials]})
    if output_file.exists():
        run=json.loads(output_file.read_text())
        if run['fingerprint']!=fingerprint:raise ValueError('Cached refusal run differs')
        ids=[r['request_id'] for r in run['records']]
        if len(ids)!=len(set(ids)) or not set(ids)<={t['request_id'] for t in trials}:
            raise ValueError('Invalid cached trial IDs')
        if len(ids)==len(trials):return run
    else:
        run={'model':model,'generation_config':copy.deepcopy(settings),'protocol':PROTOCOL_VERSION,
             'fingerprint':fingerprint,'started_at':datetime.now(timezone.utc).isoformat(),
             'attempt_policy':'One attempt per trial; no automatic retry/fallback.','records':[]}
    if not allow_requests:raise RuntimeError('No complete cached refusal run; explicitly enable new requests')
    is_qwen=model=='Qwen3-8B'
    if not is_qwen and model!='gemma-4-31b-it':raise ValueError('Unsupported model')
    context=VllmEndpoint(connection_file) if is_qwen else nullcontext(None)
    with context as endpoint:
        if is_qwen:
            counts={}
            for trial in trials:
                r=endpoint.request('/tokenize',{'model':endpoint.model_id,
                  'messages':[{'role':'user','content':trial['prompt']}],'add_generation_prompt':True,
                  'chat_template_kwargs':settings['chat_template_kwargs']})
                if not r['ok']:raise RuntimeError('Token counting failed; raw error suppressed')
                count=r['data'].get('count')
                if type(count)is not int or count+settings['max_tokens']>endpoint.context_limit:
                    raise ValueError('Withheld prompt cannot fit with unchanged baseline settings')
                counts[trial['request_id']]=count
            run['prompt_token_counts']=counts;run['model_id_sha256']=_hash(endpoint.model_id)
        write_json(output_file,run,forbidden_values)
        existing={r['request_id'] for r in run['records']}
        for trial in trials:
            if trial['request_id'] in existing:continue
            prompt=trial['prompt']
            if any(s and s in prompt for s in forbidden_values):raise ValueError('Sensitive prompt blocked')
            started=time.monotonic()
            if is_qwen:
                messages=[{'role':'user','content':prompt}]
                response=endpoint.request('/v1/chat/completions',{'model':endpoint.model_id,'messages':messages,**settings})
                visible_request={'messages':messages,'generation_config':settings,'model_identifier':'private connection manifest; hash recorded'}
            else:
                body={'contents':[{'role':'user','parts':[{'text':prompt}]}],'generationConfig':settings}
                response=google_request('models/'+model+':generateContent',body)
                visible_request=body
            record={'request_id':trial['request_id'],'retrieved_apis':trial['retrieved_apis'],
                    'prompt':prompt,'prompt_sha256':trial['prompt_sha256'],
                    'prompt_utf8_sha256':trial['prompt_utf8_sha256'],'request_content':visible_request,
                    'api_status':'OK' if response['ok'] else 'API_ERROR','http_status':response.get('http_status'),
                    'attempts':1,'latency_seconds':round(time.monotonic()-started,3),
                    'text':None,'exact_response_text':None}
            if response['ok']:
                data=response['data']
                if is_qwen:
                    choice=(data.get('choices') or [{}])[0]
                    record['exact_response_text']=choice.get('message',{}).get('content')
                    record.update(normalize_completion(data))
                    record['model_version']=model
                else:
                    candidate=(data.get('candidates') or [{}])[0]
                    parts=[p for p in candidate.get('content',{}).get('parts',[]) if not p.get('thought')]
                    text=''.join(p.get('text','') for p in parts)
                    record.update(text=text,exact_response_text=text,final_response_parts=parts,
                                  finish_reason=candidate.get('finishReason'),usage=data.get('usageMetadata',{}),
                                  model_version=data.get('modelVersion',model))
            else:record['error_type']=response.get('error','API_ERROR')
            run['records'].append(record);run['updated_at']=datetime.now(timezone.utc).isoformat()
            write_json(output_file,run,forbidden_values)
            print(model,trial['request_id'],record['api_status'],flush=True)
            if not is_qwen:time.sleep(2.1)
    return run
