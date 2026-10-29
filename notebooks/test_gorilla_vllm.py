"""Offline tests for vLLM transport, budget checks, and cache isolation."""
import importlib.util
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest

MODULE = Path(__file__).with_name('gorilla_vllm.py')

class VllmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = None
        if MODULE.exists():
            spec = importlib.util.spec_from_file_location('gorilla_vllm', MODULE)
            cls.mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.mod)

    def setUp(self):
        self.assertIsNotNone(self.mod, 'vLLM adapter is not implemented yet')

    def test_forwarder_output_is_drained_without_blocking_the_child(self):
        # A forwarder can emit more than a pipe buffer of diagnostics.
        self.assertTrue(hasattr(self.mod, 'start_output_drain'), 'Output drain is missing')
        import subprocess, sys
        p = subprocess.Popen([sys.executable, '-c',
            "import sys; sys.stdout.write('x' * 1048576); sys.stdout.flush()"],
            stdout=subprocess.PIPE, text=True)
        try:
            thread = self.mod.start_output_drain(p.stdout)
            self.assertEqual(p.wait(timeout=5), 0)
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        finally:
            if p.poll() is None: p.kill(); p.wait()
            p.stdout.close()

    def test_output_budget_fits_every_unchanged_prompt(self):
        self.assertEqual(self.mod.output_budget(4096, [1000, 1600], 4096), 2496)
        self.assertEqual(self.mod.output_budget(8192, [1000], 4096), 4096)
        with self.assertRaises(ValueError):self.mod.output_budget(1024, [1024], 4096)

    def test_response_normalizes_finish_and_excludes_reasoning(self):
        source = {'choices':[{'message':{'content':'core_v1.list_namespaced_pod(namespace="demo")',
                    'reasoning_content':'DO_NOT_RETAIN'},'finish_reason':'stop'}],
                  'usage':{'prompt_tokens':10,'completion_tokens':20,'total_tokens':30}}
        result = self.mod.normalize_completion(source)
        self.assertEqual(result['finish_reason'], 'STOP')
        self.assertNotIn('DO_NOT_RETAIN', json.dumps(result))
        self.assertEqual(result['usage']['totalTokenCount'], 30)

    def test_thinking_block_is_separated_but_partial_output_is_not_accepted(self):
        source = {'choices':[{'message':{'content':'<think>analysis</think>\nNO_MATCHING_API'},'finish_reason':'stop'}]}
        self.assertEqual(self.mod.normalize_completion(source)['text'], 'NO_MATCHING_API')
        source['choices'][0]['finish_reason'] = 'length'
        self.assertEqual(self.mod.normalize_completion(source)['finish_reason'], 'MAX_TOKENS')
        source['choices'][0]['message']['content'] = '<think>unfinished'
        self.assertNotEqual(self.mod.normalize_completion(source)['finish_reason'], 'STOP')

    def test_transport_rejects_external_endpoints(self):
        r = self.mod.http_json('https://example.com', '/v1/models')
        self.assertFalse(r['ok'])
        self.assertEqual(r['error'], 'INVALID_LOCAL_ENDPOINT')

    def test_http_transport_has_no_google_credentials_and_sanitizes_errors(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append((dict(self.headers),body))
                self.send_response(500);self.send_header('Content-Type','application/json');self.end_headers()
                self.wfile.write(b'{"message":"PRIVATE_ERROR_BODY"}')
            def log_message(self,*args):pass
        server = HTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            r = self.mod.http_json('http://127.0.0.1:'+str(server.server_port),'/v1/chat/completions',{'temperature':0})
            self.assertEqual(r['http_status'],500)
            self.assertNotIn('PRIVATE_ERROR_BODY',json.dumps(r))
            headers = {k.lower():v for k,v in seen[0][0].items()}
            self.assertNotIn('authorization',headers)
            self.assertNotIn('x-goog-api-key',headers)
            self.assertEqual(seen[0][1]['temperature'],0)
        finally:server.shutdown();server.server_close();thread.join()

    def test_comparison_requires_same_prompts_and_retrieval(self):
        left = {'records':[{'request_id':'x','prompt_sha256':'one','retrieved_apis':['a','b','c']}]}
        same = json.loads(json.dumps(left));self.mod.verify_same_inputs(left,same)
        same['records'][0]['prompt_sha256']='different'
        with self.assertRaises(ValueError):self.mod.verify_same_inputs(left,same)

if __name__ == '__main__':unittest.main()
