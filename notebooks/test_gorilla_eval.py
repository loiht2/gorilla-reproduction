"""Offline evaluator tests. No model or Kubernetes calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = Path(__file__).with_name('gorilla_eval.py')

class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODULE.exists():
            spec = importlib.util.spec_from_file_location('gorilla_eval', MODULE)
            cls.e = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.e)
            cls.apis = json.loads((ROOT / 'data/apis.json').read_text())['apis']
            cls.by_id = {a['api_name']: a for a in cls.apis}
        else:
            cls.e = None

    def setUp(self):
        self.assertIsNotNone(self.e, 'AST evaluator is not implemented yet')
        self.read = {'request_id': 'example', 'correct_api': 'readCoreV1NamespacedPod',
                     'reference_request': {'path_arguments': {'namespace': 'demo', 'name': 'example'},
                                           'query_arguments': {}, 'headers': {}, 'body': None}}

    def score(self, text, answer=None, allowed=None):
        a = answer or self.read
        return self.e.score_call(text, a, self.apis, allowed or [a['correct_api']])

    def test_named_read_and_keyword_order(self):
        self.assertTrue(self.score("core_v1.read_namespaced_pod(name='example', namespace='demo')")['correct_call'])

    def test_wrong_name_is_an_argument_error(self):
        result = self.score("core_v1.read_namespaced_pod(name='wrong', namespace='demo')")
        self.assertTrue(result['function_correct'])
        self.assertFalse(result['correct_call'])

    def test_wrong_function_is_not_correct(self):
        self.assertFalse(self.score("core_v1.delete_namespaced_pod(name='example', namespace='demo')")['function_correct'])

    def test_nested_code_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'must-not-exist'
            text = "core_v1.read_namespaced_pod(name=__import__('pathlib').Path(" + repr(str(target)) + ").touch(), namespace='demo')"
            self.assertEqual(self.score(text)['status'], 'INVALID_OUTPUT')
            self.assertFalse(target.exists())

    def test_multiple_statements_and_expansion_rejected(self):
        for text in ["core_v1.read_namespaced_pod(**{'name':'example','namespace':'demo'})",
                     "core_v1.read_namespaced_pod(name='example', namespace='demo'); print('extra')"]:
            with self.subTest(text=text):self.assertEqual(self.score(text)['status'], 'INVALID_OUTPUT')

    def test_duplicate_keyword_and_dict_key_rejected(self):
        for text in ["core_v1.read_namespaced_pod(name='wrong', name='example', namespace='demo')",
                     "core_v1.read_namespaced_pod(name={'x': 1, 'x': 2}, namespace='demo')"]:
            self.assertEqual(self.score(text)['status'], 'INVALID_OUTPUT')

    def test_omitted_false_watch_and_wrong_integer_watch(self):
        answer = {'correct_api':'listCoreV1NamespacedPod','reference_request':{
            'path_arguments':{'namespace':'demo'},'query_arguments':{'watch':False},'headers':{},'body':None}}
        self.assertTrue(self.score("core_v1.list_namespaced_pod(namespace='demo')", answer)['correct_call'])
        self.assertFalse(self.score("core_v1.list_namespaced_pod(namespace='demo', watch=0)", answer)['correct_call'])

    def test_patch_media_type_matters(self):
        answer = {'correct_api':'patchAppsV1NamespacedDeploymentScale','reference_request':{
            'path_arguments':{'namespace':'demo','name':'web'},'query_arguments':{'dryRun':'All'},
            'headers':{'Content-Type':'application/merge-patch+json'},'body':{'spec':{'replicas':5}}}}
        call = "CLIENT.patch_namespaced_deployment_scale(name='web', namespace='demo', dry_run='All', body={'spec':{'replicas':5}})"
        self.assertTrue(self.score(call.replace('CLIENT', 'apps_v1_merge'), answer)['correct_call'])
        self.assertFalse(self.score(call.replace('CLIENT', 'apps_v1_strategic'), answer)['correct_call'])

    def test_body_namespace_may_be_redundant_but_values_must_match(self):
        answer = {'correct_api':'createCoreV1NamespacedPod','reference_request':{
            'path_arguments':{'namespace':'demo'},'query_arguments':{},'headers':{'Content-Type':'application/json'},
            'body':{'metadata':{'name':'example','namespace':'demo'},'spec':{'restartPolicy':'Never'}}}}
        call = "core_v1.create_namespaced_pod(namespace='demo', body={'spec':{'restartPolicy':'Never'},'metadata':{'name':'example'}})"
        self.assertTrue(self.score(call, answer)['correct_call'])
        self.assertFalse(self.score(call.replace("'Never'", "'Always'"), answer)['correct_call'])

    def test_bool_and_integer_body_values_differ(self):
        answer = {'correct_api':'createSchedulingV1PriorityClass','reference_request':{
            'path_arguments':{},'query_arguments':{},'headers':{},'body':{'value':1,'globalDefault':False}}}
        self.assertFalse(self.score("scheduling_v1.create_priority_class(body={'value':True,'globalDefault':False})", answer)['correct_call'])

    def test_abstention_is_not_a_correct_call(self):
        self.assertEqual(self.score('NO_MATCHING_API')['status'], 'NO_MATCHING_API')
        self.assertFalse(self.score('NO_MATCHING_API')['correct_call'])

    def test_correct_but_unretrieved_call_is_out_of_context(self):
        r = self.score("core_v1.read_namespaced_pod(name='example', namespace='demo')", allowed=['createCoreV1NamespacedPod'])
        self.assertEqual(r['status'], 'OUT_OF_CONTEXT')
        self.assertFalse(r['correct_call'])

    def test_api_errors_have_separate_denominator(self):
        rows = [{'status':'CORRECT_CALL','correct_call':True,'function_correct':True},
                {'status':'WRONG_ARGUMENTS','correct_call':False,'function_correct':True},
                {'status':'API_ERROR','correct_call':False,'function_correct':False}]
        r = self.e.summarize(rows)
        self.assertEqual(r['api_errors'], 1)
        self.assertAlmostEqual(r['correct_call_rate_all_queries'], 1/3)
        self.assertAlmostEqual(r['correct_call_rate_api_successes'], 1/2)

    def test_prompt_contains_only_three_public_cards_and_query_input(self):
        ids = ['createCoreV1NamespacedPod','readCoreV1NamespacedPod','listCoreV1NamespacedPod']
        q = {'request_id':'example','input':'Read the requested Pod.', 'correct_api':'DO_NOT_SEND_GOLD'}
        r = self.e.make_prompt(q, ids, self.apis)
        self.assertNotIn('DO_NOT_SEND_GOLD', r)
        self.assertNotIn('reference_request', r)
        self.assertIn(q['input'], r)
        for name in ids:self.assertIn(self.by_id[name]['description'], r)

if __name__ == '__main__':
    unittest.main()
