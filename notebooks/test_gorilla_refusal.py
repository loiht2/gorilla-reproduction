"""Offline tests for independent exclusion and refusal outcome taxonomy."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
MODULE=Path(__file__).with_name('gorilla_refusal.py')

class RefusalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod=None
        if MODULE.exists():
            spec=importlib.util.spec_from_file_location('gorilla_refusal',MODULE)
            cls.mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.mod)
        cls.apis=json.loads((ROOT/'data/apis.json').read_text())['apis']
        cls.registry=json.loads((ROOT/'data/evaluation/refusal/kubernetes_api_registry.json').read_text())
        cls.removed='readCoreV1NamespacedPod'
        cls.full=[a['api_name'] for a in cls.apis]
        cls.remaining=[x for x in cls.full if x!=cls.removed]
        cls.retrieved=['listCoreV1NamespacedPod','readCoreV1NamespacedPodLog','createCoreV1NamespacedPod']
    def setUp(self):self.assertIsNotNone(self.mod,'Refusal experiment not implemented')
    def classify(self,text,api_status='OK',finish='STOP'):
        return self.mod.classify_response({'api_status':api_status,'finish_reason':finish,'text':text},
              {'removed_api':self.removed,'index_api_ids':self.remaining,'retrieved_apis':self.retrieved},self.apis,self.registry)
    def test_exclusions_are_independent_and_original_is_unchanged(self):
        queries=json.loads((ROOT/'data/queries.json').read_text())['requests']
        answers=json.loads((ROOT/'data/evaluation/answers.json').read_text())['answers']
        before=json.dumps(self.apis,sort_keys=True)
        trials=self.mod.build_trials(self.apis,queries,answers,['req_002','req_004'])
        self.assertEqual([len(t['index_api_ids']) for t in trials],[29,29])
        for i,t in enumerate(trials):
            self.assertNotIn(t['removed_api'],t['index_api_ids'])
            self.assertNotIn(t['removed_api'],t['retrieved_apis'])
            self.assertIn(trials[1-i]['removed_api'],t['index_api_ids'])
            self.assertIn('idf_sha256',t['index_metadata'])
        self.assertEqual(before,json.dumps(self.apis,sort_keys=True))
    def test_only_bare_sentinel_is_primary_refusal(self):
        self.assertTrue(self.classify('  NO_MATCHING_API\n')['exact_refusal'])
        fenced=self.classify('```python\nNO_MATCHING_API\n```')
        self.assertFalse(fenced['exact_refusal']);self.assertTrue(fenced['normalized_refusal'])
        self.assertFalse(self.classify('NO_MATCHING_API because unavailable')['exact_refusal'])
    def test_removed_real_api_is_not_unknown(self):
        r=self.classify("core_v1.read_namespaced_pod(name='sample',namespace='demo')")
        self.assertEqual(r['outcome'],'REAL_BUT_UNAVAILABLE')
        self.assertEqual(r['availability'],'REMOVED_CORRECT_API')
    def test_real_canonical_operation_with_invalid_python_binding_is_not_invented(self):
        r=self.classify("readCoreV1NamespacedPod(name='sample',namespace='demo')")
        self.assertEqual(r['outcome'],'REAL_BUT_UNAVAILABLE')
        self.assertFalse(r['call_format_valid'])
    def test_real_sdk_method_outside_catalog_is_not_unknown(self):
        r=self.classify("core_v1.list_namespaced_service(namespace='demo')")
        self.assertEqual(r['outcome'],'REAL_BUT_UNAVAILABLE')
        self.assertEqual(r['availability'],'OUTSIDE_ORIGINAL_INDEX')
    def test_real_remaining_but_unretrieved_api_is_separate(self):
        r=self.classify("core_v1.delete_namespaced_pod(name='sample',namespace='demo')")
        self.assertEqual(r['availability'],'IN_INDEX_NOT_RETRIEVED')
    def test_unknown_method_and_unresolved_binding_differ(self):
        self.assertEqual(self.classify('core_v1.teleport_namespaced_pod()')['outcome'],'UNKNOWN_API')
        self.assertEqual(self.classify('unbound.read_namespaced_pod()')['outcome'],'UNRESOLVED_API_BINDING')
    def test_available_wrong_api_and_noncall_output_are_separate(self):
        self.assertEqual(self.classify("core_v1.list_namespaced_pod(namespace='demo')")['outcome'],'AVAILABLE_WRONG_API')
        self.assertEqual(self.classify('{"operation":"read"}')['outcome'],'INVALID_OUTPUT')
    def test_api_errors_and_incomplete_outputs_are_not_refusals(self):
        self.assertEqual(self.classify('NO_MATCHING_API','API_ERROR')['outcome'],'API_ERROR')
        self.assertFalse(self.classify('NO_MATCHING_API',finish='MAX_TOKENS')['exact_refusal'])
    def test_denominators_keep_api_failures(self):
        rows=[self.classify('NO_MATCHING_API'),self.classify('NO_MATCHING_API','API_ERROR'),self.classify('core_v1.teleport_namespaced_pod()')]
        summary=self.mod.refusal_summary(rows)
        self.assertEqual(summary['api_errors'],1)
        self.assertAlmostEqual(summary['refusal_rate_all_queries'],1/3)
        self.assertAlmostEqual(summary['refusal_rate_api_successes'],1/2)
    def test_secret_guard_refuses_to_write_sensitive_content(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'artifact.json'
            with self.assertRaises(ValueError):self.mod.write_json(p,{'text':'TEST_SECRET_VALUE'},['TEST_SECRET_VALUE'])
            self.assertFalse(p.exists())

if __name__=='__main__':unittest.main()
