"""Export measured baseline/withheld results; never generate or execute model code."""
import hashlib
import json
from pathlib import Path

from gorilla_eval import _hash
from gorilla_refusal import classify_response, evaluate_refusal, refusal_summary, write_json

MODELS = [('gemma', 'gemma_4_31b', 'gemma-4-31b-it'), ('qwen', 'qwen3_8b', 'Qwen3-8B')]


def save_experiment(root, forbidden_values=()):
    root = Path(root)
    folder = root / 'data/evaluation/refusal'
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    plan = load(folder / 'plan.json')
    for rel, digest in plan['source_sha256'].items():
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source changed: ' + rel)
    for rel, key in [('notebooks/gorilla_refusal.py', 'refusal_helper_sha256'),
                     ('data/evaluation/refusal/trials.json', 'trials_sha256'),
                     ('data/evaluation/refusal/kubernetes_api_registry.json', 'registry_sha256')]:
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != plan[key]:
            raise ValueError('Frozen experiment artifact changed: ' + rel)
    apis = load(root / 'data/apis.json')['apis']
    answers = load(root / 'data/evaluation/answers.json')['answers']
    answer_map = {a['request_id']: a for a in answers}
    trials = load(folder / 'trials.json')['trials']
    trial_map = {t['request_id']: t for t in trials}
    prompts = load(folder / 'baseline_prompts.json')['records']
    prompt_map = {p['request_id']: p for p in prompts}
    registry = load(folder / 'kubernetes_api_registry.json')
    all_ids = [a['api_name'] for a in apis]
    failures = []
    summary = {
        'retrieval': load(folder / 'retrieval_metrics.json'),
        'baseline': {}, 'withheld': {},
        'model_settings': {'baseline': {}, 'withheld': plan['model_settings']},
        'query_selection': plan['query_ids'],
        'protocol_amendment': plan.get('protocol_amendment'),
        'generated_code_executed': False,
        'cluster_resources_changed': False,
        'definitions': {
            'correct_call': 'Frozen AST scorer: correct SDK function, retrieved membership, and verified selected arguments. No generated code execution.',
            'primary_refusal': 'Complete successful final answer, stripped of surrounding whitespace, equals bare NO_MATCHING_API.',
            'conditional_denominator': 'API-successful requests only. Incomplete successful generations remain in the denominator but never count as refusals.',
            'unknown_api': 'Named operation absent from the pinned Kubernetes 35.0.0 SDK / Kubernetes v1.35.2 REST registry. Not a claim about all third-party APIs.',
            'real_but_unavailable': 'Real operation absent from the three offered descriptions: removed correct API, still indexed but not retrieved, or outside the original 30. Does not mean disabled in the cluster.',
            'format_vs_identity': 'API identity is classified independently of literal-call format; ambiguous client bindings are unresolved, not invented APIs.',
        },
        'limitations': [
            'Ten coverage-selected queries, fixed before withheld runs; not a random sample. Existing baseline outcomes were already available.',
            'Withheld gold retrieval accuracy is zero by construction, not an estimate of normal retrieval quality.',
            'Qwen output cap differs by 23 tokens after pre-generation context counting; other settings and the prompt template are unchanged.',
            'Baseline prompts are reconstructed exactly and checked against recorded hashes. Baseline provider HTTP envelopes and raw reasoning were not retained; only actual saved final answers are available.',
            'One attempt per query/model; no automatic retry, model fallback, or fabricated replacement for API errors.',
        ],
    }
    baseline_prompt_hashes = {}
    withheld_prompt_hashes = {}
    for short, stem, model in MODELS:
        base = load(root / 'data/evaluation' / (stem + '_generation.json'))
        scored = load(root / 'data/evaluation' / (stem + '_evaluation.json'))
        score_map = {r['request_id']: r for r in scored['rows']}
        if len(base['records']) != len(prompt_map) or {r['request_id'] for r in base['records']} != set(prompt_map):
            raise ValueError('Incomplete baseline')
        base_rows, trace = [], []
        baseline_prompt_hashes[model] = {}
        for record in base['records']:
            rid = record['request_id']; prompt = prompt_map[rid]
            if _hash(prompt['prompt']) != record['prompt_sha256'] or record['retrieved_apis'] != prompt['retrieved_apis']:
                raise ValueError('Baseline prompt reconstruction mismatch')
            baseline_prompt_hashes[model][rid] = record['prompt_sha256']
            trial = {'removed_api': None, 'index_api_ids': all_ids, 'retrieved_apis': record['retrieved_apis']}
            taxonomy = classify_response(record, trial, apis, registry)
            taxonomy['correct_call'] = score_map[rid]['correct_call']
            base_rows.append(taxonomy)
            item = {
                'request_id': rid, 'prompt': prompt['prompt'],
                'prompt_sha256': prompt['prompt_sha256'], 'prompt_utf8_sha256': prompt['prompt_utf8_sha256'],
                'retrieved_apis': record['retrieved_apis'],
                'response_text': record.get('text') if record['api_status'] == 'OK' else None,
                'response_provenance': 'Actual saved final text; raw provider HTTP envelope/reasoning was not retained.',
                'record': record, 'score': score_map[rid], 'taxonomy': taxonomy,
            }
            trace.append(item)
            if not score_map[rid]['correct_call']:
                failures.append({'condition': 'baseline', 'model': model, **item,
                                 'expected_reference': answer_map[rid]})
        summary['baseline'][model] = {**scored['summary'], 'refusal_and_identity': refusal_summary(base_rows)}
        summary['model_settings']['baseline'][model] = base['generation_config']
        write_json(folder / (short + '_baseline_trace.json'), {
            'model': model, 'generation_config': base['generation_config'],
            'prompt_provenance': 'Reconstructed from preserved inputs and verified against every original recorded prompt hash.',
            'records': trace,
        }, forbidden_values)
        run = load(folder / (short + '_generation.json'))
        if run['generation_config'] != plan['model_settings'][model]:
            raise ValueError('Withheld settings differ from the amended frozen plan')
        withheld_prompt_hashes[model] = {}
        for record in run['records']:
            trial = trial_map[record['request_id']]
            if record['prompt'] != trial['prompt'] or record['prompt_sha256'] != _hash(trial['prompt']) or record['retrieved_apis'] != trial['retrieved_apis']:
                raise ValueError('Withheld prompt mismatch')
            withheld_prompt_hashes[model][record['request_id']] = record['prompt_sha256']
        evaluated = evaluate_refusal(run, trials, answers, apis, registry)
        write_json(folder / (short + '_evaluation.json'), evaluated, forbidden_values)
        summary['withheld'][model] = evaluated['summary']
        row_map = {r['request_id']: r for r in evaluated['rows']}
        for record in run['records']:
            rid = record['request_id']; row = row_map[rid]
            if not row['exact_refusal']:
                failures.append({
                    'condition': 'withheld', 'model': model, 'request_id': rid,
                    'prompt': record['prompt'], 'prompt_sha256': record['prompt_sha256'],
                    'retrieved_apis': record['retrieved_apis'], 'removed_api': trial_map[rid]['removed_api'],
                    'response_text': record.get('exact_response_text'),
                    'response_provenance': 'Actual provider final content; null for API errors.',
                    'record': record, 'taxonomy': row, 'expected_output': 'NO_MATCHING_API',
                    'excluded_reference': answer_map[rid],
                })
    if len({_hash(x) for x in baseline_prompt_hashes.values()}) != 1 or len({_hash(x) for x in withheld_prompt_hashes.values()}) != 1:
        raise ValueError('Models received different prompts within a condition')
    summary['identical_prompts_between_models_per_condition'] = True
    write_json(folder / 'failure_examples.json', {
        'selection': 'All observed baseline non-correct calls and all withheld non-exact-refusals, including API errors. No illustrative or fabricated outputs.',
        'count': len(failures), 'examples': failures,
    }, forbidden_values)
    write_json(folder / 'summary.json', summary, forbidden_values)
    return summary
