# Independent API-withholding experiment

The notebook evaluates both models on 10 fixed queries, independently rebuilding TF-IDF from the original 30 APIs minus that query's gold API. Each trial has 29 candidates and three retrieved descriptions. The original query and prompt template are reused; gold labels and exclusion metadata never enter model input.

Selected IDs: `req_002`, `req_004`, `req_007`, `req_009`, `req_010`, `req_012`, `req_013`, `req_014`, `req_015`, `req_019`. Selection covers nine resource types and create/read/list/patch operations, including scale. IDs were fixed before withheld runs; baseline results already existed. This is a small coverage sample, not a random or blinded sample.

## Measured results

Baseline retrieval: top-1 **10/20 (50%)**, top-3 **17/20 (85%)**.

| Model | Baseline correct calls | Baseline API errors | Withheld exact refusals | Among API successes | Withheld API errors |
|---|---:|---:|---:|---:|---:|
| gemma-4-31b-it | 3/20 (15%) | 16/20 | 1/10 (10%) | 1/1 (100%) | 9/10 |
| Qwen3-8B | 4/20 (20%) | 0/20 | 3/10 (30%) | 3/10 (30%) | 0/10 |

Gemma's withheld errors: eight HTTP 500s and one request failure with no HTTP status. Its single successful response is insufficient for a reliable comparison. Baseline correct-call rates among API successes are 3/4 and 4/20, respectively.

Qwen's other withheld outputs: two calls to the removed real API, one unknown API method, two wrong available APIs, and two invalid outputs. Every successful generation finished normally. No API error is counted as a refusal. Fence-normalized refusal counts equal the exact counts here.

## Definitions and settings

Primary refusal: complete successful final answer equals bare `NO_MATCHING_API` after trimming whitespace. All-attempt and API-success-only denominators are reported separately. Withheld gold retrieval is zero by construction; it is not normal retrieval accuracy.

Unknown API names are checked against 891 methods from the pinned Kubernetes Python SDK 35.0.0 and 1,067 REST operations from Kubernetes v1.35.2. Real but unavailable means missing from the three offered cards, subdivided into removed gold, indexed but unretrieved, or outside the original catalog. It does not imply the cluster disables that API. Syntax validity and API identity are separate checks; unresolved client bindings are not labeled invented APIs.

Temperature is 0 for both models. Gemma retains `maxOutputTokens=4096`. Qwen retains non-thinking mode; its fixed output cap changed from 3310 to 3287 because token counting found an 809-token prompt. The original plan is preserved. This adjustment preceded all withheld Qwen generation; there were no failed generation attempts or outcome-based changes. Maximum observed Qwen output was 133 tokens. No automatic retries or model substitutions occurred.

## Evidence and reproduction

- `plan.json`, `plan_before_context_preflight.json`, `qwen_context_preflight.json`: selection, settings, source hashes, and documented adjustment.
- `trials.json`: each full ranking, independent index IDs, IDF hash, and exact prompt.
- `gemma_generation.json`, `qwen_generation.json`: exact sent prompts, final responses, settings, HTTP outcomes, usage, and timing.
- `*_baseline_trace.json`, `baseline_prompts.json`: original saved baseline answers and prompts reconstructed exactly against all 40 recorded hashes. Original raw HTTP envelopes/reasoning were not retained and are not fabricated.
- `summary.json`, `retrieval_metrics.json`, `*_evaluation.json`: measured metrics and per-query AST classifications.
- `failure_examples.json`: all 49 observed baseline non-correct calls and withheld non-refusals, including API errors with null response text. No synthetic examples.
- `verification.json`: checks and artifact hashes. `originals/`: previous notebook and historical file hashes (before publication metadata cleanup).

Run `notebooks/gorilla_reproduction.ipynb` from the project environment. All model-request flags default to false and use recorded results. The 32 offline tests run with `python -m unittest discover -s notebooks -p 'test_*.py' -q`. Model inputs, reference-call scores, recorded responses, and baseline helpers remain unchanged. Unavailable-source metadata was cleaned; see ../publication_cleanup.json. Original raw cluster-check files are unavailable. No generated code was executed, no cluster resources were modified, and no credential values were saved.
