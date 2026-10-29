# Gorilla — Kubernetes API pilot

An experiment with **30 Kubernetes v1.35.2 APIs** and **20 requests**:
TF-IDF + cosine similarity → top-3 APIs → Gemma/Qwen → AST evaluation.
This adapts Gorilla's ideas; it **does not reproduce fine-tuning**.

## Run the notebook (Python 3.12)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab notebooks/gorilla_reproduction.ipynb
```

By default, the notebook uses saved responses without calling models or changing
the cluster. `RUN_NEW_REQUESTS=True` sends only requests without saved results.
New requests require a Google key in `credentials` and a working vLLM connection.

## Main files

- `data/apis.json`, `data/queries.json`: API catalog and requests.
- `data/evaluation/answers.json`: reference answers, used only for evaluation.
- [Results and evidence](data/evaluation/refusal/README.md): metrics, prompts, responses, and actual failures.
- `deploy/vllm/`, `notebooks/vllm_connection.local.json`: deployment and connection settings used for the run; adjust them for another cluster.
- `slides/`: the 12-slide presentation, published as PDF and PowerPoint only.

The original cluster-validation records were deleted and could not be fully
recovered. Recorded conclusions remain, but the repository lacks the original
evidence needed to independently verify those checks. Metadata cleanup did not
change the requests or measured results ([details](data/evaluation/publication_cleanup.json)).

`credentials` and `llm.jsonl` are excluded from GitHub.
