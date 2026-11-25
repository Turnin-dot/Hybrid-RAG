# HybridRAG

A Python technical-document RAG project with static MDX cleaning, source mapping,
structure-aware chunking, hard embedding-token limits, hybrid retrieval, reranking,
evidence-constrained generation, and explicit evaluation metrics.

The public repository contains code and hand-authored synthetic test fixtures.
It does **not** distribute upstream documentation, processed corpora, evaluation
quotes, model weights, indexes, API traces, human answers, or private credentials.

## Components

- `src/hybrid_rag/ingestion`: Loader, static MDX Cleaner, Section Parser, baseline
  and improved Chunkers, tokenizer refinement, provenance and validation.
- `src/hybrid_rag/retrieval`: BGE encoding, BM25, local Qdrant cosine retrieval,
  LlamaIndex RRF, Cross-Encoder reranking, context selection and generation.
- `src/hybrid_rag/evaluation`: Section/evidence metrics, grounded Judge,
  refusal policy, calibration and paired statistics.
- `scripts`: ingestion, indexing, query/API, evaluation and review entry points.
- `tests`: synthetic unit tests plus explicitly gated external-asset regressions.

LangChain is used for splitting and model-provider adapters; LlamaIndex is used
for BM25/RRF. The project does not implement a complex Agent orchestration system.

## Install

Use Python 3.11+ in a fresh virtual environment. Commands run from repository root:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -e ".[dev,analysis,retrieval,api,generation]"
python -m pytest -q -rs --basetemp .pytest_tmp
```

Dependencies have compatibility ranges in `pyproject.toml`; this is not a fully
locked cross-platform environment. Select a PyTorch/CUDA build compatible with
your GPU. For source-only ingestion tests, installing `.[dev]` is sufficient;
the entire test suite needs the optional dependencies listed above.

Corpus, model and historical-run integration tests are skipped **with explicit
reasons** if their assets are absent. This does not claim reproduction of the
original full-workspace test count or any real LLM quality measurement.

## Source-only smoke test

The checked-in MDX fixture is authored for tests, not an upstream document copy.

```sh
python scripts/smoke_ingestion.py --output data/processed/synthetic-smoke
```

This produces `documents.jsonl`, `sections.jsonl`, `chunks.jsonl` and
`ingestion_report.json`. Use a fresh output directory on every invocation.

## Bring your own documentation

The original experiments used the LangChain documentation snapshot below.
Fetch it independently and comply with its upstream license; do not commit it:

```sh
git clone https://github.com/langchain-ai/docs.git docs
git -C docs checkout 04d4d4a45cb57537c3d6e977f25333f2c5a4881b
python scripts/prepare_data.py --docs-root docs/src --output data/processed/phase1-v3-final
```

`docs/`, including its independent Git history, is ignored. The Cleaner reads
trusted static files; it never executes MDX, JavaScript or source code blocks.
Unknown syntax produces warnings/placeholders. Review ingestion findings rather
than treating process exit success as proof of complete semantic fidelity.

## Prepare models and the hard-token export

Download models explicitly before local-only loading. The default cache uses the
user's Hugging Face cache; set `HF_HOME` in your shell to choose another location.

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-base-en-v1.5'); snapshot_download('BAAI/bge-reranker-base')"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5').save_pretrained('data/models/bge-base-en-v1.5-tokenizer')"
python scripts/prepare_data.py --from-export data/processed/phase1-v3-final --tokenizer data/models/bge-base-en-v1.5-tokenizer --target-tokens 384 --max-tokens 512 --token-overlap 32 --output data/processed/hard512-baseline-v2
python scripts/build_retrieval_index.py --chunks data/processed/hard512-baseline-v2/chunks.jsonl --output data/retrieval/index-v2 --top-k 40
```

Complete `embedding_text`, including headings and special tokens, must be at most
512 tokens. In-budget chunks remain unchanged. Over-budget chunks are refined
with source-order/coverage checks; overlap is labelled. No silent truncation.
The local Qdrant/BM25 index is versioned against model and source hashes, and
existing indexes are not overwritten.

## Query and API

```sh
python scripts/query_rag.py "How do checkpoints work?" --strategy hybrid_rerank --candidate-k 40 --top-k 5 --reranker-model BAAI/bge-reranker-base
python scripts/query_rag.py "How do checkpoints work?" --strategy hybrid_rerank --candidate-k 40 --top-k 5 --reranker-model BAAI/bge-reranker-base --ask
python scripts/serve_api.py
```

API routes: `GET /health`, `POST /retrieve`, `POST /ask`. Example request:

```json
{"query":"How do checkpoints work?","strategy":"hybrid_rerank","candidate_k":40,"top_k":5}
```

Export the variables documented in `.env.example` in your shell. The file is a
template and is **not automatically loaded** by all entry points. Never put a
real key in the template. Generation uses `DEEPSEEK_API_KEY`; without configuration,
CLI `--ask` returns contexts/prompt as a dry run. Real LLM calls may incur fees.
Reranking requires `HYBRIDRAG_RERANKER_MODEL` for the API. API/CLI defaults are
Hybrid; select `hybrid_rerank` explicitly. All 40 fused candidates are reranked
before selecting Top5. Citation substring validation is separate from semantic
support/faithfulness. See [citation retries](CITATION_RETRIES.md).

## Evaluation and reproducibility limits

The metric and generic evaluation runners are included. Supply your own reviewed
dataset using `src/hybrid_rag/evaluation/schema.py` as its contract. Questions,
facts and evidence ranges must match your own frozen corpus. No evaluation dataset
or historical outputs are bundled. The 120-question historical seed module
`scripts/eval_topics.py` contains document quotations/anchors and is intentionally
excluded; `build_eval_dataset.py` requires that local-only module. Frozen Test80,
saved-response recovery, historical auditing and exact old-score replay scripts
require the corresponding local manifests/assets and are not fresh-clone demos.

The historical internal project used 169 documents and 5,404 baseline chunks.
On 74 answerable Test questions, Hybrid to Hybrid+Rerank improved complete required
fact evidence coverage@5 from 74.32% to 85.14%, and Section-based MRR@10 from .681
to .797. These are historical internal observations, not reproduced public
benchmark results. Gold candidate review, small dataset size and related topic
variants limit generalization. Do not equate evidence coverage with answer accuracy.

For a new experiment, freeze corpus and dataset versions; compare BM25, Dense,
Hybrid and Hybrid+Rerank using the same candidate pool. Save original ranked
chunks, context ranges, prompts, usage and failure records locally. Judge labels
require human calibration. Never commit these records: they may contain source
text, credentials echoed by providers, or user input.

No root Git history, upstream documentation history, personal author identity,
or historical experiment files are included in the publication copy. A project
license must be chosen by the owner before describing it as openly licensed.
