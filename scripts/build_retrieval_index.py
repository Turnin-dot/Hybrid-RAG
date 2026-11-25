"""Step 2/3: build retrieval units, BM25 index and Qdrant dense index.

The units are the *same* for BM25 and Dense.  Everything is persisted so the
ablation and query scripts reload a frozen index without re-encoding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, default=ROOT / "data/processed/hard512-baseline-v2/chunks.jsonl")
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--output", type=Path, default=ROOT / "data/retrieval/index-v2")
    p.add_argument("--top-k", type=int, default=40)
    a = p.parse_args()

    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer
    from hybrid_rag.retrieval.units import build_units, RetrievalUnit
    from hybrid_rag.retrieval.embeddings import EmbeddingModel
    from hybrid_rag.retrieval.engine import RetrievalEngine

    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"Output index directory is not empty: {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)

    embedding = EmbeddingModel(a.model)
    tokenizer = embedding.tokenizer
    def count(t: str) -> int:
        return len(tokenizer(t, add_special_tokens=True,truncation=False)["input_ids"])

    source_sha=sha(a.chunks)
    chunks = [json.loads(l) for l in a.chunks.read_text(encoding="utf-8").splitlines()]
    units = build_units(chunks, count, a.max_tokens)

    (a.output / "units.jsonl").write_text(
        "".join(RetrievalUnit.model_dump_json(u) + "\n" for u in units), encoding="utf-8",
    )

    engine = RetrievalEngine(units, embedding, a.output / "qdrant", top_k=a.top_k)
    info = engine.build()
    if sha(a.chunks)!=source_sha:
        engine.close();raise ValueError('Chunk source changed during index construction')

    meta = {
        "chunks_source": str(a.chunks),
        "chunks_sha256": source_sha,
        "model": a.model,
        "embedding_identity":embedding.identity,
        "units_sha256":sha(a.output/'units.jsonl'),
        "format_version":2,
        "max_tokens": a.max_tokens,
        "top_k": a.top_k,
        "units": len(units),
        "chunks": len(chunks),
        "split_added_units": len(units) - len(chunks),
        **info,
    }
    (a.output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    engine.close()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
