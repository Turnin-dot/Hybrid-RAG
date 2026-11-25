"""Step 1: verify BGE-base-en-v1.5 token lengths for every chunk.

Measures complete ``embedding_text`` with the selected local model tokenizer.
The retrieval adapter preserves bounded Chunks one-to-one and rejects overflow;
it does not implement another splitting pipeline.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", type=Path, default=ROOT / "data/processed/hard512-baseline-v2/chunks.jsonl")
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--output", type=Path, default=ROOT / "data/eval/runs/token_lengths_v2.json")
    a = p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)

    warnings.filterwarnings("ignore")
    from hybrid_rag.retrieval.embeddings import EmbeddingModel
    from hybrid_rag.retrieval.units import build_units

    model = EmbeddingModel(a.model)
    tokenizer = model.tokenizer
    def count(t: str) -> int:
        return len(tokenizer(t, add_special_tokens=True,truncation=False)["input_ids"])

    chunks = [json.loads(l) for l in a.chunks.read_text(encoding="utf-8").splitlines()]
    lens = [count(c["embedding_text"]) for c in chunks]
    over = [l for l in lens if l > a.max_tokens]
    units = build_units(chunks, count, a.max_tokens)
    unit_lengths=[count(u.text) for u in units]
    if any(n>a.max_tokens for n in unit_lengths):raise ValueError('Unit exceeds budget')
    split_chunks = [c["chunk_id"] for c in chunks if count(c["embedding_text"]) > a.max_tokens]

    def pct(q: float) -> int:
        return int(sorted(lens)[int((len(lens) - 1) * q)])

    report = {
        "model": a.model,
        "embedding_identity": model.identity,
        "chunks_sha256": hashlib.sha256(a.chunks.read_bytes()).hexdigest(),
        "max_tokens": a.max_tokens,
        "chunks": len(chunks),
        "over_limit_chunks": len(over),
        "over_limit_ratio": round(len(over) / len(chunks), 4) if chunks else 0,
        "max_tokens_observed": max(lens),
        "token_percentiles": {f"p{q}": pct(q / 100) for q in (50, 90, 95, 99, 100)},
        "oversized_flag_count": sum(1 for c in chunks if c.get("oversized")),
        "over_limit_and_oversized": sum(1 for c, l in zip(chunks, lens) if l > a.max_tokens and c.get("oversized")),
        "units_after_split": len(units),
        "output_over_limit":sum(n>a.max_tokens for n in unit_lengths),
        "output_maximum":max(unit_lengths,default=0),
        "split_added_units": len(units) - len(chunks),
        "split_chunk_ids": split_chunks,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
