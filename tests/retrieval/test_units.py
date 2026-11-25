"""Unit tests for retrieval-unit building and the retrieval engine (no model I/O)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from hybrid_rag.retrieval.units import (
    RetrievalUnit,
    build_units,
    iter_blocks,
    split_content,
)


def _word_count(text: str) -> int:
    # Cheap deterministic tokenizer stand-in: one token per whitespace-separated word.
    return len(text.split())


def _chunk(chunk_id, section_id, content, heading_path, section_start=0, section_end=0):
    return {
        "chunk_id": chunk_id, "doc_id": "doc-1", "section_id": section_id,
        "source_path": "oss/langchain/tools.mdx", "heading_path": heading_path,
        "content": content, "embedding_text": " > ".join(heading_path) + "\n\n" + content,
        "section_start": section_start, "section_end": section_end or section_start+len(content),
    }


def test_iter_blocks_tiles_exactly():
    text = "para one line\n\n\n```python\nprint(1)\nprint(2)\n```\n\ntail line\n"
    blocks = iter_blocks(text)
    assert "".join(b for _, _, b in blocks) == text
    assert all(b == text[s:e] for s, e, b in blocks)


def test_build_units_rejects_oversized_and_preserves_short_records():
    with pytest.raises(ValueError,match='refine the ingestion export'):
        build_units([_chunk('c1','s1','word '*200,['Heading'])],_word_count,40)
    chunk=_chunk('c1','s1','unchanged text',['Heading'])
    unit,=build_units([chunk],_word_count,40)
    assert unit.unit_id==chunk['chunk_id'] and unit.text==chunk['embedding_text']


def test_build_units_preserves_noncontiguous_parts():
    chunk=_chunk('c1','s1','abcdef',['H'],section_start=0,section_end=13)
    chunk['content_parts']=[dict(start=0,end=3,section_start=0,section_end=3),dict(start=3,end=6,section_start=10,section_end=13)]
    unit,=build_units([chunk],_word_count)
    assert unit.ranges==[[0,3],[10,13]]
    assert len(unit.content_parts)==2


def test_split_content_offsets():
    text = "one two three\n\n" + "x " * 100 + "\n\nfour five six"
    pieces = split_content(text, 8, _word_count)
    assert "".join(t for t, _, _ in pieces) == text
    for t, s, e in pieces:
        assert text[s:e] == t
        assert _word_count(t) <= 8


def test_hard_split_single_oversized_line():
    text = " ".join(["tok"] * 50)  # one line, 50 tokens
    pieces = split_content(text, 12, _word_count)
    assert all(_word_count(t) <= 12 for t, _, _ in pieces)
    assert "".join(t for t, _, _ in pieces) == text


class _FakeEmbedding:
    dim = 4

    def encode(self, texts, batch_size=128, normalize=True):
        out = []
        for t in texts:
            v = np.zeros(4, dtype=np.float32)
            v[hash(t) % 4] = 1.0
            out.append(v)
        return np.array(out)


def test_engine_build_and_retrieve_dedup():
    from pathlib import Path

    from hybrid_rag.retrieval.engine import RetrievalEngine

    content = "checkpointer stores graph state\n\nstore holds cross thread data\n\n" + "\n\n".join(f"unrelated {i}" for i in range(200))
    chunks = [_chunk("c1", "s1", content, ["Persistence"])]
    units=build_units([_chunk(f'c{i}','s1',f'checkpointer store persistence {i}',['Persistence'],section_start=i*40) for i in range(25)],_word_count,max_tokens=20)
    assert len(units)>10

    embedding = _FakeEmbedding()
    index_dir = Path("index") / "_test_engine_qdrant"
    shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    try:
        engine = RetrievalEngine(units, embedding, index_dir, top_k=10)
        engine.build()

        for strategy in ("bm25", "dense", "hybrid"):
            res = engine.retrieve("checkpointer store", strategy=strategy, top_k=5)
            assert res, strategy
            ids = [r["unit_id"] for r in res]
            assert len(ids) == len(set(ids)), f"{strategy} produced duplicates"
    finally:
        if engine._qdrant:engine.close()
        shutil.rmtree(index_dir, ignore_errors=True)
