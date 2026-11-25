"""Retrieval engine: BM25, BGE dense (Qdrant), LlamaIndex RRF and reranking.

BM25 and Dense operate on the *same* retrieval units and stable ``unit_id``.
The dense index is persisted with Qdrant's local mode; the BM25 index is
rebuilt in memory from the unit list.  Hybrid fusion is delegated to
LlamaIndex ``QueryFusionRetriever`` with ``num_queries=1`` so no LLM query
rewrite is introduced.
"""
from __future__ import annotations

import hashlib
import json
import uuid
import threading
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .embeddings import CrossEncoderReranker, EmbeddingModel
from .units import RetrievalUnit

COLLECTION = "units"


def _point_id(unit_id: str) -> str:
    """Qdrant local mode requires UUID or uint point ids; derive one deterministically."""
    return str(uuid.UUID(hex=hashlib.sha256(unit_id.encode("utf-8")).hexdigest()[:32]))


def _make_node(unit_id: str, text: str, section_id: str, source_path: str):
    """Build a LlamaIndex TextNode with a canonical metadata dict.

    BM25 and Dense nodes for the same unit must share identical content and
    metadata, otherwise QueryFusionRetriever's node-hash dedup treats them as
    different results and emits duplicates.
    """
    from llama_index.core.schema import TextNode
    return TextNode(id_=unit_id, text=text, metadata={
        "unit_id": unit_id, "section_id": section_id, "source_path": source_path,
    })


class _DenseRetriever:
    """Minimal adapter exposing Qdrant dense results as LlamaIndex nodes."""

    def __init__(self, qdrant: QdrantClient, embedding: EmbeddingModel, top_k: int):
        self.qdrant = qdrant
        self.embedding = embedding
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        from llama_index.core.schema import QueryBundle
        vec = self.embedding.encode([query])[0].tolist()
        resp = self.qdrant.query_points(
            COLLECTION, query=vec, limit=self.top_k, with_payload=True,
        )
        out = []
        for p in resp.points:
            out.append({"unit_id": p.payload["unit_id"], "score": float(p.score), "payload": p.payload})
        return out


class _DenseLlamaRetriever:
    """Wrap dense results so LlamaIndex QueryFusionRetriever can fuse them."""

    def __init__(self, dense: _DenseRetriever):
        self.dense = dense

    def retrieve(self, query_bundle) -> list:
        from llama_index.core.schema import NodeWithScore
        nodes = []
        for item in self.dense.retrieve(query_bundle.query_str):
            p = item["payload"]
            node = _make_node(p["unit_id"], p["text"], p["section_id"], p["source_path"])
            nodes.append(NodeWithScore(node=node, score=item["score"]))
        return nodes


class RetrievalEngine:
    def __init__(self, units: list[RetrievalUnit], embedding: EmbeddingModel,
                 index_dir: str | Path, top_k: int = 10):
        self.units = units
        self.embedding = embedding
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self._by_id = {u.unit_id: u for u in units}
        self._bm25 = None
        self._qdrant: QdrantClient | None = None
        self._dense = None
        self._lock=threading.RLock()
        if len(self._by_id)!=len(units):raise ValueError('Duplicate unit IDs')

    def _manifest(self):
        return dict(format_version=2,units_sha256=hashlib.sha256(''.join(u.model_dump_json()+'\n' for u in self.units).encode()).hexdigest(),
            embedding=getattr(self.embedding,'identity',dict(test_model=type(self.embedding).__qualname__,dimension=self.embedding.dim)),units=len(self.units))

    # ------------------------------------------------------------------ build
    def build(self, batch_size: int = 128) -> dict:
        from llama_index.core.schema import TextNode
        from llama_index.retrievers.bm25 import BM25Retriever

        self.index_dir.mkdir(parents=True, exist_ok=True)
        nodes = [_make_node(u.unit_id, u.text, u.section_id, u.source_path) for u in self.units]
        self._bm25 = BM25Retriever(nodes=nodes, similarity_top_k=self.top_k, language="en")

        self._qdrant = QdrantClient(path=str(self.index_dir))
        if self._qdrant.collection_exists(COLLECTION):
            self._qdrant.close();raise FileExistsError('Refusing to overwrite an existing collection')
        self._qdrant.create_collection(
            COLLECTION, vectors_config=VectorParams(size=self.embedding.dim, distance=Distance.COSINE),
        )
        for start in range(0, len(self.units), batch_size):
            batch = self.units[start:start + batch_size]
            vecs = self.embedding.encode([u.text for u in batch], batch_size=batch_size)
            points = [
                PointStruct(id=_point_id(u.unit_id), vector=vec.tolist(), payload=self._payload(u))
                for u, vec in zip(batch, vecs)
            ]
            self._qdrant.upsert(COLLECTION, points)
        self._dense = _DenseRetriever(self._qdrant, self.embedding, self.top_k)
        (self.index_dir/'engine_manifest.json').write_text(json.dumps(self._manifest(),sort_keys=True,indent=2),encoding='utf-8')
        return {"units": len(self.units), "bm25_nodes": len(nodes),
                "qdrant_collection": COLLECTION, "qdrant_points": self._qdrant.count(COLLECTION).count}

    def load(self) -> dict:
        """Reopen the persisted dense index and rebuild BM25 in memory (no re-encode)."""
        from llama_index.retrievers.bm25 import BM25Retriever

        manifest=self.index_dir/'engine_manifest.json'
        if not manifest.exists():raise ValueError('Legacy/unversioned index; build a separate versioned index')
        if json.loads(manifest.read_text(encoding='utf-8'))!=self._manifest():raise ValueError('Index units/model/version mismatch')

        nodes = [_make_node(u.unit_id, u.text, u.section_id, u.source_path) for u in self.units]
        self._bm25 = BM25Retriever(nodes=nodes, similarity_top_k=self.top_k, language="en")
        self._qdrant = QdrantClient(path=str(self.index_dir))
        config=self._qdrant.get_collection(COLLECTION)
        if config.config.params.vectors.size!=self.embedding.dim or config.config.params.vectors.distance!=Distance.COSINE or self._qdrant.count(COLLECTION).count!=len(self.units):
            self._qdrant.close();raise ValueError('Index vector dimension/count mismatch')
        self._dense = _DenseRetriever(self._qdrant, self.embedding, self.top_k)
        return {"units": len(self.units), "bm25_nodes": len(nodes),
                "qdrant_collection": COLLECTION, "qdrant_points": self._qdrant.count(COLLECTION).count}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _payload(u: RetrievalUnit) -> dict:
        return {**u.model_dump(),"ranges":u.ranges,**{
            "unit_id": u.unit_id, "chunk_id": u.chunk_id, "doc_id": u.doc_id,
            "section_id": u.section_id, "source_path": u.source_path,
            "heading_path": u.heading_path, "text": u.text,
            "section_start": u.section_start, "section_end": u.section_end,
            "split_index": u.split_index, "parent_count": u.parent_count,
        }}

    def _result(self, unit_id: str, score: float, rank: int) -> dict:
        u = self._by_id[unit_id]
        return {"doc_id":u.doc_id,"ranges":u.ranges,"content_parts":[p.model_dump() for p in u.content_parts],
            "source_spans":[s.model_dump() for s in u.source_spans],**{
            "unit_id": u.unit_id, "chunk_id": u.chunk_id, "section_id": u.section_id,
            "source_path": u.source_path, "heading_path": u.heading_path, "text": u.text,
            "section_start": u.section_start, "section_end": u.section_end,
            "score": score, "rank": rank,
        }}

    def _bm25_retrieve(self, query: str, top_k: int) -> list[dict]:
        from llama_index.core.schema import QueryBundle
        nodes = self._bm25.retrieve(QueryBundle(query_str=query))
        return [self._result(n.node.id_, float(n.score), i + 1) for i, n in enumerate(nodes[:top_k])]

    def _dense_retrieve(self, query: str, top_k: int) -> list[dict]:
        items = self._dense.retrieve(query)[:top_k]
        return [self._result(i["unit_id"], i["score"], r + 1) for r, i in enumerate(items)]

    def _hybrid_retrieve(self, query: str, top_k: int) -> list[dict]:
        from llama_index.core import Settings
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

        # num_queries=1 means no LLM query generation; disable the LLM entirely so
        # query rewriting is never introduced into the hybrid baseline.
        Settings.llm = None
        fusion = QueryFusionRetriever(
            [self._bm25, _DenseLlamaRetriever(self._dense)],
            mode=FUSION_MODES.RECIPROCAL_RANK,
            similarity_top_k=top_k, num_queries=1, use_async=False,
        )
        from llama_index.core.schema import QueryBundle
        nodes = fusion.retrieve(QueryBundle(query_str=query))
        return [self._result(n.node.id_, float(n.score), i + 1) for i, n in enumerate(nodes)]

    # --------------------------------------------------------------- retrieve
    def retrieve(self, query: str, strategy: str = "hybrid", top_k: int | None = None,
                 reranker: CrossEncoderReranker | None = None, candidate_k: int | None = None) -> list[dict]:
        with self._lock:return self._retrieve(query,strategy,top_k,reranker,candidate_k)

    def _retrieve(self, query, strategy, top_k, reranker, candidate_k):
        k = self.top_k if top_k is None else top_k
        pool=self.top_k if candidate_k is None else candidate_k
        if k<1 or pool<k:raise ValueError('Require candidate_k >= top_k >= 1')
        if not query.strip():raise ValueError('Empty query')
        if strategy=='hybrid_rerank' and reranker is None:raise ValueError('hybrid_rerank requires a configured Reranker')
        # Supplying a cached reranker never changes the explicitly named strategy.
        use_rerank=strategy=='hybrid_rerank'
        self._bm25.similarity_top_k=pool
        self._dense.top_k=pool
        if strategy == "bm25":
            results = self._bm25_retrieve(query, pool)
        elif strategy == "dense":
            results = self._dense_retrieve(query, pool)
        elif strategy in {"hybrid","hybrid_rerank"}:
            results = self._hybrid_retrieve(query, pool)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Defensive dedup by unit_id (the fusion dedups by node hash; keep it exact).
        seen = set()
        deduped = []
        for r in results:
            if r["unit_id"] in seen:
                continue
            seen.add(r["unit_id"])
            deduped.append(r)
        results = deduped

        if use_rerank:
            cand = results[:pool]
            order = reranker.rerank(query, [r["text"] for r in cand])
            results = [dict(cand[i],reranker_score=score) for i,score in order]
        results=results[:k]
        for i, r in enumerate(results, 1):
            r["rank"] = i
        return results

    def close(self):
        if self._qdrant:self._qdrant.close()


def load_index(directory, candidate_k=40):
    """One version-checked load path shared by API, CLI and experiments."""
    directory=Path(directory)
    meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('format_version')!=2 or not {'model','embedding_identity','units_sha256','chunks_source','chunks_sha256'}<=meta.keys():
        raise ValueError('Legacy/incomplete index manifest; build a separate versioned index')
    unit_file=directory/'units.jsonl'
    if hashlib.sha256(unit_file.read_bytes()).hexdigest()!=meta['units_sha256']:raise ValueError('Persisted units file changed')
    source=Path(meta['chunks_source'])
    if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=meta['chunks_sha256']:raise ValueError('Frozen Chunk source changed/missing')
    embedding=EmbeddingModel(meta['model'])
    if embedding.identity!=meta['embedding_identity']:raise ValueError('Embedding checkpoint/Tokenizer changed')
    units=[RetrievalUnit.model_validate_json(l) for l in unit_file.read_text(encoding='utf-8').splitlines()]
    engine=RetrievalEngine(units,embedding,directory/'qdrant',top_k=candidate_k)
    engine.load()
    return engine,meta
