"""FastAPI service exposing the HybridRAG retrieval/generation stack."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="HybridRAG", version="0.2.0")

# Lazy-loaded globals so the module imports without loading the model.
_engine = None
_embedding = None
_reranker = None
_INDEX = Path(os.environ.get("HYBRIDRAG_INDEX", ROOT / "data/retrieval/index-v2"))
_RERANKER_MODEL = os.environ.get("HYBRIDRAG_RERANKER_MODEL") or None


class QueryRequest(BaseModel):
    query: str
    strategy: str = Field("hybrid", pattern="^(bm25|dense|hybrid|hybrid_rerank)$")
    top_k: int = Field(5, ge=1, le=100)
    candidate_k: int = Field(40,ge=1,le=200)


def _load():
    global _engine, _embedding, _reranker
    if _engine is None:
        import warnings
        warnings.filterwarnings("ignore")
        from hybrid_rag.retrieval.units import RetrievalUnit
        from hybrid_rag.retrieval.embeddings import EmbeddingModel, CrossEncoderReranker
        from hybrid_rag.retrieval.engine import RetrievalEngine

        from hybrid_rag.retrieval.engine import load_index
        engine,_=load_index(_INDEX)
        _engine=engine;_embedding=engine.embedding
        if _RERANKER_MODEL:
            _reranker = CrossEncoderReranker(_RERANKER_MODEL)
    return _engine


@app.get("/health")
def health():
    return {"status": "ok", "index": str(_INDEX)}


@app.post("/retrieve")
def retrieve(req: QueryRequest):
    engine = _load()
    try:
        results = engine.retrieve(req.query,strategy=req.strategy,top_k=req.top_k,candidate_k=req.candidate_k,
            reranker=_reranker if req.strategy=='hybrid_rerank' else None)
    except ValueError as exc:raise HTTPException(status_code=400,detail=str(exc)) from exc
    return {"query": req.query, "strategy": req.strategy, "results": results}


@app.post("/ask")
def ask(req: QueryRequest):
    results=retrieve(req)['results']
    from hybrid_rag.retrieval.generation import answer_question
    try:return dict(query=req.query,**answer_question(req.query,results))
    except Exception as exc:raise HTTPException(status_code=502,detail='Generation failed: '+type(exc).__name__) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_api:app", host="127.0.0.1", port=8000, reload=False)
