"""Command-line RAG query against a built retrieval index."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("--index", type=Path, default=ROOT / "data/retrieval/index-v2")
    p.add_argument("--strategy", choices=["bm25", "dense", "hybrid", "hybrid_rerank"], default="hybrid")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidate-k",type=int,default=40)
    p.add_argument("--output",type=Path,help='Save the complete contexts, prompt and answer trace')
    p.add_argument("--reranker-model", default=None)
    p.add_argument("--ask", action="store_true", help="run generation (needs HYBRIDRAG_LLM_*)")
    a = p.parse_args()

    warnings.filterwarnings("ignore")
    from hybrid_rag.retrieval.units import RetrievalUnit
    from hybrid_rag.retrieval.embeddings import EmbeddingModel, CrossEncoderReranker
    from hybrid_rag.retrieval.engine import RetrievalEngine
    from hybrid_rag.retrieval.generation import generate, make_chat_model

    from hybrid_rag.retrieval.engine import load_index
    if a.output and a.output.exists():raise FileExistsError(a.output)
    engine,meta=load_index(a.index,a.candidate_k)

    try:
        reranker = CrossEncoderReranker(a.reranker_model) if a.reranker_model else None
        results = engine.retrieve(a.query,strategy=a.strategy,top_k=a.top_k,reranker=reranker if a.strategy=='hybrid_rerank' else None,candidate_k=a.candidate_k)

        for r in results:
            print(f"#{r['rank']} [{r['strategy'] if 'strategy' in r else a.strategy}] score={r['score']:.3f}")
            print(f"    source: {r['source_path']}  {' > '.join(r['heading_path'])}")
            print(f"    section range: [{r['section_start']}, {r['section_end']})")
            print("    " + r["text"][:200].replace("\n", " ") + ("..." if len(r["text"]) > 200 else ""))

        if a.ask:
            from hybrid_rag.retrieval.generation import answer_question
            out=answer_question(a.query,results)
            print("\n--- Answer ---")
            print(json.dumps(out,ensure_ascii=True,indent=2))
        else:out=dict(contexts=results)
        if a.output:
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps(dict(question=a.query,index_metadata=meta,**out),ensure_ascii=False,indent=2),encoding='utf-8')
        engine.close()
    finally:engine.close()



if __name__ == "__main__":
    main()
