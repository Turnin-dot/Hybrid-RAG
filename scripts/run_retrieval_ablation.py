"""Step 5: A/B/C/D retrieval ablation against the frozen English eval set.

A = BM25, B = BGE Dense, C = BM25+Dense RRF, D = C + Cross-Encoder rerank.
Metrics use the existing evidence-grounded evaluation module (Recall@K, MRR,
nDCG, Hit, Required-Fact coverage) at Section/evidence granularity, with the
same retrieval units and stable IDs for every strategy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def units_to_ranked(results: list[dict]) -> list[dict]:
    """Keep every original rank; metric code handles unique Section gains."""
    return [dict(section_id=r['section_id'],ranges=r['ranges'] if 'ranges' in r else [[r['section_start'],r['section_end']]]) for r in results]



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data/eval/hybridrag-en-v2")
    p.add_argument("--split", choices=["dev", "test", "pilot30"], default="pilot30")
    p.add_argument("--index", type=Path, default=ROOT / "data/retrieval/index-v2")
    p.add_argument("--strategies", default="bm25,dense,hybrid")
    p.add_argument("--candidate-k", type=int, default=40)
    p.add_argument("--k-list", default="5,10")
    p.add_argument("--reranker-model", default=None)
    p.add_argument("--reranker-candidate-k", type=int, default=None)
    p.add_argument("--output", type=Path, default=ROOT / "runs/retrieval_ablation.json")
    p.add_argument("--limit", type=int, default=0, help="cap questions for a fast smoke run (0 = all)")
    p.add_argument('--allow-unreviewed',action='store_true',help='Explicit diagnostic mode; never official gold scoring')
    a = p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.reranker_candidate_k is not None and a.reranker_candidate_k!=a.candidate_k:
        p.error('A/B/C/D must use one fixed candidate pool; use --candidate-k for all strategies')

    warnings.filterwarnings("ignore")
    from hybrid_rag.evaluation.dataset import load_questions
    from hybrid_rag.evaluation.retrieval import retrieval_metrics
    from hybrid_rag.retrieval.units import RetrievalUnit
    from hybrid_rag.retrieval.embeddings import EmbeddingModel, CrossEncoderReranker
    from hybrid_rag.retrieval.engine import RetrievalEngine

    questions = [q for q in load_questions(a.dataset, a.split, allow_unreviewed=a.allow_unreviewed) if q.answerable]
    if a.limit:
        questions = questions[:a.limit]
    if not questions:
        raise ValueError("No answerable questions in split")

    from hybrid_rag.retrieval.engine import load_index
    from hybrid_rag.evaluation.statistics import bootstrap,cluster_bootstrap
    import hashlib
    engine,index_meta=load_index(a.index,a.candidate_k)
    embedding=engine.embedding
    a.output.parent.mkdir(parents=True,exist_ok=True)
    journal=a.output.with_suffix('.predictions.jsonl')
    if journal.exists():raise FileExistsError(journal)

    try:
        reranker = None
        strategies = a.strategies.split(",")
        if a.reranker_model:
            reranker = CrossEncoderReranker(a.reranker_model)
            strategies = [s for s in strategies if s != "hybrid_rerank"] + ["hybrid_rerank"]

        ks = [int(x) for x in a.k_list.split(",")]
        rows: list[dict] = []
        predictions=[]
        latency: dict[str, list[float]] = defaultdict(list)

        # Warm up the dense model once so latency is not dominated by lazy load.
        embedding.encode(["warmup"])
        if reranker:
            reranker.scores([("warmup", "warmup")])

        for qi, q in enumerate(questions, 1):
            for strategy in strategies:
                t0 = time.perf_counter()
                if strategy == "hybrid_rerank":
                    results = engine.retrieve(q.question, strategy="hybrid_rerank", top_k=a.candidate_k,
                                              reranker=reranker, candidate_k=a.candidate_k)
                else:
                    results = engine.retrieve(q.question, strategy=strategy, top_k=a.candidate_k)
                latency[strategy].append(time.perf_counter() - t0)
                predictions.append(dict(query_id=q.query_id,topic_group=q.topic_group,question=q.question,strategy=strategy,results=results,latency_s=latency[strategy][-1]))
                with journal.open('a',encoding='utf-8') as handle:handle.write(json.dumps(predictions[-1],ensure_ascii=False)+'\n')
                ranked = units_to_ranked(results)
                for k in ks:
                    m = retrieval_metrics(q, ranked, k)
                    rows.append({"query_id": q.query_id, "category": q.category, "topic_group":q.topic_group, "strategy": strategy, "k": k, **m})
            if qi % 10 == 0:
                print(f"  ... {qi}/{len(questions)}")

        # Aggregate per strategy and k.
        summary = {}
        for strategy in strategies:
            for k in ks:
                subset = [r for r in rows if r["strategy"] == strategy and r["k"] == k]
                agg = {}
                for metric in ["hit_section", "recall_section", "mrr", "ndcg", "evidence_hit",
                               "required_fact_coverage", "all_required_facts"]:
                    vals = [r[metric] for r in subset]
                    agg[metric] = sum(vals) / len(vals)
                agg['confidence_intervals']={key:dict(question=bootstrap([r[key] for r in subset]),topic_cluster=cluster_bootstrap(subset,key)) for key in agg}
                agg["n"] = len(subset)
                agg["latency_p50_s"] = sorted(latency[strategy])[int((len(latency[strategy]) - 1) * .5)]
                agg["latency_p95_s"] = sorted(latency[strategy])[int((len(latency[strategy]) - 1) * .95)]
                summary[f"{strategy}@{k}"] = agg

        report = {
            "status":"diagnostic_unreviewed" if a.allow_unreviewed else "scored_reviewed_labels",
            "dataset_version":questions[0].dataset_version,
            "dataset_sha256":hashlib.sha256((a.dataset/'questions.jsonl').read_bytes()).hexdigest(),
            "index_metadata":index_meta,"embedding":embedding.identity,"reranker_identity":reranker.identity if reranker else None,
            "reranker_candidate_k":a.candidate_k if reranker else None,
            "predictions":predictions,
            "failures":[r for r in rows if r['all_required_facts']<1],
            "implementation_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*list((ROOT/'src/hybrid_rag/retrieval').glob('*.py'))]},
            "dataset": str(a.dataset), "split": a.split, "index": str(a.index),
            "strategies": strategies, "candidate_k": a.candidate_k, "k_list": ks,
            "reranker_model": a.reranker_model, "answerable_questions": len(questions),
            "summary": summary, "per_query": rows,
        }
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('w',encoding='utf-8') as handle:json.dump(report,handle,indent=2)
        print(json.dumps({k:{m:v for m,v in r.items() if m!='confidence_intervals'} for k,r in summary.items()}, indent=2))
        print(f"report -> {a.output}")
    finally:engine.close()



if __name__ == "__main__":
    main()
