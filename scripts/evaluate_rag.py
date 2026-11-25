"""Evaluate saved results; no retriever, generator or Judge is invoked here."""
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_rag.evaluation.dataset import load_questions
from hybrid_rag.evaluation.retrieval import resolve_results,retrieval_metrics
from hybrid_rag.evaluation.generation import GenerationJudgment,score_generation
from hybrid_rag.evaluation.statistics import bootstrap,aggregate_runs,cluster_bootstrap
from hybrid_rag.schemas.document import Section,Chunk


def rows(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--mode',choices=['retrieval','generation'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split','--subset',choices=['dev','test','pilot30'],default='dev');p.add_argument('--allow-unreviewed',action='store_true')
    p.add_argument('--chunks',type=Path,default=ROOT/'data/processed/phase1-v3-final/chunks.jsonl')
    p.add_argument('--contexts',type=Path);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();questions=load_questions(a.dataset,a.split,a.allow_unreviewed);by={q.query_id:q for q in questions}
    meta=json.loads((a.dataset/'metadata.json').read_text(encoding='utf-8'));manifest=json.loads(a.manifest.read_text(encoding='utf-8'))
    for key in ['dataset_version','repository_commit','ingestion_config_hash','chunking_config_hash','embedding_model_version','index_version','retriever_config','reranker_config','generator_model_version','generator_prompt_hash','temperature','seed','judge_model_version','judge_prompt_hash','metric_code_version']:
        if key not in manifest:raise ValueError('Missing experiment version: '+key)
    for key in ['dataset_version','repository_commit','ingestion_config_hash']:
        if manifest[key]!=meta[key]:raise ValueError('Dataset/manifest version mismatch: '+key)
    if not manifest['index_version'] or not manifest['metric_code_version']:raise ValueError('Real index and metric versions required')
    if not isinstance(manifest['retriever_config'],dict) or not manifest['retriever_config']:raise ValueError('Actual retriever configuration required')
    if a.mode=='generation' and (not manifest['generator_model_version'] or not manifest['generator_prompt_hash']):raise ValueError('Actual generator version and prompt hash required')
    if manifest.get('dataset_sha256')!=hashlib.sha256((a.dataset/'questions.jsonl').read_bytes()).hexdigest():raise ValueError('Manifest must match dataset_sha256')
    if manifest.get('chunks_sha256')!=hashlib.sha256(a.chunks.read_bytes()).hexdigest():raise ValueError('Manifest must match actual chunks_sha256')
    for path,digest in meta['corpus_hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed frozen corpus: '+path)
    if a.output.exists():raise FileExistsError(a.output)
    data=rows(a.results);results=[]
    if a.mode=='retrieval':
        if len({r['query_id'] for r in data})!=len(data):raise ValueError('Duplicate retrieval query ID')
        sections={s.section_id:s for s in (Section.model_validate(x) for x in rows(ROOT/'data/processed/phase1-v3-final/sections.jsonl'))}
        chunks={c.chunk_id:c for c in (Chunk.model_validate(x) for x in rows(a.chunks))}
        for row in data:
            q=by[row['query_id']];ranked=resolve_results(row['results'],sections,chunks)
            for k in (5,10):
                metric=retrieval_metrics(q,ranked,k)
                if metric is not None:results.append(dict(query_id=q.query_id,category=q.category,topic_group=q.topic_group,k=k,**metric))
    else:
        if a.contexts is None:raise ValueError('Actual context snapshots required')
        snapshots={(r['query_id'],r['run_id']):r['contexts'] for r in rows(a.contexts)}
        for row in data:
            j=GenerationJudgment.model_validate(row);q=by[j.query_id];context=snapshots[(j.query_id,j.run_id)]
            if canonical(context)!=j.context_snapshot_hash or set(j.context_ids)!={c['context_id'] for c in context}:raise ValueError('Actual context/hash mismatch')
            results.append(dict(query_id=q.query_id,category=q.category,topic_group=q.topic_group,run_id=j.run_id,context_snapshot_hash=j.context_snapshot_hash,
                                latency_ms=j.latency_ms,cost_usd=j.cost_usd,input_tokens=j.input_tokens,output_tokens=j.output_tokens,**score_generation(q,j,a.allow_unreviewed)))
    if {r['query_id'] for r in data}!=set(by):raise ValueError('Results must cover the selected split exactly')
    summary={}
    if a.mode=='retrieval':
        for k in (5,10):
            for category in ['all',*sorted({q.category for q in questions if q.answerable})]:
                subset=[r for r in results if r['k']==k and (category=='all' or r['category']==category)]
                summary[f'{category}@{k}']={key:dict(question_bootstrap=bootstrap([r[key] for r in subset]),topic_cluster_bootstrap=cluster_bootstrap(subset,key)) for key in ['hit_section','recall_section','mrr','ndcg','evidence_hit','required_fact_coverage','all_required_facts']}
    else:
        for category in ['all',*sorted({q.category for q in questions})]:
            subset=[r for r in results if category=='all' or r['category']==category]
            summary[category]={key:aggregate_runs(subset,key) for key in ['fact_precision','fact_recall','fact_f1','completeness','correctness_bucket','answer_relevance','faithfulness','citation_accuracy','answer_success','correct_abstention','false_refusal','unsupported_answer']}
            for key in summary[category]:summary[category][key]['topic_cluster_bootstrap']=cluster_bootstrap(subset,key)
    report=dict(status='diagnostic_unreviewed' if a.allow_unreviewed else 'scored_reviewed_labels',mode=a.mode,split=a.split,
                dataset_version=meta['dataset_version'],manifest=manifest,results=results,summary=summary,
                selected_questions=len(questions),answerable_questions=sum(q.answerable for q in questions),unanswerable_questions=sum(not q.answerable for q in questions),
                evaluation_code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/hybrid_rag/evaluation').glob('*.py'))},
                labels_exhaustive=False,note='Question and topic-cluster confidence intervals are both reported. Unjudged results are not automatically proven irrelevant.')
    report['input_records']=data
    report['evaluation_code_sha256']['evaluate_rag.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['input_artifacts']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.results,a.manifest,*([a.contexts] if a.contexts else [])]}
    if a.mode=='generation':
        report['telemetry']={}
        for key in ['latency_ms','cost_usd','input_tokens','output_tokens']:
            values=sorted(r[key] for r in results if r[key] is not None)
            report['telemetry'][key]=dict(unit='per_generation_run',observations=len(values),mean=sum(values)/len(values) if values else None,
                                         p95=values[int((len(values)-1)*.95)] if values else None,total=sum(values) if values else None)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'records':len(results),'output':str(a.output)},indent=2))


if __name__=='__main__':main()
