"""Sequential frozen-answer Judge repair and frozen-candidate context experiments."""
import argparse
import hashlib
import json
import sys
import time
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_rag.evaluation.dataset import load_questions
from hybrid_rag.evaluation.generation import score_generation
from hybrid_rag.evaluation.judge import evaluate_answer,cohort_statistics,calibration_report,digest,EXTRACT,ASSESS
from hybrid_rag.evaluation.retrieval import retrieval_metrics
from hybrid_rag.retrieval.context_selection import select_contexts
from hybrid_rag.retrieval.generation import make_chat_model,generate,build_prompt,RUBRIC
from run_generation_eval import write_review_artifacts,write_jsonl,InvocationRecorder


def read(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def verify_resume_policy(previous, current):
    # Historical experiments predate this explicit field and used no citation retry.
    for key, default in [('citation_retries',0),('temperature',0),('sdk_max_retries',1)]:
        if previous.get(key,default)!=current[key]:
            raise ValueError('Resume policy changed: '+key)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['judge','context'],required=True)
    p.add_argument('--source',type=Path,default=ROOT/'data/eval/runs/dev40-20261116-v3')
    p.add_argument('--judge-results',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-from',type=Path,help='Copy prior checkpoints into a NEW directory; retry only unscored questions')
    p.add_argument('--citation-retries',type=int,choices=range(4),default=0,
        help='Explicit extra citation correction calls, identical for all policies; historical comparison used 0')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    dataset=ROOT/'data/eval/hybridrag-en-v2'
    questions=load_questions(dataset,'dev',allow_unreviewed=True);by={q.query_id:q for q in questions}
    original=read(a.source/'answers.jsonl');diagnostics=read(a.source/'retrieval_diagnostics.jsonl')
    if len(original)!=40 or {r['query_id'] for r in original}!=set(by):raise ValueError('Require frozen complete Dev40')
    for r in original:
        if r['dry_run'] or digest(r['contexts'])!=r['context_snapshot_hash']:raise ValueError('Invalid frozen answer snapshot')
        if r['prompt_template_sha256']!=hashlib.sha256(RUBRIC.encode()).hexdigest():raise ValueError('Generation Prompt changed')
    source_report=json.loads((a.source/'generation_report.json').read_text(encoding='utf-8'))
    if source_report['dataset_sha256']!=hashlib.sha256((dataset/'questions.jsonl').read_bytes()).hexdigest():raise ValueError('Dataset changed')
    if a.phase=='context':
        if not a.judge_results:raise ValueError('Complete phase one first')
        phase1=json.loads((a.judge_results/'report.json').read_text(encoding='utf-8'))
        if phase1['status']!='complete' or phase1['phase']!='judge' or phase1['source_answers_sha256']!=hashlib.sha256((a.source/'answers.jsonl').read_bytes()).hexdigest():raise ValueError('Phase one not verified')
        if phase1['extraction_prompt_sha256']!=hashlib.sha256(EXTRACT.encode()).hexdigest() or phase1['assessment_prompt_sha256']!=hashlib.sha256(ASSESS.encode()).hexdigest():raise ValueError('Judge changed since calibration')
    judge=make_chat_model(provider='deepseek',model='deepseek-chat',max_tokens=4096,temperature=0,max_retries=1)
    generator=make_chat_model(provider='deepseek',model='deepseek-chat',max_tokens=768,temperature=0,max_retries=1) if a.phase=='context' else None
    a.output.mkdir(parents=True,exist_ok=False)
    protected={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*a.source.glob('*.*'),ROOT/'src/hybrid_rag/retrieval/generation.py'] if f.is_file()}
    config=dict(phase=a.phase,status='running',source=str(a.source),source_answers_sha256=hashlib.sha256((a.source/'answers.jsonl').read_bytes()).hexdigest(),
        protected_sha256=protected,generator_model='deepseek-chat',generator_max_tokens=768,judge_max_tokens=4096,
        extraction_prompt_sha256=hashlib.sha256(EXTRACT.encode()).hexdigest(),assessment_prompt_sha256=hashlib.sha256(ASSESS.encode()).hexdigest(),
        code_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'scripts/run_generation_eval.py',ROOT/'src/hybrid_rag/evaluation/judge.py',ROOT/'src/hybrid_rag/evaluation/generation.py',ROOT/'src/hybrid_rag/retrieval/context_selection.py',ROOT/'src/hybrid_rag/retrieval/generation.py']},
        citation_retries=a.citation_retries,temperature=0,sdk_max_retries=1,
        generation_prompt_sha256=hashlib.sha256(RUBRIC.encode()).hexdigest(),
        dataset_sha256=source_report['dataset_sha256'],
        candidate_snapshot_sha256=hashlib.sha256((a.source/'retrieval_diagnostics.jsonl').read_bytes()).hexdigest(),
        query_ids_in_order=[r['query_id'] for r in original],
        dynamic=dict(max_contexts=10,prompt_budget_bge_proxy=6000,one_per_section_first=True,jaccard_redundancy_threshold=.85),
        candidate_pool='frozen ordered Hybrid+Rerank top40; no new retrieval timing',test_used=False)
    if a.resume_from:
        previous=json.loads((a.resume_from/'config.json').read_text(encoding='utf-8'))
        for key in ['phase','source_answers_sha256','generator_model','generator_max_tokens','judge_max_tokens','extraction_prompt_sha256','assessment_prompt_sha256','dynamic']:
            if previous[key]!=config[key]:raise ValueError('Resume configuration changed: '+key)
        verify_resume_policy(previous,config)
        config['resume_parent']=str(a.resume_from.resolve())
    for relative in config['code_sha256']:
        target=a.output/'code_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    (a.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    strategies=['frozen_top5'] if a.phase=='judge' else ['top5','top10','dynamic']
    if a.phase=='context':
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(ROOT/'data/models/bge-base-en-v1.5-tokenizer',local_files_only=True)
        count=lambda text:len(tok.encode(text,add_special_tokens=True,truncation=False))
    reports={};all_failures=[]
    for strategy in strategies:
        dest=a.output/strategy
        if a.resume_from and (a.resume_from/strategy).exists():
            shutil.copytree(a.resume_from/strategy,dest)
            if (dest/'failures.jsonl').exists():shutil.copyfile(dest/'failures.jsonl',dest/'parent_failures.jsonl')
        else:dest.mkdir()
        restore=lambda name:read(dest/name) if (dest/name).exists() else []
        answers=restore('answers.jsonl');judgments=restore('judgments.jsonl');scores=restore('scores.jsonl');traces=restore('traces.jsonl');generation_traces=restore('generation_traces.jsonl');selection=restore('selection.jsonl');failures=[]
        completed={r['query_id'] for r in scores};saved_answers={r['query_id']:r for r in answers}
        for saved,diag in zip(original,diagnostics):
            if diag['query_id']!=saved['query_id']:raise ValueError('Candidate order changed')
            q=by[saved['query_id']];stage='selection'
            if q.query_id in completed:continue
            try:
                if q.query_id in saved_answers:row=saved_answers[q.query_id]
                elif a.phase=='judge':row=saved
                else:
                    start=time.perf_counter();contexts=select_contexts(q.question,diag['candidates'],strategy,count)
                    selection_ms=(time.perf_counter()-start)*1000
                    ranked=[dict(section_id=c['section_id'],ranges=c['ranges']) for c in contexts]
                    evidence= [e for e in q.gold_evidence]
                    from hybrid_rag.evaluation.retrieval import covered
                    ev_recall=sum(covered(e.section_span,[span for c in contexts if c['section_id']==e.section_id for span in c['ranges']]) for e in evidence)/len(evidence) if evidence else None
                    selection=[r for r in selection if r['query_id']!=q.query_id]
                    selection.append(dict(query_id=q.query_id,chunk_ids=[c['chunk_id'] for c in contexts],context_count=len(contexts),
                        evidence_recall=ev_recall,selection_ms=selection_ms,prompt_tokens_bge_proxy=count(build_prompt(q.question,contexts))))
                    stage='generation';row=dict(query_id=q.query_id,question=q.question,run_id='run-1',**generate(q.question,contexts,InvocationRecorder(generator,q.query_id,generation_traces),citation_retries=a.citation_retries))
                    if generation_traces and generation_traces[-1]['query_id']==q.query_id:generation_traces[-1]['output_validation']='passed'
                if q.query_id not in saved_answers:answers.append(row)
                stage='judge'
                j=evaluate_answer(q,row,judge,traces);judgments.append(j.model_dump())
                scores.append(dict(query_id=q.query_id,topic_group=q.topic_group,run_id=row['run_id'],context_snapshot_hash=row['context_snapshot_hash'],**score_generation(q,j,allow_unreviewed=True)))
            except Exception as exc:
                error=dict(query_id=q.query_id,strategy=strategy,stage=stage,error=type(exc).__name__)
                if isinstance(exc,ValueError):error['validation_detail']=str(exc)
                if stage=='generation' and generation_traces and generation_traces[-1]['query_id']==q.query_id:generation_traces[-1]['output_validation']='failed'
                failures.append(error)
            finally:
                for name,rows in [('answers.jsonl',answers),('judgments.jsonl',judgments),('scores.jsonl',scores),('traces.jsonl',traces),('generation_traces.jsonl',generation_traces),('failures.jsonl',failures),('selection.jsonl',selection)]:write_jsonl(dest,name,rows)
                print(json.dumps(dict(strategy=strategy,query_id=q.query_id,answers=len(answers),scored=len(scores),failures=len(failures))),flush=True)
        write_review_artifacts(dest,dataset,questions,answers,judgments,scores,failures)
        summary=cohort_statistics(questions,scores,answers)
        def metric(values):
            values=sorted(v for v in values if v is not None)
            return dict(n=len(values),mean=sum(values)/len(values) if values else None,p50=values[(len(values)-1)//2] if values else None,p95=values[int((len(values)-1)*.95)] if values else None)
        summary['resources']=dict(generation_input_tokens=sum((r.get('usage') or {}).get('input_tokens',0) or 0 for r in answers),
            generation_output_tokens=sum((r.get('usage') or {}).get('output_tokens',0) or 0 for r in answers),
            judge_input_tokens=sum((r.get('usage') or {}).get('input_tokens',0) or 0 for r in traces),
            judge_output_tokens=sum((r.get('usage') or {}).get('output_tokens',0) or 0 for r in traces),
            generation_latency_ms=metric([r.get('latency_ms') for r in answers]),judge_latency_ms=metric([r.get('latency_ms') for r in traces]),
            selection_latency_ms=metric([r['selection_ms'] for r in selection]),
            evidence_recall=metric([r['evidence_recall'] for r in selection]),context_count=metric([r['context_count'] for r in selection]))
        if a.phase=='judge' and not failures:
            human={r['query_id']:r['score'] for r in read(a.source/'human_scores.imported.jsonl')}
            summary['calibration']=calibration_report(human,scores)
            old_scores=source_report['scores'];summary['before_calibration']=calibration_report(human,old_scores)
            summary['before_cohorts']=cohort_statistics(questions,old_scores,original)
            summary['disagreement_audit']=[dict(query_id=q.query_id,human=human[q.query_id],before=next(r['correctness_bucket'] for r in old_scores if r['query_id']==q.query_id),after=next(r['correctness_bucket'] for r in scores if r['query_id']==q.query_id),judgment=j) for q,j in [(by[j['query_id']],j) for j in judgments] if human[q.query_id]!=next(r['correctness_bucket'] for r in scores if r['query_id']==q.query_id) or human[q.query_id]!=next(r['correctness_bucket'] for r in old_scores if r['query_id']==q.query_id)]
        (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        reports[strategy]=summary;all_failures.extend(failures)
    if any(hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h for f,h in protected.items()):raise ValueError('Frozen source changed during experiment')
    config.update(status='failed' if all_failures else 'complete',failures=all_failures,results=reports)
    (a.output/'report.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=config['status'],output=str(a.output))))
    return int(bool(all_failures))


if __name__=='__main__':raise SystemExit(main())
