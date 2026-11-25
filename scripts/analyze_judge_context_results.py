"""Reproduce comparisons without any model calls or historical result writes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_rag.evaluation.dataset import load_questions
from hybrid_rag.evaluation.generation import GenerationJudgment,score_generation
from hybrid_rag.evaluation.judge import cohort_statistics,calibration_report
from hybrid_rag.evaluation.statistics import paired_bootstrap,cluster_bootstrap


def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT/'data/eval/runs/judge-context-v3');parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    questions=load_questions(ROOT/'data/eval/hybridrag-en-v2','dev',allow_unreviewed=True);by={q.query_id:q for q in questions}
    def scores(path):
        return [dict(query_id=j['query_id'],topic_group=by[j['query_id']].topic_group,run_id=j['run_id'],context_snapshot_hash=j['context_snapshot_hash'],**score_generation(by[j['query_id']],GenerationJudgment.model_validate(j),True)) for j in read(path/'judgments.jsonl')]
    source=ROOT/'data/eval/runs/dev40-20261116-v3'
    human={r['query_id']:r['score'] for r in read(source/'human_scores.imported.jsonl')}
    judge_path=a.root/'judge/frozen_top5';phase1=scores(judge_path)
    judge_summary=dict(calibration=calibration_report(human,phase1),cohorts=cohort_statistics(questions,phase1,read(judge_path/'answers.jsonl')))
    policies={};per_policy={};primary_answers={};corrections=[]
    for name in ['top5','top10','dynamic']:
        p=a.root/'context'/name;ss=scores(p);per_policy[name]={r['query_id']:r for r in ss};answers=read(p/'answers.jsonl');primary_answers[name]={r['query_id']:r for r in answers}
        selection=read(p/'selection.jsonl');historical=json.loads((p/'summary.json').read_text(encoding='utf-8'))
        failures=read(p/'failures.jsonl')
        avg=lambda k:sum(r[k] for r in ss if r[k] is not None)/sum(r[k] is not None for r in ss) if any(r[k] is not None for r in ss) else None
        policies[name]=dict(valid_answers=len(answers),valid_judgments=len(ss),expected_questions=40,failures=failures,
            cohorts=cohort_statistics(questions,ss,answers),conditional_mean_score=avg('correctness_bucket'),
            valid_and_full_score_rate_over40=sum(r['correctness_bucket']==1 for r in ss)/40,
            strict_answer_success_rate_over40=sum(r['answer_success'] for r in ss)/40,
            faithfulness=avg('faithfulness'),citation_accuracy=avg('citation_accuracy'),resources=historical['resources'],
            known_generation_token_coverage=f'{len(answers)}/40 first-attempt outputs; rejected raw outputs not recorded in initial version')
        old={r['query_id']:r for r in read(p/'scores.jsonl')}
        for r in ss:
            for k in r:
                if k in old[r['query_id']] and old[r['query_id']][k]!=r[k]:corrections.append(dict(policy=name,query_id=r['query_id'],metric=k,old=old[r['query_id']][k],new=r[k]))
    common=set.intersection(*(set(s) for s in per_policy.values()))
    paired={}
    for name in ['top10','dynamic']:
        first={q:per_policy['top5'][q]['correctness_bucket'] for q in common};second={q:per_policy[name][q]['correctness_bucket'] for q in common}
        deltas=[dict(query_id=q,topic_group=by[q].topic_group,delta=second[q]-first[q]) for q in sorted(common)]
        paired[name]=dict(n=len(common),common_query_ids=sorted(common),top5_mean=sum(first.values())/len(common),policy_mean=sum(second.values())/len(common),question=paired_bootstrap(first,second),topic_cluster=cluster_bootstrap(deltas,'delta'))
    retry={}
    for name in ['top5','top10','dynamic']:
        p=a.root/'context-retry'/name;ss=scores(p);answers=read(p/'answers.jsonl')
        now={r['query_id']:r for r in answers}
        assert all(now[k]==v for k,v in primary_answers[name].items())
        retry[name]=dict(valid_answers=len(answers),valid_judgments=len(ss),failures=read(p/'failures.jsonl'),cohorts=cohort_statistics(questions,ss,answers),diagnostic_generation_responses=len(read(p/'generation_traces.jsonl')) if (p/'generation_traces.jsonl').exists() else 0)
    focus={}
    for qid in ['q071','q092','q093','q094','q095','q096','q097','q098']:
        focus[qid]={name:per_policy[name].get(qid) for name in per_policy}
    phase1_by={r['query_id']:r for r in phase1}
    baseline_variation=[dict(query_id=q,before=phase1_by[q]['correctness_bucket'],fresh_top5=per_policy['top5'][q]['correctness_bucket']) for q in phase1_by if phase1_by[q]['correctness_bucket']!=per_policy['top5'][q]['correctness_bucket']]
    for f,h in json.loads((a.root/'judge/report.json').read_text(encoding='utf-8'))['protected_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    result=dict(status='context_strategies_not_accepted',judge=judge_summary,primary=policies,paired_common_complete=paired,
        retry_diagnostic_only=retry,focus=focus,baseline_variation=baseline_variation,metric_corrections=corrections,
        limitation='First-attempt failures remain failures. Conditional means exclude unavailable outputs; paired analyses use common IDs. Retry does not replace the primary experiment. No new human calibration for new answers.',protected_source_verified=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(a.output),common_complete=len(common),judge_calibration=judge_summary['calibration'],metric_corrections=corrections),ensure_ascii=False))


if __name__=='__main__':main()
