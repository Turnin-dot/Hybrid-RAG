"""Test the remaining 80 questions once with frozen Top5 / grounded-v2 settings.

--plan-only validates counts, split isolation and frozen file hashes without
loading retrieval models or calling an API. Real runs use candidate labels and
remain diagnostic until the dataset and new answers are human approved.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from run_generation_eval import main as run_evaluation
from hybrid_rag.evaluation.dataset import load_questions

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/'data/eval/test80_frozen_config.json'
SETTINGS=dict(subset='test',expected_count=80,answerable=74,unanswerable=6,
    strategy='hybrid_rerank',candidate_k=40,top_k=5,provider='deepseek',model='deepseek-chat',
    judge_model='deepseek-chat',judge_version='grounded-v2',generator_max_tokens=768,judge_max_tokens=4096,temperature=0)


def file_sha256(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def preflight():
    config=json.loads(FREEZE.read_text(encoding='utf-8'))
    if config['settings']!=SETTINGS:raise ValueError('Frozen Test80 settings differ; do not tune Test configuration')
    for relative,expected in config['files_sha256'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or file_sha256(path)!=expected:raise ValueError('Frozen file changed: '+relative)
    dataset=ROOT/config['dataset']
    test=load_questions(dataset,'test',allow_unreviewed=True)
    dev=load_questions(dataset,'dev',allow_unreviewed=True)
    ids=[q.query_id for q in test]
    if len(test)!=80 or len(set(ids))!=80 or sum(q.answerable for q in test)!=74 or any(q.split!='test' for q in test):raise ValueError('Expected exactly Test80: 74 answerable and 6 unanswerable')
    if ids!=config['query_ids']:raise ValueError('Frozen test membership or ordering changed')
    if set(ids)&{q.query_id for q in dev}:raise ValueError('Dev/Test query leakage')
    if {q.topic_group for q in dev}&{q.topic_group for q in test}:raise ValueError('Dev/Test topic leakage')
    if {e.section_id for q in dev for e in q.gold_evidence}&{e.section_id for q in test for e in q.gold_evidence}:raise ValueError('Dev/Test gold Section leakage')
    return dict(status='preflight_passed_no_api',questions=80,answerable=74,unanswerable=6,
        query_ids=ids,dev_test_isolation_verified=True,freeze_sha256=file_sha256(FREEZE),
        candidate_labels=any(q.annotation_status!='approved' for q in test),settings=config['settings'],
        note='Membership validation only; no Test answers or scores generated. Do not tune settings after observing Test results.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args(argv)
    name=('test80-plan-' if args.plan_only else 'test80-')+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    output=args.output or ROOT/'data/eval/runs'/name
    if output.exists():raise FileExistsError(output)
    plan=preflight()
    if args.plan_only:
        output.mkdir(parents=True,exist_ok=False)
        (output/'preflight.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
        print(json.dumps(dict(status=plan['status'],questions=80,answerable=74,unanswerable=6,output=str(output))))
        return 0
    config=json.loads(FREEZE.read_text(encoding='utf-8'));s=config['settings']
    result=run_evaluation(['--dataset',str(ROOT/config['dataset']),'--index',str(ROOT/config['index']),
        '--subset','test','--query-ids','all','--expected-count','80','--allow-unreviewed','--require-api',
        '--provider',s['provider'],'--model',s['model'],'--judge-model',s['judge_model'],'--judge-version',s['judge_version'],
        '--strategy',s['strategy'],'--candidate-k',str(s['candidate_k']),'--top-k',str(s['top_k']),
        '--citation-retries','0','--freeze-manifest',str(FREEZE),'--output',str(output)])
    try:
        after=preflight()
        if after['freeze_sha256']!=plan['freeze_sha256']:raise ValueError('Freeze manifest changed during evaluation')
    except ValueError as exc:
        report_path=output/'generation_report.json'
        if report_path.exists():
            report=json.loads(report_path.read_text(encoding='utf-8'));report['status']='failed_frozen_inputs_changed'
            report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    (output/'freeze_verification.json').write_text(json.dumps(dict(status='frozen_inputs_unchanged',freeze_sha256=plan['freeze_sha256'],generation_exit_code=result),indent=2),encoding='utf-8')
    return result


if __name__=='__main__':raise SystemExit(main())
