"""Audit saved unanswerable responses under an explicit semantic refusal policy.

Never regenerate answers, retrieve context, overwrite labels or change grounded-v2.
--live creates new answer-only DeepSeek annotations; --assessments replays them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from run_generation_eval import ROOT, summary, run_statistics
from hybrid_rag.evaluation.dataset import load_questions
from hybrid_rag.evaluation.generation import GenerationJudgment
from hybrid_rag.evaluation.judge import calibration_report
from hybrid_rag.evaluation.refusal import POLICY, REFUSAL_PROMPT, RefusalAssessment, assess_refusal, score_with_refusal
from hybrid_rag.retrieval.generation import make_chat_model

CONTROLS = [
    dict(query_id='control-refusal', question='What is the measured p95 latency of my unreleased application?',
        answer='I cannot determine a measured p95 latency: no measurements of your application are provided.', expected='full_refusal'),
    dict(query_id='control-substantive', question='What is the private deployment password?',
        answer='I cannot verify it, but the deployment password is EXAMPLE_ONLY_PASSWORD.', expected='substantive_answer'),
    dict(query_id='control-partial', question='Which parameter supplies memory files, and when are skill details loaded?',
        answer='The excerpts do not identify the memory-file parameter. Skill details are loaded only when relevant.', expected='partial_answer'),
]


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(source, output, dataset, *, live=False, assessments=None, human=None):
    source, output, dataset = source.resolve(), output.resolve(), dataset.resolve()
    if live == (assessments is not None):
        raise ValueError('Choose exactly one of live annotation or saved annotation replay')
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use a new independent result directory')
    protected = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    old = json.loads((source/'generation_report.json').read_text(encoding='utf-8'))
    if sha(dataset/'questions.jsonl') != old['dataset_sha256'] or old['judge_version'] != 'grounded-v2':
        raise ValueError('Expected the same dataset and frozen grounded-v2 judgments')
    questions = load_questions(dataset, old['subset'], allow_unreviewed=True)
    if {q.query_id for q in questions} != set(old['query_ids']):
        raise ValueError('Question membership changed')
    answers = read_rows(source/'answers.jsonl')
    judgments = read_rows(source/'judgments.jsonl')
    by_answer = {r['query_id']: r for r in answers}
    by_judge = {r['query_id']: r for r in judgments}
    if len(answers) != len(questions) or len(by_answer) != len(questions) or set(by_answer) != set(by_judge):
        raise ValueError('Exactly one saved answer and judgment per question required')
    targets = [q for q in questions if not q.answerable]
    jobs = [dict(query_id=q.query_id, question=q.question, answer=by_answer[q.query_id]['answer']) for q in targets]+CONTROLS
    if live:
        model = make_chat_model(provider=old['provider'], model=old['requested_judge_model'], max_tokens=1024)
        annotations = {}
    else:
        entries = read_rows(assessments)
        annotations = {r['query_id']: RefusalAssessment.model_validate(r) for r in entries}
        if len(entries) != len(annotations) or set(annotations) != {j['query_id'] for j in jobs}:
            raise ValueError('Saved annotations must match targets and all controls exactly')
    output.mkdir(parents=True, exist_ok=False)
    traces, failures = [], []
    config = dict(policy=POLICY, scope='All saved unanswerable questions; answerable scores unchanged',
        source=str(source), live_semantic_judge=live, generation_calls=0, retrieval_calls=0,
        prompt_sha256=hashlib.sha256(REFUSAL_PROMPT.encode()).hexdigest(), source_sha256=protected,
        implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT/'src/hybrid_rag/evaluation/refusal.py', ROOT/'src/hybrid_rag/evaluation/generation.py']})
    if assessments:
        config['annotation_input_sha256'] = sha(assessments)
    (output/'policy_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    for job in jobs:
        try:
            a = assess_refusal(job['query_id'], job['question'], job['answer'], model, traces) if live else annotations[job['query_id']]
            a.validate_answer(job['query_id'], job['answer'])
            annotations[job['query_id']] = a
            if 'expected' in job and a.response_type != job['expected']:
                raise ValueError('Semantic negative/control classification failed')
        except Exception as exc:
            failures.append(dict(query_id=job['query_id'], stage='semantic_refusal', error_type=type(exc).__name__,
                validation_detail=str(exc) if isinstance(exc, ValueError) else None))
        write_rows(output/'refusal_assessments.jsonl', [a.model_dump() for a in annotations.values()])
        write_rows(output/'semantic_traces.jsonl', traces)
        write_rows(output/'failures.jsonl', failures)
        print(json.dumps(dict(query_id=job['query_id'], annotated=job['query_id'] in annotations, failures=len(failures))), flush=True)
    if failures:
        (output/'generation_report.json').write_text(json.dumps(dict(status='failed', failures=failures)), encoding='utf-8')
        return 1
    updated = {}
    conflict_rows = []
    for q in targets:
        j = GenerationJudgment.model_validate(by_judge[q.query_id])
        metrics = score_with_refusal(q, j, annotations[q.query_id], allow_unreviewed=True)
        updated[q.query_id] = metrics
        conflict_rows.append(dict(query_id=q.query_id, protocol_abstained=by_answer[q.query_id]['abstained'],
            semantic_response_type=annotations[q.query_id].response_type, conflict=metrics['refusal_conflict'],
            answer_quotes=annotations[q.query_id].answer_quotes, reason=annotations[q.query_id].reason))
    scores = [dict(r, **updated.get(r['query_id'], {})) for r in old['scores']]
    before_by = {r['query_id']: r for r in old['scores']}
    changes = [dict(query_id=r['query_id'], before=before_by[r['query_id']]['correctness_bucket'], after=r['correctness_bucket'])
        for r in scores if r['correctness_bucket'] != before_by[r['query_id']]['correctness_bucket']]
    statistics = run_statistics(questions, answers, scores, read_rows(source/'retrieval_diagnostics.jsonl'),
        read_rows(source/'generation_traces.jsonl'), read_rows(source/'judge_traces.jsonl'))
    known = [t['usage'] for t in traces if t.get('usage')]
    api = dict(calls=len(traces), calls_with_usage=len(known), calls_without_usage=len(traces)-len(known),
        input_tokens=sum(u.get('input_tokens', 0) or 0 for u in known),
        output_tokens=sum(u.get('output_tokens', 0) or 0 for u in known),
        note='Only new semantic annotation calls. Replay makes zero API calls; historical annotation origin remains automatic.')
    result = dict(old, status='diagnostic_new_scoring_policy_pending_human_review', scores=scores,
        summary=summary(scores), statistics=statistics, scoring_policy=POLICY, policy_scope=config['scope'],
        policy_parent=str(source), refusal_conflicts=conflict_rows, score_changes=changes, semantic_api_usage=api,
        semantic_refusal_count=sum(a.response_type == 'full_refusal' for qid, a in annotations.items()
            if qid in {q.query_id for q in targets}), protocol_refusal_count=sum(by_answer[q.query_id]['abstained'] for q in targets),
        note='New policy diagnosis after Test observation, not a fresh holdout result. Original grounded-v2 scores and all answerable scores unchanged.')
    if human:
        h = read_rows(human)
        if any(not r.get('reviewer') or r.get('score') not in (0, .5, 1) for r in h):
            raise ValueError('Explicit original human scores required')
        labels = {r['query_id']: r['score'] for r in h}
        result['calibration_before'] = calibration_report(labels, old['scores'])
        result['calibration_after'] = calibration_report(labels, scores)
        result['human_input_sha256'] = sha(human)
    write_rows(output/'scores.jsonl', scores)
    write_rows(output/'automatic_scores.jsonl', [dict(query_id=r['query_id'], score=r['correctness_bucket']) for r in scores])
    write_rows(output/'refusal_conflicts.jsonl', conflict_rows)
    for name in ['answers.jsonl', 'contexts.jsonl', 'judgments.jsonl']:
        shutil.copyfile(source/name, output/name)
    unchanged = all(sha(source/name) == expected for name, expected in protected.items())
    if not unchanged:
        result['status'] = 'failed_source_changed'
    (output/'generation_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'verification.json').write_text(json.dumps(dict(source_unchanged=unchanged,
        answerable_scores_unchanged=all(r == before_by[r['query_id']] for r in scores if r['query_id'] not in updated),
        original_answers_and_judgments_copied_exactly=True, controls_passed=len(CONTROLS), new_generation_calls=0), indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], changes=changes, semantic_api_usage=api, output=str(output)), indent=2))
    return int(not unchanged)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/eval/hybridrag-en-v2')
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true')
    mode.add_argument('--assessments', type=Path)
    parser.add_argument('--human', type=Path)
    a = parser.parse_args(argv)
    return run(a.source, a.output, a.dataset, live=a.live, assessments=a.assessments, human=a.human)


if __name__ == '__main__':
    raise SystemExit(main())
