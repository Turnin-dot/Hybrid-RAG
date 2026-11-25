"""Recover saved citation failures without retrieval or changing accepted rows."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from run_generation_eval import (ROOT, InvocationRecorder, canonical, run_statistics,
    summary, write_jsonl, write_review_artifacts)
from hybrid_rag.evaluation.dataset import load_questions
from hybrid_rag.evaluation.generation import score_generation
from hybrid_rag.evaluation.judge import EXTRACT, ASSESS, evaluate_answer
from hybrid_rag.retrieval.generation import RUBRIC, build_prompt, generate, make_chat_model


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def directory_hashes(source):
    return {p.relative_to(source).as_posix(): sha(p) for p in sorted(source.rglob('*')) if p.is_file()}


def prepare(source, dataset, requested):
    report = json.loads((source/'generation_report.json').read_text(encoding='utf-8'))
    if sha(dataset/'questions.jsonl') != report['dataset_sha256']:
        raise ValueError('Dataset changed')
    questions = load_questions(dataset, 'test', allow_unreviewed=True)
    by = {q.query_id: q for q in questions}
    ids = report['query_ids']
    if len(ids) != len(set(ids)) or set(ids) != set(by):
        raise ValueError('Expected the same complete held-out question membership')
    if report['judge_version'] != 'grounded-v2' or report['provider'] != 'deepseek':
        raise ValueError('Expected the original DeepSeek / grounded-v2 experiment')
    if report['judge_rubric_sha256'] != hashlib.sha256((EXTRACT+ASSESS).encode()).hexdigest():
        raise ValueError('Judge rules changed')
    frozen = json.loads((source/'frozen_config.snapshot.json').read_text(encoding='utf-8'))
    # Generation orchestration changed for recovery; evaluation rules/data stay frozen.
    for relative, expected in frozen['files_sha256'].items():
        if relative.startswith('src/hybrid_rag/evaluation/') and sha(ROOT/relative) != expected:
            raise ValueError('Frozen evaluation implementation changed: '+relative)
    answers = read_rows(source/'answers.jsonl')
    judgments = read_rows(source/'judgments.jsonl')
    scores = report['scores']
    accepted = {r['query_id'] for r in answers}
    if any(len(rows) != len({r['query_id'] for r in rows}) for rows in [answers, judgments, scores]):
        raise ValueError('Duplicate historical accepted rows')
    if accepted != {r['query_id'] for r in judgments} or accepted != {r['query_id'] for r in scores}:
        raise ValueError('Recovery requires matching accepted answers, judgments and scores')
    missing = set(ids)-accepted
    if missing != set(requested) or len(requested) != len(set(requested)):
        raise ValueError('Recover exactly the missing questions, never accepted answers')
    failures = report['failures']
    if {r['query_id'] for r in failures} != missing or any(
        r['stage'] != 'generation' or r.get('validation_detail') != 'Citation does not match an actual retrieved source'
        for r in failures):
        raise ValueError('Expected only saved citation-matching generation failures')
    retrieval = read_rows(source/'retrieval_diagnostics.jsonl')
    diagnostics = {r['query_id']: r for r in retrieval}
    traces = read_rows(source/'generation_traces.jsonl')
    jobs = []
    for qid in ids:
        if qid not in missing:
            continue
        previous = [r for r in traces if r['query_id'] == qid]
        if len(previous) != 1:
            raise ValueError('Expected exactly one original generation attempt per failed question')
        original = previous[0]
        candidates = {c['chunk_id']: c for c in diagnostics[qid]['candidates']}
        contexts = [candidates[cid] for cid in diagnostics[qid]['selected_chunk_ids']]
        question = by[qid]
        prompt = build_prompt(question.question, contexts)
        messages = [dict(role='system', content=RUBRIC), dict(role='user', content=prompt[len(RUBRIC):])]
        if messages != original['messages'] or canonical(messages) != original['prompt_sha256']:
            raise ValueError('Saved question, prompt or actual selected contexts changed: '+qid)
        jobs.append((question, contexts, original))
    return report, [by[qid] for qid in ids], answers, judgments, scores, retrieval, jobs


def run(source, output, dataset, requested, citation_retries=2):
    source, output, dataset = source.resolve(), output.resolve(), dataset.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Recovery output must be independent of historical input')
    if output.exists():
        raise FileExistsError(output)
    if type(citation_retries) is not int or not 1 <= citation_retries <= 3:
        raise ValueError('Recovery permits 1..3 additional real generation calls per question')
    protected = directory_hashes(source)
    old, questions, original_answers, original_judgments, original_scores, retrieval, jobs = prepare(source, dataset, requested)
    model = make_chat_model(provider='deepseek', model=old['requested_generator_model'], max_tokens=768)
    judge = make_chat_model(provider='deepseek', model=old['requested_judge_model'], max_tokens=4096)
    output.mkdir(parents=True, exist_ok=False)
    answers, judgments, scores, snapshots = [], [], [], []
    gen_traces, judge_traces, failures, outcomes = [], [], [], []
    old_gen = read_rows(source/'generation_traces.jsonl')
    old_judge = read_rows(source/'judge_traces.jsonl')
    source_files = ['answers.jsonl', 'contexts.jsonl', 'judgments.jsonl', 'automatic_scores.jsonl',
                    'generation_traces.jsonl', 'judge_traces.jsonl']
    for name in ['retrieval_diagnostics.jsonl']:
        shutil.copyfile(source/name, output/name)
    shutil.copyfile(source/'failures.jsonl', output/'parent_failures.jsonl')
    shutil.copyfile(source/'frozen_config.snapshot.json', output/'parent_frozen_config.snapshot.json')
    config = dict(experiment='saved-context citation recovery; not a new first-attempt Test80 run',
        parent=str(source), requested_ids=requested, citation_retries=citation_retries,
        new_generation_calls_per_question=citation_retries, retrieval_repeated=False,
        inherited_scores_recomputed=False, source_sha256=protected,
        implementation_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT/'scripts/run_generation_eval.py', ROOT/'src/hybrid_rag/retrieval/generation.py',
            ROOT/'src/hybrid_rag/evaluation/judge.py', ROOT/'src/hybrid_rag/evaluation/generation.py']})
    (output/'recovery_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    def checkpoint():
        additions = dict(zip(source_files, [answers, snapshots, judgments,
            [dict(query_id=r['query_id'], score=r['correctness_bucket']) for r in scores], gen_traces, judge_traces]))
        # Each accepted historical JSONL line is retained byte-for-byte as a prefix.
        for name, rows in additions.items():
            prefix = (source/name).read_bytes()
            if prefix and not prefix.endswith(b'\n'):
                prefix += b'\n'
            suffix = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode('utf-8')
            (output/name).write_bytes(prefix+suffix)
        write_jsonl(output, 'recovery_generation_traces.jsonl', gen_traces)
        write_jsonl(output, 'recovery_judge_traces.jsonl', judge_traces)
        write_jsonl(output, 'recovery_outcomes.jsonl', outcomes)
        write_jsonl(output, 'failures.jsonl', failures)

    checkpoint()
    for q, contexts, previous in jobs:
        stage = 'generation'
        start = time.perf_counter()
        outcome = dict(query_id=q.query_id, historical_first_attempt_valid=False,
            context_reused_exactly=True, parent_prompt_sha256=previous['prompt_sha256'],
            recovered=False, scored=False)
        try:
            out = generate(q.question, contexts, InvocationRecorder(model, q.query_id, gen_traces),
                citation_retries=citation_retries, previous_response=previous['raw_response'])
            row = dict(query_id=q.query_id, run_id='run-1', question=q.question, **out,
                recovery_parent=str(source), recovery_latency_ms=(time.perf_counter()-start)*1000,
                rag_latency_ms=None)
            # Latency is recovery generation only: no retrieval has been rerun.
            answers.append(row)
            snapshots.append(dict(query_id=q.query_id, run_id='run-1', contexts=out['contexts']))
            outcome['recovered'] = True
            stage = 'judge'
            j = evaluate_answer(q, row, judge, judge_traces)
            judgments.append(j.model_dump())
            scores.append(dict(query_id=q.query_id, topic_group=q.topic_group, run_id='run-1',
                context_snapshot_hash=out['context_snapshot_hash'], **score_generation(q, j, allow_unreviewed=True)))
            outcome['scored'] = True
        except Exception as exc:
            error = dict(query_id=q.query_id, stage=stage, error_type=type(exc).__name__)
            if isinstance(exc, ValueError):
                error['validation_detail'] = str(exc)
            failures.append(error)
        outcome['new_generation_calls'] = sum(r['query_id'] == q.query_id for r in gen_traces)
        outcome['new_judge_calls'] = sum(r['query_id'] == q.query_id for r in judge_traces)
        outcomes.append(outcome)
        checkpoint()
        print(json.dumps(outcome), flush=True)

    merged_answers, merged_judgments, merged_scores = original_answers+answers, original_judgments+judgments, original_scores+scores
    expected = len(questions)
    first = dict(valid_answers=len(original_answers), scored=len(original_scores), denominator=expected,
        generation_validity_rate=len(original_answers)/expected, scored_rate=len(original_scores)/expected)
    after = dict(valid_answers=len(merged_answers), scored=len(merged_scores), denominator=expected,
        generation_validity_rate=len(merged_answers)/expected, scored_rate=len(merged_scores)/expected,
        recovered=len(answers), requested=len(jobs), recovery_rate=len(answers)/len(jobs))
    unchanged = directory_hashes(source) == protected
    status = ('failed' if failures or not unchanged or len(merged_scores) != expected
              else 'diagnostic_automatic_judge_pending_human_calibration')
    report = dict(old, status=status, scores=merged_scores, failures=failures,
        answered_questions=len(merged_answers), judged_questions=len(merged_scores),
        summary=summary(merged_scores), recovery_parent=str(source), recovery_policy=config,
        first_attempt=first, after_recovery=after, recovery_outcomes=outcomes,
        statistics=run_statistics(questions, merged_answers, merged_scores, retrieval,
                                  old_gen+gen_traces, old_judge+judge_traces),
        recovery_statistics=run_statistics(questions, merged_answers, merged_scores, [], gen_traces, judge_traces),
        note='Validity rates are protocol success, not answer correctness. Recovery is a separately versioned experiment; original 77 answers/scores unchanged.')
    write_review_artifacts(output, dataset, questions, merged_answers, merged_judgments, merged_scores, failures)
    # Review writer also emits automatic_scores.jsonl; restore the exact historical prefix.
    checkpoint()
    (output/'generation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    verification = dict(source_files_unchanged=unchanged, source_sha256=protected,
        accepted_answer_rows_unchanged=all(r == merged_answers[i] for i, r in enumerate(original_answers)),
        accepted_score_rows_unchanged=all(r == merged_scores[i] for i, r in enumerate(original_scores)),
        new_ids_only=[r['query_id'] for r in answers], retrieval_repeated=False)
    (output/'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    manifest = dict(status=status, recovery_parent=str(source), query_ids=old['query_ids'],
        first_attempt=first, after_recovery=after, citation_retries=citation_retries,
        calibration_status='pending_human', source_files_unchanged=unchanged)
    (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=status, first_attempt=first, after_recovery=after, output=str(output)), indent=2))
    return int(status == 'failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/eval/hybridrag-en-v2')
    parser.add_argument('--query-ids', default='q058,q077,q090')
    parser.add_argument('--citation-retries', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT/'data/eval/runs'/('test80-recovery-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ'))
    return run(args.source, output, args.dataset, args.query_ids.split(','), args.citation_retries)


if __name__ == '__main__':
    raise SystemExit(main())
