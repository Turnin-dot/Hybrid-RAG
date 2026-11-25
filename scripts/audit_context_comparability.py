"""Offline comparability audit of saved Dev context experiments; no API calls."""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hybrid_rag.evaluation.dataset import load_questions
from hybrid_rag.evaluation.generation import GenerationJudgment, score_generation
from hybrid_rag.evaluation.judge import EXTRACT, ASSESS, cohort_statistics
from hybrid_rag.evaluation.retrieval import covered, retrieval_metrics
from hybrid_rag.evaluation.statistics import paired_bootstrap, cluster_bootstrap
from hybrid_rag.retrieval.context_selection import select_contexts
from hybrid_rag.retrieval.generation import RUBRIC, build_prompt, _parse_answer

POLICIES = ['top5', 'top10', 'dynamic']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def unique(rows):
    result = {r['query_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate query IDs')
    return result


def resource_summary(answers, expected):
    known = [r['usage'] for r in answers if r.get('usage')]
    latency = sorted(r['latency_ms'] for r in answers if r.get('latency_ms') is not None)
    return dict(known_output_records=len(answers), missing_output_records=expected-len(answers),
        records_with_usage=len(known), records_without_usage=len(answers)-len(known),
        known_input_tokens=sum(r.get('input_tokens', 0) or 0 for r in known),
        known_output_tokens=sum(r.get('output_tokens', 0) or 0 for r in known),
        total_input_tokens=None if expected != len(known) else sum(r.get('input_tokens', 0) or 0 for r in known),
        failed_call_cost=None if expected != len(answers) else 0,
        mean_input_per_known_response=sum(r.get('input_tokens',0) or 0 for r in known)/len(known) if known else None,
        latency_ms=dict(n=len(latency),mean=statistics.mean(latency) if latency else None,
            p50=latency[(len(latency)-1)//2] if latency else None,
            p95=latency[int((len(latency)-1)*.95)] if latency else None),
        note='Known valid responses only. Missing call usage is unknown, never zero. SDK transport retries may be invisible.')


def summarize(scores, expected):
    values = [s['correctness_bucket'] for s in scores]
    total = sum(values)
    n = len(values)
    return dict(expected=expected, valid=n, missing=expected-n,
        conditional_mean=total/n if n else None, full_score_over_expected=sum(x == 1 for x in values)/expected,
        strict_success_over_expected=sum(s['answer_success'] for s in scores)/expected,
        missing_score_bounds=[total/expected, (total+expected-n)/expected],
        bounds_note='Hypothetical bounds for unavailable quality labels, not imputed scores or observed accuracy.')


def compare(first, second, questions, ids=None):
    keys = sorted(set(first) & set(second) if ids is None else set(ids))
    if not set(keys) <= set(first) & set(second):
        raise ValueError('Paired comparison includes missing outputs')
    a = {k: first[k]['correctness_bucket'] for k in keys}
    b = {k: second[k]['correctness_bucket'] for k in keys}
    return dict(n=len(keys), query_ids=keys,
        baseline_mean=statistics.mean(a.values()) if keys else None,
        policy_mean=statistics.mean(b.values()) if keys else None,
        paired=paired_bootstrap(a, b) if keys else None,
        topic_cluster=cluster_bootstrap([dict(query_id=k, topic_group=questions[k].topic_group, delta=b[k]-a[k])
            for k in keys], 'delta') if keys else None,
        improved=[k for k in keys if b[k] > a[k]], worsened=[k for k in keys if b[k] < a[k]],
        unchanged=[k for k in keys if b[k] == a[k]],
        limitation='Conditional on the exact same valid IDs; not an intention-to-test mean or repeated-model stability estimate.')


def verify_answer(question, row, contexts):
    expected_contexts = [dict(c, context_id=f'ctx{i}') for i, c in enumerate(contexts, 1)]
    if row['question'] != question or row['contexts'] != expected_contexts:
        raise ValueError('Question, selected text or source mapping changed')
    prompt = build_prompt(question, contexts)
    messages = [dict(role='system', content=RUBRIC), dict(role='user', content=prompt[len(RUBRIC):])]
    if row['messages'] != messages or row['prompt'] != prompt or row['prompt_sha256'] != digest(messages):
        raise ValueError('Saved generator input differs from selected context prompt')
    if row['context_snapshot_hash'] != digest(expected_contexts) or row['prompt_template_sha256'] != hashlib.sha256(RUBRIC.encode()).hexdigest():
        raise ValueError('Saved prompt/context signature changed')
    if row['dry_run'] or not row['llm_invoked']:
        raise ValueError('Mock/dry-run output is not valid real experiment evidence')
    _parse_answer(json.dumps({k: row[k] for k in ['answer', 'abstained', 'citations']}), contexts)


def verify_retry(primary, retried, failed_ids):
    if not all(k in retried and retried[k] == row for k, row in primary.items()):
        raise ValueError('Retry replaced an already accepted answer')
    added = set(retried)-set(primary)
    if not added <= set(failed_ids):
        raise ValueError('Retry added an unexpected question')
    return sorted(added)


def literal_constant(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError('Missing frozen literal: '+name)


def run(base, output, dataset):
    base, output = base.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError('Use a new output outside historical experiments')
    protected = {}
    def read(path, lines=False):
        raw = path.read_bytes()
        protected[str(path)] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.decode('utf-8').splitlines() if s.strip()] if lines else json.loads(raw)
    def file_hash(path):
        raw = path.read_bytes(); h = hashlib.sha256(raw).hexdigest(); protected[str(path)] = h; return h

    questions = {q.query_id: q for q in load_questions(dataset, 'dev', allow_unreviewed=True)}
    ids = set(questions)
    if len(ids) != 40 or sum(q.answerable for q in questions.values()) != 36:
        raise ValueError('Expected complete Dev40 with 36 answerable questions')
    config = read(base/'context/config.json')
    retry_config = read(base/'context-retry/config.json')
    source = Path(config['source'])
    if output.is_relative_to(source.resolve()) or source.resolve().is_relative_to(output):
        raise ValueError('Output must not alter the frozen source directory')
    source_report = read(source/'generation_report.json')
    primary_report = read(base/'context/report.json')
    retry_report = read(base/'context-retry/report.json')
    phase1_report = read(base/'judge/report.json')
    if phase1_report['status']!='complete' or phase1_report['source_answers_sha256']!=config['source_answers_sha256']:
        raise ValueError('Historical phase-one calibration was not complete')
    if file_hash(dataset/'questions.jsonl') != source_report['dataset_sha256']:
        raise ValueError('Dataset differs from frozen source')
    if file_hash(source/'answers.jsonl') != config['source_answers_sha256']:
        raise ValueError('Frozen source answers changed')
    original = unique(read(source/'answers.jsonl', True))
    candidates = unique(read(source/'retrieval_diagnostics.jsonl', True))
    if set(original) != ids or set(candidates) != ids:
        raise ValueError('Source membership differs from Dev40')
    for row in candidates.values():
        cc = row['candidates']
        if len(cc) != 40 or len({c['chunk_id'] for c in cc}) != 40 or [c['rank'] for c in cc] != list(range(1, 41)):
            raise ValueError('Expected unique ordered Top40 candidates')
    for key in ['source_answers_sha256', 'generator_model', 'generator_max_tokens', 'judge_max_tokens',
                'extraction_prompt_sha256', 'assessment_prompt_sha256', 'dynamic']:
        if config[key] != retry_config[key]:
            raise ValueError('Retry policy mismatch: '+key)
    for key, prompt in [('extraction_prompt_sha256', EXTRACT), ('assessment_prompt_sha256', ASSESS)]:
        if config[key] != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError('Judge prompt differs from historical experiment')
    snapshot = base/'context/code_snapshot'
    for relative, expected in config['code_sha256'].items():
        if file_hash(snapshot/relative) != expected:
            raise ValueError('Archived experiment code differs from its recorded hash')
    gen_path = snapshot/'src/hybrid_rag/retrieval/generation.py'
    expected_gen = config['protected_sha256'][str(ROOT/'src/hybrid_rag/retrieval/generation.py')]
    if file_hash(gen_path) != expected_gen or literal_constant(gen_path, 'RUBRIC') != RUBRIC:
        raise ValueError('Frozen generator snapshot/prompt mismatch')
    for relative in ['src/hybrid_rag/evaluation/judge.py', 'src/hybrid_rag/retrieval/generation.py']:
        file_hash(snapshot/relative)
    for name, value in [('EXTRACT', EXTRACT), ('ASSESS', ASSESS)]:
        if literal_constant(snapshot/'src/hybrid_rag/evaluation/judge.py', name) != value:
            raise ValueError('Archived Judge prompt changed')
    frozen_path = snapshot/'src/hybrid_rag/evaluation/generation.py'
    spec = importlib.util.spec_from_file_location('verified_frozen_context_scoring', frozen_path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    # Local tokenizer only: no embeddings, remote models or API clients.
    from transformers import AutoTokenizer
    tok_path = ROOT/'data/models/bge-base-en-v1.5-tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    tokenizer_hashes = {p.name: file_hash(p) for p in tok_path.iterdir() if p.is_file()}
    policy_data, summaries, corrections, rows = {}, {}, [], {qid: dict(query_id=qid, question=questions[qid].question,
        answerable=questions[qid].answerable, policies={}) for qid in sorted(ids)}
    snapshot_reproductions = 0
    metadata = {'generation': {}, 'judge': {}}
    for policy in POLICIES:
        path = base/'context'/policy
        answers = unique(read(path/'answers.jsonl', True))
        judgments = unique(read(path/'judgments.jsonl', True))
        saved_scores = unique(read(path/'scores.jsonl', True))
        selection = unique(read(path/'selection.jsonl', True))
        failures = read(path/'failures.jsonl', True)
        traces = read(path/'traces.jsonl', True)
        if set(selection) != ids or set(answers) != set(judgments) or set(answers) != set(saved_scores):
            raise ValueError('Incomplete/mismatched policy records')
        if set(answers) | {r['query_id'] for r in failures} != ids:
            raise ValueError('Some attempted questions have neither answer nor failure')
        computed, chosen = {}, {}
        judge_seen = set()
        for trace in traces:
            qid = trace['query_id']; q = questions[qid]
            payload = json.loads(trace['messages'][1]['content'])
            if payload['question'] != q.question or payload['actual_answer'] != answers[qid]['answer']:
                raise ValueError('Judge evaluated different question/answer')
            prompt = EXTRACT if trace['stage'] == 'answer_only_extraction' else ASSESS
            if trace['messages'][0]['content'] != prompt or trace['prompt_sha256'] != digest(trace['messages']):
                raise ValueError('Judge request/rubric signature mismatch')
            if trace['stage'] == 'answer_only_extraction' and set(payload) != {'question', 'actual_answer'}:
                raise ValueError('Answer-only extraction leaked reference information')
            if trace['stage'] != 'answer_only_extraction' and payload['actual_contexts'] != answers[qid]['contexts']:
                raise ValueError('Judge received a different context')
            judge_seen.add((qid, trace['stage']))
        for qid in sorted(ids):
            q = questions[qid]; sel = selection[qid]
            selected = select_contexts(q.question, candidates[qid]['candidates'], policy, count)
            chosen[qid] = selected
            if [c['chunk_id'] for c in selected] != sel['chunk_ids'] or len(selected) != sel['context_count']:
                raise ValueError('Context selection cannot reproduce historical result')
            if count(build_prompt(q.question, selected)) != sel['prompt_tokens_bge_proxy']:
                raise ValueError('Tokenizer proxy count cannot reproduce historical selection')
            ranks = [c['rank'] for c in selected]
            metrics = retrieval_metrics(q, [dict(section_id=c['section_id'], ranges=c['ranges']) for c in selected], len(selected)) if q.answerable else None
            hits = [e for e in q.gold_evidence if covered(e.section_span,
                [span for c in selected if c['section_id'] == e.section_id for span in c['ranges']])]
            recall = len(hits)/len(q.gold_evidence) if q.gold_evidence else None
            if recall != sel['evidence_recall']:
                raise ValueError('Evidence metric differs from saved selection')
            cell = dict(valid=qid in answers, selected_candidate_ranks=ranks, selected_chunk_ids=sel['chunk_ids'],
                context_count=len(selected), prompt_tokens_bge_proxy=sel['prompt_tokens_bge_proxy'],
                evidence_recall=recall, retrieval_metrics=metrics,
                missing_fact_ids=sorted({f.fact_id for f in q.reference_facts if f.required}-{f for e in hits for f in e.supports}),
                failure=next((r for r in failures if r['query_id'] == qid), None),
                score=None, input_tokens=None, output_tokens=None)
            if qid in answers:
                row = answers[qid]
                verify_answer(q.question, row, selected)
                j = judgments[qid]
                if j['answer'] != row['answer'] or j['context_snapshot_hash'] != row['context_snapshot_hash']:
                    raise ValueError('Judgment answer/context does not match saved generation')
                for stage in ['answer_only_extraction', 'evidence_assessment']:
                    if (qid, stage) not in judge_seen:
                        raise ValueError('Missing Judge stage')
                old_score = frozen.score_generation(q, frozen.GenerationJudgment.model_validate(j), True)
                if any(saved_scores[qid][k] != v for k, v in old_score.items()):
                    raise ValueError('Frozen scoring snapshot cannot reproduce historical metrics')
                snapshot_reproductions += 1
                now = score_generation(q, GenerationJudgment.model_validate(j), True)
                computed[qid] = dict(query_id=qid, topic_group=q.topic_group, **now)
                for key, value in now.items():
                    if saved_scores[qid][key] != value:
                        corrections.append(dict(query_id=qid, strategy=policy, metric=key, before=saved_scores[qid][key], after=value))
                usage = row.get('usage') or {}
                cell.update(score=now, answer=row['answer'], citations=row['citations'],
                    answer_sha256=hashlib.sha256(row['answer'].encode()).hexdigest(), context_snapshot_hash=row['context_snapshot_hash'],
                    input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'),
                    generation_latency_ms=row.get('latency_ms'))
            rows[qid]['policies'][policy] = cell
        g_meta = Counter((r.get('response_metadata', {}).get('model_name'), r.get('response_metadata', {}).get('system_fingerprint')) for r in answers.values())
        j_meta = Counter((r.get('metadata', {}).get('model_name'), r.get('metadata', {}).get('system_fingerprint')) for r in traces)
        metadata['generation'][policy] = [dict(model=k[0], fingerprint=k[1], records=n) for k, n in sorted(g_meta.items())]
        metadata['judge'][policy] = [dict(model=k[0], fingerprint=k[1], records=n) for k, n in sorted(j_meta.items())]
        complete = list(computed.values())
        eval_cells = [rows[qid]['policies'][policy] for qid, q in questions.items() if q.answerable]
        summaries[policy] = dict(summarize(complete, len(ids)),
            cohorts=cohort_statistics(list(questions.values()), complete, list(answers.values())),
            evidence_recall=sum(c['evidence_recall'] for c in eval_cells)/len(eval_cells),
            required_fact_coverage=sum(c['retrieval_metrics']['required_fact_coverage'] for c in eval_cells)/len(eval_cells),
            all_required_facts_count=sum(c['retrieval_metrics']['all_required_facts'] for c in eval_cells),
            resources=resource_summary(list(answers.values()), len(ids)),
            judge_resources=resource_summary([dict(usage=t.get('usage'),latency_ms=t.get('latency_ms')) for t in traces],len(traces)),
            selected_count_distribution=dict(Counter(str(len(chosen[qid])) for qid in ids)),
            maximum_prompt_bge_proxy=max(c['prompt_tokens_bge_proxy'] for c in selection.values()),
            failures=failures)
        retry_path = base/'context-retry'/policy
        retried = unique(read(retry_path/'answers.jsonl', True))
        added = verify_retry(answers, retried, [r['query_id'] for r in failures])
        retry_traces = read(retry_path/'generation_traces.jsonl', True) if (retry_path/'generation_traces.jsonl').exists() else []
        retry_failures = read(retry_path/'failures.jsonl', True)
        for qid in added:
            verify_answer(questions[qid].question, retried[qid], chosen[qid])
        for trace in retry_traces:
            qid = trace['query_id']
            prompt = build_prompt(questions[qid].question, chosen[qid])
            expected = [dict(role='system', content=RUBRIC), dict(role='user', content=prompt[len(RUBRIC):])]
            if trace['messages'] != expected:
                raise ValueError('Diagnostic retry modified original question/context/prompt')
        summaries[policy]['retry_diagnostic'] = dict(added_query_ids=added, valid_answers=len(retried),
            failures=retry_failures, recorded_new_generation_calls=len(retry_traces),
            note='Not merged into primary efficacy or first-attempt validity statistics')
        policy_data[policy] = dict(scores=computed, answers=answers, selection=chosen, traces=traces)

    common = sorted(set.intersection(*(set(policy_data[s]['scores']) for s in POLICIES)))
    paired = {s: dict(pairwise=compare(policy_data['top5']['scores'], policy_data[s]['scores'], questions),
        common_three=compare(policy_data['top5']['scores'], policy_data[s]['scores'], questions, common)) for s in ['top10', 'dynamic']}
    common_resources = {s: resource_summary([policy_data[s]['answers'][qid] for qid in common], len(common)) for s in POLICIES}
    phase1_answers = unique(read(base/'judge/frozen_top5/answers.jsonl', True))
    phase1_judges = unique(read(base/'judge/frozen_top5/judgments.jsonl', True))
    variation = []
    same_answers = []
    for qid in sorted(ids):
        baseline = policy_data['top5']['answers'][qid]
        if phase1_answers[qid]['contexts'] != baseline['contexts']:
            raise ValueError('Fresh Top5 changed original baseline context')
        old_j = GenerationJudgment.model_validate(phase1_judges[qid])
        old_s = score_generation(questions[qid], old_j, True)
        new_s = policy_data['top5']['scores'][qid]
        same_answer = phase1_answers[qid]['answer'] == baseline['answer']
        if same_answer:
            same_answers.append(qid)
        if old_s['correctness_bucket'] != new_s['correctness_bucket']:
            variation.append(dict(query_id=qid, same_context=True, same_answer=same_answer,
                old_score=old_s['correctness_bucket'], fresh_top5_score=new_s['correctness_bucket'],
                interpretation='Same answer/context Judge variation' if same_answer else 'Answer changed; generator and Judge effects not separately identified'))
    result = dict(status='offline_audit_complete_keep_top5', primary=summaries, common_three_ids=common,
        paired=paired, common_three_resources=common_resources, metric_corrections=corrections,
        frozen_score_reproductions=snapshot_reproductions, metadata=metadata, baseline_variation=variation,
        same_answer_as_original=same_answers, per_question=[rows[qid] for qid in sorted(ids)],
        protocol=dict(dataset='Dev40 only; 36 answerable and 4 unanswerable',
            requested_model=config['generator_model'], generator_max_tokens=config['generator_max_tokens'], judge_max_tokens=config['judge_max_tokens'],
            temperature='0 in verified archived factory; per-request transport parameters not separately saved',
            initial_citation_retries='0: archived generator makes one logical request',
            sdk_retries='max_retries=1 in archived factory; transport attempts not fully observable',
            judge_schema_retries='one correction attempt per extraction/assessment',
            scoring='All groups recomputed under the same current grounded-v2 metric definitions; semantic-refusal-v1 NOT selectively applied',
            dynamic=config['dynamic'], candidate_pool=config['candidate_pool'],
            tokenizer_current_files_sha256=tokenizer_hashes),
        gaps=['Seven initial rejected responses and their exact usage/latency are unavailable; cannot reconstruct true total cost offline.',
            'New three-policy answers do not have independent paired human scores; original baseline human labels must not be transferred.',
            'Single fixed-order run; fingerprints match but remote weights are not pinned and sampling/Judge variability is not measured.',
            'Diagnostic retry orchestration hash differs from primary snapshot and has no matching archived script; preserved requests verify input equality, not full implementation identity.',
            'This comparison measures generation and selection, not online retrieval latency.',
            'Shared generation default later changed to 2; historical comparisons used 0. Updated experiment runner now explicitly pins 0 and rejects resume policy changes.'],
        historical_status=dict(primary=primary_report['status'],diagnostic_retry=retry_report['status']),
        checks=dict(questions_and_gold_frozen=True, candidates_unique_ordered40=True, selections_reproduced=120,
            prompt_token_proxy_counts_reproduced=120, accepted_contexts_and_source_ranges_verified=snapshot_reproductions,
            historical_frozen_scores_reproduced=snapshot_reproductions, judge_inputs_and_rubrics_verified=True,
            retry_preserves_accepted_answers=True, retry_requests_same_original_context=True,
            api_calls=0, new_generation_calls=0))
    for path, expected in protected.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Audit input changed during read: '+path)
    output.mkdir(parents=True)
    (output/'comparability.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'per_question.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in result['per_question']), encoding='utf-8')
    (output/'input_hashes.json').write_text(json.dumps(protected, indent=2), encoding='utf-8')
    write_markdown(output, result, questions, base)
    print(json.dumps(dict(output=str(output), checks=result['checks'], primary={s: {k: v for k, v in r.items()
        if k in ['valid', 'missing', 'conditional_mean', 'missing_score_bounds', 'evidence_recall', 'required_fact_coverage', 'all_required_facts_count']}
        for s, r in summaries.items()}, corrections=corrections), ensure_ascii=False, indent=2))
    return result


def write_markdown(output, report, questions, base):
    md = ['# Dev40 上下文策略逐题对照（自动指标）\n\n',
        '只比较主实验首次结果。缺失答案不填0分；原人工标签不转移到新生成答案。E=完整Gold引文覆盖；F=必需事实覆盖，均按真实来源范围离线重算。\n\n',
        '| 题号 | Top5分数 | Top10分数 | 动态分数 | E：5/10/动态 | F：5/10/动态 |\n| --- | --- | --- | --- | --- | --- |\n']
    details = ['# 已有三组答案对照\n\n自动分数与答案同时展示；这不是新的盲审或独立人工标注。\n\n']
    for row in report['per_question']:
        qid = row['query_id']
        cells = [row['policies'][s] for s in POLICIES]
        value = lambda c: str(c['score']['correctness_bucket']) if c['valid'] else '缺失'
        ev = '/'.join(f"{c['evidence_recall']:.2f}" if c['evidence_recall'] is not None else 'n/a' for c in cells)
        facts = '/'.join(f"{c['retrieval_metrics']['required_fact_coverage']:.2f}" if c['retrieval_metrics'] else 'n/a' for c in cells)
        md.append(f"| [{qid}](answer_comparison.md#{qid}) | {value(cells[0])} | {value(cells[1])} | {value(cells[2])} | {ev} | {facts} |\n")
        details.append(f"## {qid}\n\nQuestion: {row['question']}\n\nRequired reference facts:\n\n")
        details.extend(f"- {f.fact_id}: {f.text}\n" for f in questions[qid].reference_facts if f.required)
        for strategy in POLICIES:
            c = row['policies'][strategy]
            details.append(f"\n### {strategy}\n\nSelected original candidate ranks: {c['selected_candidate_ranks']}.\n\n")
            details.append(c.get('answer', '(No valid first-attempt answer; do not assign a quality score.)')+'\n\n')
            if c['valid']:
                details.append('Automatic metrics: '+json.dumps(c['score'], ensure_ascii=False)+'\n\n')
            if c['failure']:
                details.append('Failure: '+json.dumps(c['failure'], ensure_ascii=False)+'\n\n')
            source = base/'context'/strategy/'human_review.md'
            details.append(f"[Original question, actual sources and Gold locations]({source.as_posix()}#{qid})\n\n")
    (output/'per_question.md').write_text(''.join(md), encoding='utf-8')
    (output/'answer_comparison.md').write_text(''.join(details), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'data/eval/runs/judge-context-v3')
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/eval/hybridrag-en-v2')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(argv)
    run(a.root, a.output, a.dataset)


if __name__ == '__main__':
    main()
