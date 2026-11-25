import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_rag.evaluation.generation import GenerationJudgment, score_generation
from hybrid_rag.evaluation.refusal import (RefusalAssessment, answer_hash, assess_refusal,
    score_with_refusal)
from hybrid_rag.evaluation.schema import EvalQuestion

ROOT = Path(__file__).resolve().parents[2]


def question(answerable=False):
    return EvalQuestion(query_id='synthetic', dataset_version='test', topic_group='family',
        question='A and B?' if answerable else 'What is the private password?',
        category='concept' if answerable else 'unanswerable', products=[], split='dev', answerable=answerable,
        reference_answer='A and B' if answerable else 'Unknown.',
        reference_facts=[dict(fact_id='a', text='A'), dict(fact_id='b', text='B')] if answerable else [],
        gold_evidence=[dict(evidence_id='e', doc_id='d', section_id='s', source_path='p', heading_path=[],
            section_span=[0, 1], evidence_span=[0, 1], quote='A', supports=['a', 'b'], raw_sources=[])] if answerable else [],
        section_relevance={'s': 3} if answerable else {}, unanswerable_reason=None if answerable else 'Private unknown.')


def judgment(answer='Cannot determine that.', refused=False, answerable=False):
    return GenerationJudgment(query_id='synthetic', run_id='1', answer=answer, refused=refused,
        abstention_appropriate=not answerable, directly_answers_question=True, claims=[], citations=[],
        context_ids=['ctx1'], context_snapshot_hash='a'*64, reviewer='mock', reason='Synthetic judgment',
        rubric_version='grounded-v2', fact_assessments=[dict(fact_id='a', status='full', answer_quotes=['A'], reason='Asked and stated'),
            dict(fact_id='b', status='missing', reason='Not stated')] if answerable else [])


def annotation(answer, behavior, quotes=None):
    return RefusalAssessment(query_id='synthetic', answer_sha256=answer_hash(answer), response_type=behavior,
        answer_quotes=quotes or [answer], reason='Synthetic semantic annotation', reviewer='mock-only')


@pytest.mark.parametrize('flag', [False, True])
def test_semantic_refusal_independent_of_flag_and_legacy_score_is_unchanged(flag):
    j = judgment(refused=flag)
    original = j.model_dump()
    old = score_generation(question(), j, True)
    new = score_with_refusal(question(), j, annotation(j.answer, 'full_refusal'), True)
    assert old['correctness_bucket'] == int(flag)
    assert new['correctness_bucket'] == new['correct_abstention'] == 1
    assert new['refusal_conflict'] == (not flag)
    assert new['protocol_refused'] == flag and new['semantic_refused']
    assert j.model_dump() == original


def test_substantive_answer_cannot_pass_by_setting_refused_true():
    j = judgment(answer='The password is EXAMPLE_ONLY.', refused=True)
    score = score_with_refusal(question(), j, annotation(j.answer, 'substantive_answer'), True)
    assert score['correctness_bucket'] == score['correct_abstention'] == 0
    assert score['refusal_conflict'] and score['unsupported_answer'] == 1


@pytest.mark.parametrize('flag', [False, True])
def test_partial_answer_stays_partial_and_does_not_become_complete_refusal(flag):
    j = judgment(answer='A. Cannot determine B.', refused=flag, answerable=True)
    score = score_with_refusal(question(True), j, annotation(j.answer, 'partial_answer', ['A.', 'Cannot determine B.']), True)
    assert score['correctness_bucket'] == .5 and score['false_refusal'] == 0


def test_faithfulness_and_core_errors_are_not_bypassed_by_semantic_refusal():
    j = judgment(refused=False)
    j = GenerationJudgment.model_validate(dict(j.model_dump(), claims=[dict(claim_id='c', text='Cannot determine',
        answer_quote='Cannot determine', correct=True, supported_by_context=False)]))
    result = score_with_refusal(question(), j, annotation(j.answer, 'full_refusal'), True)
    assert result['correctness_bucket'] == 1 and result['faithfulness'] == 0 and result['answer_success'] == 0
    wrong = j.model_dump()
    wrong['claims'][0]['correct'] = False
    wrong.update(has_central_error=True, core_error_claim_ids=['c'])
    result = score_with_refusal(question(), GenerationJudgment.model_validate(wrong), annotation(j.answer, 'full_refusal'), True)
    assert result['correctness_bucket'] == 0


def test_annotation_rejects_other_answers_and_invented_anchors():
    a = annotation('Real answer.', 'full_refusal')
    with pytest.raises(ValueError, match='different answer'):
        a.validate_answer('synthetic', 'Different answer.')
    a.answer_quotes = ['GOLD_ONLY fact']
    with pytest.raises(ValueError, match='anchors'):
        a.validate_answer('synthetic', 'Real answer.')


def test_semantic_model_sees_answer_only_and_records_schema_retry():
    calls, traces = [], []
    class Model:
        def invoke(self, messages):
            calls.append(messages)
            payload = json.loads(messages[1]['content'])
            assert set(payload) == {'question', 'actual_answer'}
            assert 'abstained' not in payload and 'reference_answer' not in payload
            quote = 'invented' if len(calls) == 1 else 'Cannot determine.'
            return SimpleNamespace(content=json.dumps(dict(response_type='full_refusal', answer_quotes=[quote], reason='Missing measurement')),
                usage_metadata={'input_tokens': 10}, response_metadata={})
    a = assess_refusal('synthetic', 'What latency?', 'Cannot determine.', Model(), traces)
    assert a.response_type == 'full_refusal' and len(calls) == 2
    assert [t['valid'] for t in traces] == [False, True]


def script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location(name+'_test', ROOT/f'scripts/{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_evidence_rank_keeps_original_positions_and_real_range_gaps(monkeypatch):
    m = script('analyze_retrieval_ranks', monkeypatch)
    q = dict(section_relevance={'s': 3}, gold_evidence=[dict(section_id='s', section_span=[0, 10])])
    candidates = [dict(section_id='s', ranges=[[0, 3], [7, 10]]), dict(section_id='other', ranges=[[0, 10]])]
    assert m.evidence_ranks(q, candidates)['first_complete_evidence_rank'] is None
    candidates.append(dict(section_id='s', ranges=[[3, 7]]))
    result = m.evidence_ranks(q, candidates)
    assert result['first_section_rank'] == 1 and result['first_complete_evidence_rank'] == 3
    assert m.distribution([None, 2, 4]) == dict(total=3, found=2, missing=1, conditional_mean=3,
        conditional_median=3, top1=0, top5=2, top10=2)


def test_offline_replay_preserves_history_and_only_changes_semantic_conflict(monkeypatch, tmp_path):
    m = script('rescore_refusal_policy', monkeypatch)
    source = ROOT/'data/eval/runs/test80-recovery-20261116-v1'
    old = json.loads((source/'generation_report.json').read_text(encoding='utf-8'))
    answers = {r['query_id']: r for r in m.read_rows(source/'answers.jsonl')}
    entries = []
    for qid in old['query_ids']:
        if qid not in {'q115', 'q116', 'q117', 'q118', 'q119', 'q120'}:
            continue
        a = answers[qid]
        entries.append(dict(query_id=qid, answer_sha256=answer_hash(a['answer']), response_type='full_refusal',
            answer_quotes=[a['answer']], reason='Mock semantic label, not real model calibration', reviewer='mock-only'))
    for c in m.CONTROLS:
        quotes = [c['answer']] if c['expected'] != 'partial_answer' else c['answer'].split('. ')
        entries.append(dict(query_id=c['query_id'], answer_sha256=answer_hash(c['answer']), response_type=c['expected'],
            answer_quotes=quotes, reason='Mock control', reviewer='mock-only'))
    path = tmp_path/'annotations.jsonl'
    m.write_rows(path, entries)
    monkeypatch.setattr(m, 'make_chat_model', lambda **kwargs: pytest.fail('Offline replay must never initialize an API'))
    monkeypatch.setattr(m, 'summary', lambda scores: {})
    out = tmp_path/'replay'
    assert m.run(source, out, ROOT/'data/eval/hybridrag-en-v2', assessments=path) == 0
    new = json.loads((out/'generation_report.json').read_text(encoding='utf-8'))
    assert new['score_changes'] == [dict(query_id='q120', before=0, after=1)]
    assert new['semantic_api_usage']['calls'] == 0 and new['semantic_refusal_count'] == 6
    assert new['protocol_refusal_count'] == 5
    assert (out/'answers.jsonl').read_bytes() == (source/'answers.jsonl').read_bytes()
    assert json.loads((source/'generation_report.json').read_text(encoding='utf-8')) == old
