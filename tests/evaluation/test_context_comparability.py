import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location(name+'_audit_test', ROOT/f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_outputs_are_not_fabricated_scores_or_zero_cost(monkeypatch):
    m = script('audit_context_comparability', monkeypatch)
    scores = [dict(correctness_bucket=1, answer_success=1), dict(correctness_bucket=.5, answer_success=0)]
    s = m.summarize(scores, 3)
    assert s['conditional_mean'] == .75 and s['valid'] == 2 and s['missing'] == 1
    assert s['full_score_over_expected'] == 1/3
    assert s['missing_score_bounds'] == [.5, 2.5/3]
    resources = m.resource_summary([dict(usage={'input_tokens': 12, 'output_tokens': 3})], 3)
    assert resources['known_input_tokens'] == 12
    assert resources['total_input_tokens'] is None and resources['failed_call_cost'] is None


def test_pairing_requires_same_valid_ids_and_does_not_mix_missing_rows(monkeypatch):
    m = script('audit_context_comparability', monkeypatch)
    a = {k: dict(correctness_bucket=v) for k, v in [('a', 1), ('b', .5)]}
    b = {k: dict(correctness_bucket=v) for k, v in [('b', 1), ('c', 1)]}
    questions = {k: SimpleNamespace(topic_group='family') for k in ['a', 'b', 'c']}
    result = m.compare(a, b, questions)
    assert result['query_ids'] == ['b'] and result['paired']['mean'] == .5
    assert result['improved'] == ['b']
    with pytest.raises(ValueError, match='missing'):
        m.compare(a, b, questions, ['a', 'b'])


def test_context_and_source_mutation_rejected_even_when_signature_is_rewritten(monkeypatch):
    m = script('audit_context_comparability', monkeypatch)
    cc = [dict(chunk_id='c', text='Actual evidence.', source_path='p', source_spans=[dict(source_start=2, source_end=18)])]
    prompt = m.build_prompt('Question?', cc)
    messages = [dict(role='system', content=m.RUBRIC), dict(role='user', content=prompt[len(m.RUBRIC):])]
    contexts = [dict(cc[0], context_id='ctx1')]
    row = dict(question='Question?', answer='Actual evidence [1].', abstained=False,
        citations=[dict(source=1, quote='Actual evidence.')], contexts=contexts,
        prompt=prompt, messages=messages, prompt_sha256=m.digest(messages), context_snapshot_hash=m.digest(contexts),
        prompt_template_sha256=__import__('hashlib').sha256(m.RUBRIC.encode()).hexdigest(), dry_run=False, llm_invoked=True)
    m.verify_answer('Question?', row, cc)
    changed = copy.deepcopy(row)
    changed['contexts'][0]['source_spans'][0]['source_start'] = 0
    changed['context_snapshot_hash'] = m.digest(changed['contexts'])
    with pytest.raises(ValueError, match='mapping'):
        m.verify_answer('Question?', changed, cc)


def test_diagnostic_retry_cannot_replace_a_previously_valid_answer(monkeypatch):
    m = script('audit_context_comparability', monkeypatch)
    primary = {'a': dict(answer='original')}
    assert m.verify_retry(primary, dict(primary, b=dict(answer='recovered')), ['b']) == ['b']
    with pytest.raises(ValueError, match='accepted'):
        m.verify_retry(primary, {'a': dict(answer='edited')}, ['b'])
    with pytest.raises(ValueError, match='unexpected'):
        m.verify_retry(primary, dict(primary, c=dict(answer='extra')), ['b'])


def test_resume_cannot_inherit_a_new_recovery_policy(monkeypatch):
    m = script('run_judge_context_experiment', monkeypatch)
    current = dict(citation_retries=0, temperature=0, sdk_max_retries=1)
    m.verify_resume_policy({}, current)
    for key, value in [('citation_retries', 2), ('temperature', .5), ('sdk_max_retries', 3)]:
        with pytest.raises(ValueError, match='policy changed'):
            m.verify_resume_policy({}, dict(current, **{key: value}))


@pytest.mark.parametrize('retries', [0, 2])
def test_runner_passes_one_explicit_retry_policy_to_all_three_groups(monkeypatch, tmp_path, retries):
    """Synthetic orchestration only: no true questions, answers or API scoring."""
    m = script('run_judge_context_experiment', monkeypatch)
    source = tmp_path/'source'; source.mkdir()
    cc = [dict(chunk_id='c', section_id='s', rank=1, text='Evidence.', ranges=[[0, 9]])]
    ids = [f'synthetic-{i}' for i in range(40)]
    original = [dict(query_id=qid, dry_run=False, contexts=cc, context_snapshot_hash=m.digest(cc),
        prompt_template_sha256=m.hashlib.sha256(m.RUBRIC.encode()).hexdigest()) for qid in ids]
    diagnostics = [dict(query_id=qid, candidates=cc) for qid in ids]
    for name, rows in [('answers.jsonl', original), ('retrieval_diagnostics.jsonl', diagnostics)]:
        (source/name).write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    dataset_sha = m.hashlib.sha256((ROOT/'data/eval/hybridrag-en-v2/questions.jsonl').read_bytes()).hexdigest()
    (source/'generation_report.json').write_text(json.dumps(dict(dataset_sha256=dataset_sha)))
    judge_path = tmp_path/'judge'; judge_path.mkdir()
    phase1 = dict(status='complete', phase='judge',
        source_answers_sha256=m.hashlib.sha256((source/'answers.jsonl').read_bytes()).hexdigest(),
        extraction_prompt_sha256=m.hashlib.sha256(m.EXTRACT.encode()).hexdigest(),
        assessment_prompt_sha256=m.hashlib.sha256(m.ASSESS.encode()).hexdigest())
    (judge_path/'report.json').write_text(json.dumps(phase1))
    questions = [SimpleNamespace(query_id=qid, topic_group='synthetic-family', question='Synthetic question?', answerable=False, gold_evidence=[]) for qid in ids]
    monkeypatch.setattr(m, 'load_questions', lambda *a, **kw: questions)
    model_params = []
    monkeypatch.setattr(m, 'make_chat_model', lambda **kw: model_params.append(kw) or object())
    calls = []
    def generate(question, contexts, model, **kwargs):
        calls.append(kwargs)
        return dict(answer='Synthetic answer.', contexts=contexts, context_snapshot_hash=m.digest(contexts), usage=None)
    monkeypatch.setattr(m, 'generate', generate)
    monkeypatch.setattr(m, 'evaluate_answer', lambda q, row, judge, traces: SimpleNamespace(model_dump=lambda: dict(query_id=q.query_id)))
    monkeypatch.setattr(m, 'score_generation', lambda *a, **kw: dict(correctness_bucket=1))
    monkeypatch.setattr(m, 'cohort_statistics', lambda *args: {})
    monkeypatch.setattr(m, 'write_review_artifacts', lambda *args: None)
    monkeypatch.setattr(m, 'select_contexts', lambda q, candidates, strategy, count: candidates)
    fake_tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1])
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: fake_tokenizer)))
    output = tmp_path/'out'
    argv = ['experiment', '--phase', 'context', '--source', str(source), '--judge-results', str(judge_path), '--output', str(output)]
    if retries:
        argv += ['--citation-retries', str(retries)]
    monkeypatch.setattr(sys, 'argv', argv)
    assert m.main() == 0
    assert len(calls) == 120 and all(c['citation_retries'] == retries for c in calls)
    assert all(p['temperature'] == 0 and p['max_retries'] == 1 for p in model_params)
    config = json.loads((output/'config.json').read_text())
    assert config['citation_retries'] == retries
    assert (output/'code_snapshot/src/hybrid_rag/retrieval/generation.py').exists()
