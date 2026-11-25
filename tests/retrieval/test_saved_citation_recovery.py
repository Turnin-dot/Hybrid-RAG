import json
from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest

from hybrid_rag.retrieval.generation import generate, CitationValidationError


class Model:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(content=self.outputs.pop(0), usage_metadata={'input_tokens': 10}, response_metadata={})


def response(quote):
    return json.dumps(dict(answer='Supported [1].', abstained=False, citations=[dict(source=1, quote=quote)]))


def test_saved_failure_uses_two_extra_calls_and_does_not_rebill_old_response():
    model = Model([response('wrong'), response('exact evidence')])
    out = generate('Question', [dict(text='exact evidence')], model,
        previous_response=response('old wrong'), citation_retries=2)
    assert len(model.calls) == 2 and out['generation_call_count'] == 2
    assert out['citation_retry_count'] == 2 and out['recovered_from_saved_response']
    assert [a['attempt'] for a in out['generation_attempts']] == [2, 3]
    assert model.calls[0][2]['content'] == response('old wrong')
    assert model.calls[1][2]['content'] == response('wrong')
    assert out['usage']['input_tokens'] == 20


def test_saved_failure_exhausts_only_new_budget_and_valid_answer_cannot_be_overwritten():
    model = Model([response('wrong'), response('wrong')])
    with pytest.raises(CitationValidationError):
        generate('Q', [dict(text='evidence')], model, previous_response=response('old wrong'), citation_retries=2)
    assert len(model.calls) == 2
    with pytest.raises(ValueError, match='already valid'):
        generate('Q', [dict(text='evidence')], model, previous_response=response('evidence'))
    assert len(model.calls) == 2


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/eval/runs/test80-20261116T141248-384266Z'


def script(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec = importlib.util.spec_from_file_location('saved_recovery_test', ROOT/'scripts/recover_generation_failures.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('fail', [False, True])
def test_merge_preserves_77_rows_and_original_contexts_without_retrieval(monkeypatch, tmp_path, fail):
    m = script(monkeypatch)
    calls = []
    class RecoveryModel:
        def invoke(self, messages, **kwargs):
            calls.append(messages)
            original = messages[1]['content']
            # Quote an exact substring from the FIRST saved source, never the gold.
            text = original.split('[1] ', 1)[1].split('\n(Source:', 1)[0]
            quote = 'nonexistent quote' if fail else text[:40]
            return SimpleNamespace(content=response(quote), usage_metadata={'input_tokens': 10, 'output_tokens': 5}, response_metadata={})
    monkeypatch.setattr(m, 'make_chat_model', lambda **kwargs: RecoveryModel())
    judged = []
    def judge(q, row, model, traces):
        judged.append(q.query_id)
        return SimpleNamespace(model_dump=lambda: dict(query_id=q.query_id, run_id='run-1'))
    monkeypatch.setattr(m, 'evaluate_answer', judge)
    monkeypatch.setattr(m, 'score_generation', lambda *a, **k: dict(correctness_bucket=1, answer_success=True))
    monkeypatch.setattr(m, 'summary', lambda scores: {})
    # Statistics need the full metrics; separately tested shared implementation stays unchanged.
    monkeypatch.setattr(m, 'run_statistics', lambda *args: {})
    protected = m.directory_hashes(SOURCE)
    out = tmp_path/'recovered'
    assert m.run(SOURCE, out, ROOT/'data/eval/hybridrag-en-v2', ['q058', 'q077', 'q090']) == int(fail)
    report = json.loads((out/'generation_report.json').read_text(encoding='utf-8'))
    assert report['first_attempt']['generation_validity_rate'] == 77/80
    assert report['after_recovery']['valid_answers'] == (77 if fail else 80)
    assert len(calls) == (6 if fail else 3)
    assert judged == ([] if fail else ['q058', 'q077', 'q090'])
    for name in ['answers.jsonl', 'contexts.jsonl', 'judgments.jsonl', 'automatic_scores.jsonl', 'generation_traces.jsonl', 'judge_traces.jsonl']:
        assert (out/name).read_bytes().startswith((SOURCE/name).read_bytes())
    assert m.directory_hashes(SOURCE) == protected
    historical = {r['query_id']: r for r in m.read_rows(SOURCE/'retrieval_diagnostics.jsonl')}
    new_rows = m.read_rows(out/'answers.jsonl')[77:]
    for row in new_rows:
        old = historical[row['query_id']]
        selected = [next(c for c in old['candidates'] if c['chunk_id'] == cid) for cid in old['selected_chunk_ids']]
        assert row['contexts'] == [dict(c, context_id=f'ctx{i}') for i, c in enumerate(selected, 1)]
    assert len(m.read_rows(out/'human_scores.template.jsonl')) == 80


def test_preflight_rejects_accepted_id_without_creating_output(monkeypatch, tmp_path):
    m = script(monkeypatch)
    monkeypatch.setattr(m, 'make_chat_model', lambda **kwargs: pytest.fail('No model before validated recovery'))
    with pytest.raises(ValueError, match='exactly the missing'):
        m.run(SOURCE, tmp_path/'out', ROOT/'data/eval/hybridrag-en-v2', ['q058', 'q077', 'q009'])
    assert not (tmp_path/'out').exists()
