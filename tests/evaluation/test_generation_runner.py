"""Dev40 orchestration and review exports; all external models are replaced."""
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_rag.evaluation.dataset import load_questions
from hybrid_rag.evaluation.schema import EvalQuestion

ROOT=Path(__file__).resolve().parents[2]


def runner():
    spec=importlib.util.spec_from_file_location('generation_runner_test',ROOT/'scripts/run_generation_eval.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def questions():
    result=[]
    for i in range(1,41):
        qid=f'q{i:03d}';answerable=i<=36
        result.append(EvalQuestion(query_id=qid,dataset_version='test',topic_group=f'topic-{i}',
            question=f'Question {qid}?',category='concept' if answerable else 'unanswerable',products=[],split='dev',answerable=answerable,
            reference_answer='GOLD_ONLY reference fact' if answerable else 'Not documented.',
            reference_facts=[dict(fact_id='f1',text='GOLD_ONLY reference fact')] if answerable else [],
            gold_evidence=[dict(evidence_id='e1',doc_id='d',section_id='s'+qid,source_path='p.mdx',heading_path=[],
                section_span=[0,9],evidence_span=[0,9],quote='supported',supports=['f1'],
                raw_sources=[dict(source_path='p.mdx',line_start=2,line_end=3,sha256='a'*64)])] if answerable else [],
            unanswerable_reason=None if answerable else 'Absent from documentation.',
            section_relevance={'s'+qid:3} if answerable else {}))
    return result


class Engine:
    closed=False
    calls=[]
    def __init__(self):self.calls=[]
    def retrieve(self,question,**kwargs):
        self.calls.append(kwargs)
        qid=re.search(r'q\d+',question).group()
        return [dict(unit_id=f'{qid}-{i}',chunk_id=f'{qid}-{i}',section_id='s'+qid if i==0 else 'other',
            doc_id='d',source_path='p.mdx',heading_path=[],text='supported evidence.',ranges=[[0,19]],source_spans=[],rank=i+1)
            for i in range(kwargs['top_k'])]
    def close(self):self.closed=True


class Chat:
    def __init__(self,judge=False,fail=None):self.judge=judge;self.fail=fail;self.calls=[]
    def invoke(self,messages):
        self.calls.append(messages)
        if self.judge:
            data=json.loads(messages[1]['content'])
            if self.fail and self.fail in data['question']:raise TimeoutError('Synthetic judge failure')
            answerable=data['answerable']
            obj=dict(directly_answers_question=True,has_central_error=False,abstention_appropriate=not answerable,
                claims=[dict(claim_id='c1',text='supported',correct=True,supported_by_context=True,matches_fact_ids=['f1'],evidence_refs=['ctx1'])] if answerable else [],
                citations=[dict(claim_id='c1',context_id='ctx1',supports_claim=True)] if answerable else [],reason='Mock judgment for orchestration only.')
        else:
            assert 'GOLD_ONLY' not in str(messages), 'Reference answers must never reach the generator'
            qid=int(re.search(r'Question: Question q(\d+)',messages[1]['content']).group(1))
            obj=dict(answer='supported [1]' if qid<=36 else 'Not documented.',abstained=qid>36,
                citations=[dict(source=1,quote='supported')] if qid<=36 else [])
        return SimpleNamespace(content=json.dumps(obj),response_metadata={'model_name':'mock'},usage_metadata={'input_tokens':1,'output_tokens':1})


def setup(monkeypatch,tmp_path,fail=None):
    module=runner();engine=Engine();generator=Chat();judge=Chat(judge=True,fail=fail)
    dataset=tmp_path/'dataset';dataset.mkdir();(dataset/'questions.jsonl').write_text('fixture',encoding='utf-8')
    monkeypatch.setattr(module,'load_questions',lambda *a,**k:questions())
    monkeypatch.setattr(module,'load_index',lambda *a:(engine,{}))
    monkeypatch.setattr(module,'CrossEncoderReranker',lambda:object())
    monkeypatch.setattr(module,'make_chat_model',lambda **kw:judge if kw['max_tokens']==2400 else generator)
    args=['--dataset',str(dataset),'--subset','dev','--query-ids','all','--expected-count','40','--judge-version','legacy',
          '--allow-unreviewed','--output',str(tmp_path/'output')]
    return module,engine,generator,judge,args


def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def test_actual_dev_selection_has_36_answerable_and_4_unanswerable():
    selected=load_questions(ROOT/'data/eval/hybridrag-en-v2','dev',allow_unreviewed=True)
    assert len(selected)==40 and sum(q.answerable for q in selected)==36
    assert all(q.split=='dev' for q in selected)


@pytest.mark.parametrize('fail',[None,'q003'])
def test_all_40_review_and_judge_artifacts_survive_partial_failure(monkeypatch,tmp_path,fail):
    module,engine,generator,judge,args=setup(monkeypatch,tmp_path,fail)
    assert module.main(args+['--require-api'])==int(fail is not None)
    out=tmp_path/'output';report=json.loads((out/'generation_report.json').read_text(encoding='utf-8'))
    assert report['expected_questions']==report['answered_questions']==40
    assert report['unanswerable_questions']==4 and report['judged_questions']==40-int(fail is not None)
    assert engine.closed and all(c['top_k']==40 and c['candidate_k']==40 for c in engine.calls)
    assert len(generator.calls)==len(judge.calls)==40
    assert all('gold_evidence' in json.loads(c[1]['content']) for c in judge.calls)
    assert len(rows(out/'answers.jsonl'))==len(rows(out/'human_scores.template.jsonl'))==40
    assert all(r['score'] is None and r['reviewer'] is None for r in rows(out/'human_scores.template.jsonl'))
    assert sum(a['abstained'] for a in rows(out/'answers.jsonl'))==4
    diagnostics=rows(out/'retrieval_diagnostics.jsonl')
    assert all(len(d['candidates'])==40 and len(d['selected_chunk_ids'])==5 for d in diagnostics)
    review=(out/'human_review.md').read_text(encoding='utf-8')
    assert 'Mock judgment for orchestration' not in review and 'Mock judgment for orchestration' in (out/'judge_review.md').read_text(encoding='utf-8')
    assert 'source_snapshot/p.mdx:2' in review and 'half-open' in review
    lines=review.splitlines()
    for loc in json.loads((out/'review_locations.json').read_text(encoding='utf-8')):
        assert lines[loc['question_line']-1]==f"Question {loc['query_id']}?"
        assert lines[loc['answer_start_line']-1] in ['supported [1]','Not documented.']
    if fail:
        assert report['status']=='failed' and rows(out/'failures.jsonl')[0]['stage']=='judge'
        assert rows(out/'failures.jsonl')[0]['query_id']==fail
    else:assert report['status']=='diagnostic_automatic_judge_pending_human_calibration'
    with pytest.raises(FileExistsError):module.main(args)


def test_dry_run_checkpoints_every_question_without_any_api_call(monkeypatch,tmp_path):
    module,engine,gen,judge,args=setup(monkeypatch,tmp_path)
    monkeypatch.setattr(module,'make_chat_model',lambda **kw:pytest.fail('Dry-run must not initialize API clients'))
    assert module.main(args+['--dry-run'])==0
    out=tmp_path/'output'
    assert len(rows(out/'answers.jsonl'))==40 and not rows(out/'automatic_scores.jsonl')
    assert all(r['answer_status']=='dry_run' for r in rows(out/'human_scores.template.jsonl'))
    assert not gen.calls and not judge.calls and engine.closed
    assert json.loads((out/'generation_report.json').read_text())['status']=='dry_run_no_api'


def test_retry_only_missing_judge_preserves_answer_bytes_and_labels(monkeypatch,tmp_path):
    module,engine,gen,judge,args=setup(monkeypatch,tmp_path,'q003')
    assert module.main(args+['--require-api'])==1
    source=tmp_path/'output';target=tmp_path/'recovered';fresh_judge=Chat(judge=True)
    monkeypatch.setattr(module,'make_chat_model',lambda **kw:fresh_judge)
    code=module.retry_missing_judges(source,target,tmp_path/'dataset',questions(),'deepseek','test')
    assert code==0 and len(fresh_judge.calls)==1 and len(gen.calls)==40
    assert 'q003' in fresh_judge.calls[0][1]['content']
    for name in ['answers.jsonl','contexts.jsonl','retrieval_diagnostics.jsonl']:assert (source/name).read_bytes()==(target/name).read_bytes()
    assert len(rows(target/'judgments.jsonl'))==40
    assert rows(target/'judgments.jsonl')[:39]==rows(source/'judgments.jsonl')
    assert json.loads((source/'generation_report.json').read_text())['status']=='failed'
    assert json.loads((target/'generation_report.json').read_text())['status']=='diagnostic_automatic_judge_pending_human_calibration'


def test_missing_key_and_count_mismatch_fail_before_index_or_calls(monkeypatch,tmp_path):
    module,engine,gen,judge,args=setup(monkeypatch,tmp_path)
    def absent(**kw):raise ValueError('Missing key')
    monkeypatch.setattr(module,'make_chat_model',absent)
    monkeypatch.setattr(module,'load_index',lambda *a:pytest.fail('Preflight must fail before opening index'))
    with pytest.raises(ValueError,match='Real API configuration required'):module.main(args+['--require-api'])
    with pytest.raises(ValueError,match='Expected 39'):
        module.main([s if s!='40' else '39' for s in args])
    assert not (tmp_path/'output').exists() and not gen.calls and not judge.calls


def test_dev40_entry_selects_all_requires_real_api_and_never_test(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('dev40_entry',ROOT/'scripts/test_dev40_generation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    captured=[];monkeypatch.setattr(module,'run_evaluation',lambda argv:captured.append(argv) or 0)
    assert module.main(['--output',str(tmp_path/'run')])==0
    args=captured[0]
    for flag,value in [('--subset','dev'),('--query-ids','all'),('--expected-count','40')]:assert args[args.index(flag)+1]==value
    assert '--require-api' in args and 'test' not in args
