"""Synthetic Test80 orchestration and real membership preflight; no API calls."""
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
import pytest
from hybrid_rag.evaluation.schema import EvalQuestion

ROOT=Path(__file__).resolve().parents[2]


def script(name,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location(name+'_test',ROOT/f'scripts/{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def synthetic_questions():
    result=[]
    for i in range(1,81):
        qid=f'q{i:03d}';answerable=i<=74
        result.append(EvalQuestion(query_id=qid,dataset_version='synthetic-test80',topic_group=f'family-{i}',
            question=f'Question {qid}?',category='concept' if answerable else 'unanswerable',products=[],split='test',answerable=answerable,
            reference_answer='GOLD_ONLY reference' if answerable else 'Not documented.',reference_facts=[dict(fact_id='f1',text='GOLD_ONLY reference')] if answerable else [],
            gold_evidence=[dict(evidence_id='e',doc_id='d',section_id='s'+qid,source_path='p.mdx',heading_path=[],section_span=[0,9],evidence_span=[0,9],quote='supported',supports=['f1'],raw_sources=[])] if answerable else [],
            section_relevance={'s'+qid:3} if answerable else {},unanswerable_reason=None if answerable else 'Private unspecified information.'))
    return result


class Engine:
    def __init__(self):self.closed=False;self.calls=[]
    def retrieve(self,q,**kwargs):
        self.calls.append(kwargs);qid=re.search(r'q\d+',q).group()
        return [dict(chunk_id=f'{qid}-{i}',unit_id=f'{qid}-{i}',doc_id='d',section_id='s'+qid if i==0 else 'other',source_path='p.mdx',text='supported evidence.',heading_path=[],ranges=[[0,19]],source_spans=[],rank=i+1) for i in range(kwargs['top_k'])]
    def close(self):self.closed=True


class Generator:
    def __init__(self,fail):self.fail=fail;self.calls=[]
    def invoke(self,messages,**kwargs):
        self.calls.append(messages);assert 'GOLD_ONLY' not in str(messages)
        qid=int(re.search(r'Question: Question q(\d+)',messages[1]['content']).group(1))
        obj=dict(answer='supported [1]' if qid<=74 else 'Not documented.',abstained=qid>74,citations=[dict(source=1,quote='invalid quote' if self.fail and qid==50 else 'supported')] if qid<=74 else [])
        return SimpleNamespace(content=json.dumps(obj),usage_metadata={'input_tokens':10,'output_tokens':5},response_metadata={'model_name':'mock','finish_reason':'stop'})


class Judge:
    def __init__(self,fail=False):self.fail=fail;self.calls=[]
    def invoke(self,messages,**kwargs):
        self.calls.append(messages);p=json.loads(messages[1]['content'])
        if self.fail and p['question']=='Question q050?':raise TimeoutError('Synthetic transport failure')
        if set(p)=={'question','actual_answer'}:
            obj=dict(claims=[dict(claim_id='c',text='supported',answer_quote='supported')] if p['actual_answer'].startswith('supported') else [])
        else:
            yes=p['answerable']
            obj=dict(directly_answers_question=True,has_central_error=False,core_error_claim_ids=[],abstention_appropriate=not yes,
                claims=[dict(claim_id='c',correct=True,supported_by_context=True,matches_fact_ids=['f1'],evidence_refs=['ctx1'])] if yes else [],
                citations=[dict(claim_id='c',context_id='ctx1',supports_claim=True)] if yes else [],
                fact_assessments=[dict(fact_id='f1',status='full',answer_quotes=['supported'],reason='Synthetic transport contract')] if yes else [],reason='Mock only')
        return SimpleNamespace(content=json.dumps(obj),usage_metadata={'input_tokens':10,'output_tokens':5},response_metadata={'model_name':'mock'})


def setup_run(monkeypatch,tmp_path,fail):
    m=script('run_generation_eval',monkeypatch);qs=synthetic_questions();engine=Engine();gen=Generator(fail=='generation');judge=Judge(fail=='judge')
    dataset=tmp_path/'dataset';dataset.mkdir();(dataset/'questions.jsonl').write_text('synthetic source',encoding='utf-8')
    frozen=tmp_path/'frozen.json';frozen.write_text('{"note":"synthetic freeze"}',encoding='utf-8')
    monkeypatch.setattr(m,'load_questions',lambda *a,**k:qs)
    monkeypatch.setattr(m,'load_index',lambda *a:(engine,{}));monkeypatch.setattr(m,'CrossEncoderReranker',lambda:object())
    monkeypatch.setattr(m,'make_chat_model',lambda **kw:gen if kw['max_tokens']==768 else judge)
    argv=['--dataset',str(dataset),'--subset','test','--query-ids','all','--expected-count','80','--require-api','--allow-unreviewed','--citation-retries','0','--citation-retries','0','--freeze-manifest',str(frozen),'--output',str(tmp_path/'run')]
    return m,qs,engine,gen,judge,argv


def rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def current_freeze_fixture(module, monkeypatch, tmp_path):
    """Test successful preflight against a temporary current-code manifest.

    The production manifest must remain frozen to the historical experiment;
    later recovery code is intentionally rejected by that original manifest.
    """
    frozen=json.loads(module.FREEZE.read_text(encoding='utf-8'))
    frozen['files_sha256']={p:module.file_sha256(module.ROOT/p) for p in frozen['files_sha256']}
    path=tmp_path/'current-code-fixture.json'
    path.write_text(json.dumps(frozen),encoding='utf-8')
    monkeypatch.setattr(module,'FREEZE',path)


@pytest.mark.parametrize('fail',[None,'generation','judge'])
def test_all80_include_no_answers_and_preserve_rejected_output(monkeypatch,tmp_path,fail):
    m,qs,engine,gen,judge,argv=setup_run(monkeypatch,tmp_path,fail)
    assert m.main(argv)==int(fail is not None)
    out=tmp_path/'run';report=json.loads((out/'generation_report.json').read_text(encoding='utf-8'))
    assert report['expected_questions']==80 and report['answerable_questions']==74 and report['unanswerable_questions']==6
    assert report['judged_questions']==80-int(fail is not None) and report['judge_version']=='grounded-v2'
    assert len(gen.calls)==80 and engine.closed and all(c['top_k']==40 and c['candidate_k']==40 for c in engine.calls)
    assert len(rows(out/'human_scores.template.jsonl'))==80
    assert all(r['score'] is None for r in rows(out/'human_scores.template.jsonl'))
    assert len(rows(out/'generation_traces.jsonl'))==80 and report['statistics']['api_usage']['generation']['input_tokens']==800
    assert report['statistics']['cohorts']['unanswerable']['n']==6 and report['statistics']['cohorts']['unanswerable']['correct_abstention_rate']==1
    assert (out/'frozen_config.snapshot.json').read_bytes()==(tmp_path/'frozen.json').read_bytes()
    if fail:
        assert report['status']=='failed' and report['failures'][0]['stage']==fail
        assert report['statistics']['valid_and_full_score_rate']==79/80
        if fail=='generation':
            t=next(r for r in rows(out/'generation_traces.jsonl') if r['query_id']=='q050')
            assert 'invalid quote' in t['raw_response'] and t['output_validation']=='failed'
        else:assert report['statistics']['api_usage']['judge']['calls_without_usage']==1


def test_grounded_judge_retry_cannot_silently_use_legacy(monkeypatch,tmp_path):
    m,qs,engine,gen,judge,argv=setup_run(monkeypatch,tmp_path,'judge');assert m.main(argv)==1
    source=tmp_path/'run';output=tmp_path/'recovered';fresh=Judge();construct=[]
    def factory(**kw):construct.append(kw);return fresh
    monkeypatch.setattr(m,'make_chat_model',factory)
    assert m.retry_missing_judges(source,output,tmp_path/'dataset',qs,'deepseek','deepseek-chat')==0
    assert construct[0]['max_tokens']==4096 and len(fresh.calls)==2 and len(gen.calls)==80
    assert len(rows(output/'judgments.jsonl'))==80
    for name in ['answers.jsonl','contexts.jsonl','generation_traces.jsonl','frozen_config.snapshot.json']:assert (source/name).read_bytes()==(output/name).read_bytes()
    assert json.loads((source/'generation_report.json').read_text())['status']=='failed'


def test_actual_test80_preflight_and_plan_do_not_call_models(monkeypatch,tmp_path):
    m=script('test_holdout80_generation',monkeypatch)
    current_freeze_fixture(m,monkeypatch,tmp_path)
    monkeypatch.setattr(m,'run_evaluation',lambda *a:pytest.fail('Plan must not run retrieval or models'))
    assert m.main(['--plan-only','--output',str(tmp_path/'plan')])==0
    p=json.loads((tmp_path/'plan/preflight.json').read_text())
    assert p['questions']==80 and p['answerable']==74 and p['unanswerable']==6 and p['dev_test_isolation_verified']
    assert set(p['query_ids']).isdisjoint({q.query_id for q in m.load_questions(m.ROOT/'data/eval/hybridrag-en-v2','dev',allow_unreviewed=True)})
    assert not (tmp_path/'plan/answers.jsonl').exists()


def test_freeze_rejects_settings_and_file_drift_before_execution(monkeypatch,tmp_path):
    m=script('test_holdout80_generation',monkeypatch);frozen=json.loads(m.FREEZE.read_text(encoding='utf-8'))
    changed=tmp_path/'changed.json';frozen['settings']['top_k']=10;changed.write_text(json.dumps(frozen),encoding='utf-8')
    monkeypatch.setattr(m,'FREEZE',changed)
    with pytest.raises(ValueError,match='settings'):m.preflight()
    frozen['settings']['top_k']=5;frozen['files_sha256']['scripts/run_generation_eval.py']='0'*64;changed.write_text(json.dumps(frozen),encoding='utf-8')
    with pytest.raises(ValueError,match='Frozen file changed'):m.preflight()


def test_test80_entry_has_one_frozen_strategy_and_checks_after_run(monkeypatch,tmp_path):
    m=script('test_holdout80_generation',monkeypatch);captured=[]
    current_freeze_fixture(m,monkeypatch,tmp_path)
    def fake(argv):
        captured.extend(argv);Path(argv[argv.index('--output')+1]).mkdir();return 0
    monkeypatch.setattr(m,'run_evaluation',fake)
    assert m.main(['--output',str(tmp_path/'out')])==0
    for flag,value in [('--subset','test'),('--query-ids','all'),('--expected-count','80'),('--top-k','5'),('--candidate-k','40'),('--judge-version','grounded-v2')]:assert captured[captured.index(flag)+1]==value
    assert '--require-api' in captured and '--freeze-manifest' in captured and (tmp_path/'out/freeze_verification.json').exists()


def test_midrun_freeze_drift_marks_result_failed(monkeypatch,tmp_path):
    m=script('test_holdout80_generation',monkeypatch);calls=[]
    def check():
        calls.append(1)
        if len(calls)>1:raise ValueError('Frozen file changed during run')
        return {'freeze_sha256':'a'*64}
    monkeypatch.setattr(m,'preflight',check)
    def fake(argv):
        out=Path(argv[argv.index('--output')+1]);out.mkdir();(out/'generation_report.json').write_text('{"status":"pending"}');return 0
    monkeypatch.setattr(m,'run_evaluation',fake)
    with pytest.raises(ValueError):m.main(['--output',str(tmp_path/'out')])
    assert json.loads((tmp_path/'out/generation_report.json').read_text())['status']=='failed_frozen_inputs_changed'
