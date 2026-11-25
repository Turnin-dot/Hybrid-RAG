import copy,json,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
from hybrid_rag.retrieval.generation import generate,answer_question,CitationValidationError

CONTEXT=[dict(text='Use [interrupt](url).\n    builder = StateGraph(...)\n    graph = builder.compile()',source_path='p')]
def response(quote='Use [interrupt](url).'):
    return dict(answer='Use interrupt [1].',abstained=False,citations=[dict(source=1,quote=quote)])
class Model:
    def __init__(self,results):self.results=list(results);self.calls=[]
    def invoke(self,messages,**kwargs):
        self.calls.append((copy.deepcopy(messages),kwargs))
        obj=self.results.pop(0)
        if isinstance(obj,Exception):raise obj
        return SimpleNamespace(content=json.dumps(obj),usage_metadata={'input_tokens':10,'output_tokens':2,'total_tokens':12,'input_token_details':{'cache_read':3}},response_metadata={})

def test_first_success_needs_no_retry():
    m=Model([response()]);out=generate('Question',CONTEXT,m)
    assert len(m.calls)==1 and out['citation_retry_count']==0
    assert out['usage']['input_tokens']==10

@pytest.mark.parametrize('bad_count',[1,2])
def test_feedback_recovers_with_same_evidence_and_aggregate_usage(bad_count):
    m=Model([response('Use interrupt.')]*bad_count+[response()]);before=copy.deepcopy(CONTEXT)
    out=answer_question('Question',CONTEXT,m)
    assert len(m.calls)==bad_count+1 and CONTEXT==before
    assert all(c[0][:2]==m.calls[0][0][:2] for c in m.calls)
    assert all(len(c[0])==4 for c in m.calls[1:])
    assert all('citation_retries' not in kw for _,kw in m.calls)
    assert 'EXACTLY' in m.calls[-1][0][-1]['content']
    assert out['usage']['input_tokens']==10*(bad_count+1)
    assert out['usage']['input_token_details']['cache_read']==3*(bad_count+1)
    assert [a['output_validation'] for a in out['generation_attempts']]==['failed']*bad_count+['passed']
    assert out['generation_attempts'][0]['validation_details']['source']==1
    assert out['generation_attempts'][0]['raw_response']!=out['generation_attempts'][-1]['raw_response']

@pytest.mark.parametrize('retries',[0,2,3])
def test_exhaustion_preserves_all_attempts_and_strict_validation(retries):
    m=Model([response('Use interrupt.')]*(retries+1))
    with pytest.raises(CitationValidationError) as exc:generate('Question',CONTEXT,m,citation_retries=retries)
    assert len(m.calls)==retries+1 and len(exc.value.generation_attempts)==retries+1
    assert all(a['output_validation']=='failed' for a in exc.value.generation_attempts)

@pytest.mark.parametrize('value',[-1,4,2.5,True,'2'])
def test_invalid_retry_configuration_never_calls_model(value):
    m=Model([])
    with pytest.raises(ValueError):generate('Question',CONTEXT,m,citation_retries=value)
    assert not m.calls

def test_api_failure_is_not_retried_and_prior_failed_response_survives():
    m=Model([response('wrong'),TimeoutError('offline simulated timeout')])
    with pytest.raises(TimeoutError) as exc:generate('Question',CONTEXT,m)
    assert len(m.calls)==2 and len(exc.value.generation_attempts)==1

def test_schema_failure_does_not_trigger_citation_retry():
    m=Model([dict(answer='Bad',abstained=False,citations=[])])
    with pytest.raises(ValueError,match='supporting citations'):generate('Question',CONTEXT,m)
    assert len(m.calls)==1

def test_dry_run_and_empty_context_do_not_invoke_model():
    m=Model([])
    assert generate('Question',[],m)['abstained']
    assert generate('Question',CONTEXT)['dry_run'] and not m.calls

def test_recorder_retains_failed_and_successful_attempt_usage():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('retry_test_runner',root/'scripts/run_generation_eval.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    traces=[];model=Model([response('wrong'),response()])
    out=generate('Question',CONTEXT,module.InvocationRecorder(model,'q',traces))
    assert [t['output_validation'] for t in traces]==['failed','passed']
    assert [t['attempt'] for t in traces]==[1,2]
    assert sum(t['usage']['input_tokens'] for t in traces)==out['usage']['input_tokens']==20

def test_missing_usage_is_visible_not_fabricated():
    class NoUsage(Model):
        def invoke(self,*a,**kw):
            result=super().invoke(*a,**kw);result.usage_metadata=None;return result
    out=generate('Question',CONTEXT,NoUsage([response('wrong'),response()]))
    assert out['usage'] is None and out['usage_missing_attempts']==2
