import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_failed_output_is_recorded_without_changing_generation_input(monkeypatch):
    root=Path(__file__).resolve().parents[2];monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('context_experiment_test',root/'scripts/run_judge_context_experiment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from hybrid_rag.retrieval.generation import generate
    calls=[]
    class Model:
        def invoke(self,messages,**kwargs):
            calls.append((messages,kwargs))
            return SimpleNamespace(content='{"answer":"Invented [1]","abstained":false,"citations":[{"source":1,"quote":"not in source"}]}',usage_metadata={'input_tokens':10,'output_tokens':4},response_metadata={'finish_reason':'stop'})
    traces=[];wrapped=module.InvocationRecorder(Model(),'synthetic',traces)
    with pytest.raises(ValueError,match='actual retrieved source'):generate('Question',[dict(text='actual source',source_path='p')],wrapped,citation_retries=0)
    assert len(calls)==1 and traces[0]['messages']==calls[0][0]
    assert traces[0]['usage']['output_tokens']==4 and 'not in source' in traces[0]['raw_response']
