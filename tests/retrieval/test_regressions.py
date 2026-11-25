"""Regression cases from retrieval-review-20261116; no API/model calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid_rag.evaluation.schema import EvalQuestion
from hybrid_rag.evaluation.retrieval import retrieval_metrics
from hybrid_rag.retrieval.engine import RetrievalEngine
from hybrid_rag.retrieval.units import RetrievalUnit,build_units
from hybrid_rag.retrieval.embeddings import EmbeddingModel
from hybrid_rag.retrieval.generation import generate

ROOT=Path(__file__).resolve().parents[2]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'scripts/{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def question():
    return EvalQuestion(query_id='test',dataset_version='test',topic_group='test',question='What?',category='concept',products=[],split='dev',answerable=True,
        reference_answer='Fact',reference_facts=[dict(fact_id='f1',text='Fact')],section_relevance={'s':3},
        gold_evidence=[dict(evidence_id='e',doc_id='d',section_id='s',source_path='test',heading_path=[],section_span=[0,5],evidence_span=[0,5],quote='abcde',supports=['f1'],raw_sources=[])])


def test_top_k_does_not_import_later_evidence():
    rows=[dict(section_id='s',ranges=[[0,2]])]+[dict(section_id=f'x{i}',ranges=[]) for i in range(9)]+[dict(section_id='s',ranges=[[2,5]])]
    result=retrieval_metrics(question(),script('run_retrieval_ablation').units_to_ranked(rows),10)
    assert result['evidence_hit']==result['required_fact_coverage']==0
    assert retrieval_metrics(question(),rows,11)['evidence_hit']==1


def test_duplicate_section_keeps_original_mrr_rank():
    rows=[dict(section_id='x',ranges=[]),dict(section_id='x',ranges=[]),dict(section_id='s',ranges=[[0,5]])]
    result=retrieval_metrics(question(),script('run_retrieval_ablation').units_to_ranked(rows),10)
    assert result['mrr']==pytest.approx(1/3)


def test_long_heading_cannot_bypass_budget():
    c=dict(chunk_id='c',doc_id='d',section_id='s',source_path='p',heading_path=['heading '*30],content='body',embedding_text='heading '*30+'body',section_start=0,section_end=4)
    with pytest.raises(ValueError,match='refine'):build_units([c],lambda t:len(t.split()),20)


def test_embedding_rejects_before_model_can_truncate():
    model=EmbeddingModel()
    model.__dict__['tokenizer']=lambda ts,**kw:{'input_ids':[list(range(513)) for _ in ts]}
    model._model=SimpleNamespace(encode=lambda *a,**kw:pytest.fail('Embedding was called'))
    with pytest.raises(ValueError,match='512'):model.encode(['oversized'])


def test_model_hashing_never_materializes_whole_weight_file(tmp_path,monkeypatch):
    import hashlib
    weights=tmp_path/'model.safetensors';weights.write_bytes(b'test weights')
    model=EmbeddingModel(str(tmp_path))
    def forbidden(*args):pytest.fail('Whole-file read can cause MemoryError with resident models')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    assert model.identity['files_sha256']['model.safetensors']==hashlib.sha256(b'test weights').hexdigest()


def units():
    return [RetrievalUnit(unit_id=f'u{i}',chunk_id=f'u{i}',doc_id='d',section_id='s',source_path='p',heading_path=[],text=f'facts {i}',section_start=i*10,section_end=i*10+7) for i in range(45)]


class FakeEmbedding:
    dim=4
    identity={'test_model':'deterministic','version':1}
    def encode(self,ts,**kw):
        import numpy as np
        return np.array([[1,0,0,0] for _ in ts],dtype='float32')


def test_rerank_pool_separate_from_topk_and_explicit_strategy(tmp_path):
    engine=RetrievalEngine(units(),FakeEmbedding(),tmp_path,40)
    engine._bm25=SimpleNamespace();engine._dense=SimpleNamespace()
    calls=[]
    def retrieve(q,k):
        calls.append(k);return [engine._result(f'u{i}',float(100-i),i+1) for i in range(k)]
    engine._hybrid_retrieve=retrieve;engine._bm25_retrieve=retrieve
    class Rerank:
        seen=0
        def rerank(self,q,texts):self.seen=len(texts);return [(i,float(i)) for i in reversed(range(len(texts)))]
    rerank=Rerank()
    result=engine.retrieve('test','hybrid_rerank',3,rerank,20)
    assert calls[-1]==rerank.seen==20 and [r['unit_id'] for r in result]==['u19','u18','u17']
    rerank.seen=0;engine.retrieve('test','bm25',3,rerank,20)
    assert rerank.seen==0
    with pytest.raises(ValueError,match='configured Reranker'):engine.retrieve('test','hybrid_rerank',3,candidate_k=20)
    with pytest.raises(ValueError,match='candidate_k'):engine.retrieve('test','hybrid',10,candidate_k=5)


def test_index_refuses_changed_units_model_and_overwrite(tmp_path):
    embedding=FakeEmbedding();original=units()[:3]
    engine=RetrievalEngine(original,embedding,tmp_path,3);engine.build();engine.close()
    same=RetrievalEngine(original,embedding,tmp_path,3);same.load();same.close()
    changed=[u.model_copy(update={'text':'tampered'}) for u in original]
    with pytest.raises(ValueError,match='version mismatch'):RetrievalEngine(changed,embedding,tmp_path).load()
    other=FakeEmbedding();other.identity={'test_model':'different'}
    with pytest.raises(ValueError,match='version mismatch'):RetrievalEngine(original,other,tmp_path).load()
    with pytest.raises(FileExistsError):RetrievalEngine(original,embedding,tmp_path).build()


def test_shared_loader_checks_units_and_frozen_source_hash(tmp_path,monkeypatch):
    import hashlib
    import hybrid_rag.retrieval.engine as module
    embedding=FakeEmbedding();original=units()[:3]
    index=tmp_path/'index';index.mkdir()
    raw=''.join(u.model_dump_json()+'\n' for u in original)
    unit_file=index/'units.jsonl';unit_file.write_text(raw,encoding='utf-8')
    source=tmp_path/'chunks.jsonl';source.write_text('frozen source',encoding='utf-8')
    engine=RetrievalEngine(original,embedding,index/'qdrant',3);engine.build();engine.close()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    meta=dict(format_version=2,model='fake',embedding_identity=embedding.identity,units_sha256=digest(unit_file),chunks_source=str(source),chunks_sha256=digest(source))
    (index/'metadata.json').write_text(json.dumps(meta),encoding='utf-8')
    monkeypatch.setattr(module,'EmbeddingModel',lambda name:FakeEmbedding())
    loaded,_=module.load_index(index,3);loaded.close()
    source.write_text('changed source',encoding='utf-8')
    with pytest.raises(ValueError,match='Frozen Chunk'):module.load_index(index)
    source.write_text('frozen source',encoding='utf-8')
    unit_file.write_text(raw+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='units file changed'):module.load_index(index)
    (index/'metadata.json').write_text(json.dumps({'model':'fake'}),encoding='utf-8')
    with pytest.raises(ValueError,match='Legacy/incomplete'):module.load_index(index)


def test_reranker_windows_are_explicit_and_pair_bounded():
    from transformers import AutoTokenizer
    from hybrid_rag.retrieval.embeddings import CrossEncoderReranker
    import numpy as np
    tok=AutoTokenizer.from_pretrained(ROOT/'data/models/bge-base-en-v1.5-tokenizer',local_files_only=True)
    seen=[]
    def predict(pairs,**kw):seen.extend(pairs);return np.arange(len(pairs),dtype='float32')
    r=CrossEncoderReranker();r._model=SimpleNamespace(tokenizer=tok,predict=predict)
    result=r.scores([('What is stored?',' '.join(f'parameter_{i}' for i in range(600)))])
    assert len(seen)>1 and r.last_window_counts==[len(seen)]
    assert result==[float(len(seen)-1)]
    assert all(len(tok(q,t,add_special_tokens=True,truncation=False)['input_ids'])<=512 for q,t in seen)


def test_api_strategy_and_missing_reranker(monkeypatch):
    from fastapi.testclient import TestClient
    module=script('serve_api');calls=[]
    class Engine:
        def retrieve(self,q,**kw):
            calls.append(kw)
            if kw['strategy']=='hybrid_rerank' and kw['reranker'] is None:raise ValueError('Reranker required')
            return []
    monkeypatch.setattr(module,'_load',lambda:Engine());monkeypatch.setattr(module,'_reranker',object())
    client=TestClient(module.app)
    for mode in ['bm25','dense','hybrid']:
        assert client.post('/retrieve',json=dict(query='test',strategy=mode,top_k=5,candidate_k=20)).status_code==200
        assert calls[-1]['strategy']==mode and calls[-1]['reranker'] is None and calls[-1]['candidate_k']==20
    monkeypatch.setattr(module,'_reranker',None)
    assert client.post('/retrieve',json=dict(query='test',strategy='hybrid_rerank')).status_code==400
    assert client.post('/retrieve',json=dict(query='test',top_k=0)).status_code==422


def test_cli_ask_without_model_configuration_is_dry_run(monkeypatch,tmp_path,capsys):
    import hybrid_rag.retrieval.engine as engines
    import hybrid_rag.retrieval.generation as generation
    module=script('query_rag')
    context=dict(text='Supported evidence.',unit_id='u',chunk_id='u',section_id='s',source_path='p',heading_path=[],section_start=0,section_end=19,score=1,rank=1,ranges=[[0,19]])
    engine=SimpleNamespace(retrieve=lambda *a,**kw:[context],close=lambda:None)
    monkeypatch.setattr(engines,'load_index',lambda *a:(engine,{}))
    def no_config():raise ValueError('No model configured')
    monkeypatch.setattr(generation,'make_chat_model',no_config)
    output=tmp_path/'trace.json'
    monkeypatch.setattr('sys.argv',['query_rag.py','test','--ask','--output',str(output)])
    module.main();out=json.loads(output.read_text(encoding='utf-8'))
    assert out['dry_run'] and out['answer'] is None and out['prompt_sha256']


def test_generation_quotes_and_refusal_are_grounded_without_api():
    class Mock:
        def __init__(self,obj):self.obj=obj
        def invoke(self,prompt,**kw):return SimpleNamespace(content=json.dumps(self.obj),usage_metadata={'input_tokens':1,'output_tokens':1})
    context=[dict(text='A checkpointer stores graph state.',source_path='p',unit_id='u',ranges=[[0,34]])]
    out=generate('What is saved?',context,Mock(dict(answer='It stores graph state [1].',abstained=False,citations=[dict(source=1,quote='stores graph state')])))
    assert out['llm_invoked'] and out['citations'][0]['context_id']=='ctx1'
    assert generate('Question',[])['abstained'] is True
    assert generate('Question',context)['dry_run'] is True
    with pytest.raises(ValueError,match='actual retrieved source'):
        generate('Question',context,Mock(dict(answer='Secret [1].',abstained=False,citations=[dict(source=1,quote='invented secret')])))
    with pytest.raises(ValueError,match='supporting citations'):
        generate('Question',context,Mock(dict(answer='Uncited answer',abstained=False,citations=[])))


def test_real_factory_missing_key_never_calls_provider(monkeypatch):
    from hybrid_rag.retrieval.generation import make_chat_model,answer_question
    for name in ['DEEPSEEK_API_KEY','HYBRIDRAG_DEEPSEEK_API_KEY','HYBRIDRAG_LLM_PROVIDER']:
        monkeypatch.delenv(name,raising=False)
    with pytest.raises(ValueError,match='not set'):make_chat_model(provider='deepseek')
    out=answer_question('Question',[dict(text='Evidence',source_path='p')])
    assert out['dry_run'] and out['answer'] is None and not out['llm_invoked']


def test_http_ask_preserves_context_snapshot_with_mock_model(monkeypatch):
    from fastapi.testclient import TestClient
    import hybrid_rag.retrieval.generation as generation
    module=script('serve_api')
    context=dict(text='The graph stores state.',unit_id='u',source_path='p',ranges=[[0,23]])
    monkeypatch.setattr(module,'_load',lambda:SimpleNamespace(retrieve=lambda *a,**kw:[context]))
    class Model:
        def invoke(self,*a,**kw):return SimpleNamespace(content=json.dumps(dict(answer='It stores state [1].',abstained=False,citations=[dict(source=1,quote='stores state')])))
    monkeypatch.setattr(generation,'make_chat_model',lambda:Model())
    response=TestClient(module.app).post('/ask',json=dict(query='What is stored?',strategy='hybrid'))
    assert response.status_code==200
    out=response.json()
    assert not out['dry_run'] and out['contexts'][0]['context_id']=='ctx1' and out['context_snapshot_hash']


def test_actual_hard512_chunks_are_one_to_one():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'data/models/bge-base-en-v1.5-tokenizer',local_files_only=True)
    rows=[json.loads(s) for s in (ROOT/'data/processed/hard512-improved-v2/chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    mapped=build_units(rows,lambda t:len(tok.encode(t,add_special_tokens=True,truncation=False)))
    assert len(mapped)==len(rows)
    for c,u in zip(rows,mapped):
        expected=[[p['section_start'],p['section_end']] for p in c.get('content_parts',[])] or [[c['section_start'],c['section_end']]]
        assert u.ranges==expected and u.unit_id==c['chunk_id'] and u.text==c['embedding_text']
