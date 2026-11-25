from pathlib import Path
import json
import pytest
from pydantic import ValidationError
from hybrid_rag.evaluation.schema import EvalQuestion
from hybrid_rag.evaluation.dataset import load_questions,validate_dataset
from hybrid_rag.evaluation.retrieval import retrieval_metrics
from hybrid_rag.evaluation.generation import GenerationJudgment,score_generation
from hybrid_rag.evaluation.statistics import bootstrap,aggregate_runs,paired_bootstrap,mcnemar,calibration,cluster_bootstrap

ROOT=Path(__file__).parents[2]
DATASET=ROOT/'data/eval/hybridrag-en-v1'


def question(answerable=True):
    evidence=[dict(evidence_id=f'e{i}',section_id=s,doc_id='d',source_path='example.mdx',heading_path=['Example'],
                   section_span=span,evidence_span=span,quote='x'*(span[1]-span[0]),supports=[fid],raw_sources=[]) for i,(s,span,fid) in enumerate([('s1',[0,5],'f1'),('s2',[2,6],'f2'),('s3',[0,5],'f1')],1)]
    return EvalQuestion(query_id='unit-query',dataset_version='unit',topic_group='unit-topic',question='What is the behavior?',category='concept' if answerable else 'unanswerable',
        products=['unit'],split='dev',answerable=answerable,reference_answer='Unit reference',
        reference_facts=[dict(fact_id='f1',text='First fact'),dict(fact_id='f2',text='Second fact')] if answerable else [],
        gold_evidence=evidence if answerable else [],unanswerable_reason=None if answerable else 'outside_snapshot',
        section_relevance={'s1':2,'s2':2,'s3':2} if answerable else {},annotation_status='approved',reviewed_by='unit-fixture',reviewed_at='2026-11-15T00:00:00Z')


def judgment(claims=None,**updates):
    claims=claims if claims is not None else [dict(claim_id=f'c{i}',text='Unit claim',correct=True,supported_by_context=True,matches_fact_ids=[f'f{i}'],evidence_refs=['ctx']) for i in (1,2)]
    data=dict(query_id='unit-query',run_id='unit-run',answer='Unit answer',refused=False,directly_answers_question=True,
              claims=claims,citations=[dict(claim_id=c['claim_id'],context_id='ctx',supports_claim=True) for c in claims],
              context_ids=['ctx'],context_snapshot_hash='a'*64,reviewer='unit-fixture',judgment_status='approved',reason='Synthetic metric unit test, not a retrieval experiment.')
    data.update(updates)
    return GenerationJudgment.model_validate(data)


def test_actual_dataset_all_positions_and_splits():
    r=validate_dataset(DATASET,ROOT)
    assert r['errors']==[] and r['questions']==120 and r['approved']==0
    assert r['evidence_spans']==146
    qs=load_questions(DATASET,allow_unreviewed=True)
    assert sum(q.split=='dev' for q in qs)==40
    assert sum(q.category=='unanswerable' for q in qs)==10
    assert len({q.topic_group for q in qs})==58


def test_v2_categories_products_and_unchanged_evidence():
    from collections import Counter
    latest=ROOT/'data/eval/hybridrag-en-v2'
    assert validate_dataset(latest,ROOT)['errors']==[]
    old={q.query_id:q for q in load_questions(DATASET,allow_unreviewed=True)}
    new=load_questions(latest,allow_unreviewed=True)
    counts=Counter(q.category for q in new)
    assert counts['api']==24 and counts['concept']==26
    assert {q.category for q in new if q.query_id in {'q049','q050'}}=={'concept'}
    for q in new:
        previous=old[q.query_id]
        assert (q.question,q.reference_answer,q.gold_evidence,q.split)==(previous.question,previous.reference_answer,previous.gold_evidence,previous.split)
        assert q.annotation_status=='pending_review'
        if not q.answerable:
            assert q.products==[] and q.absence_check['product_scope']=='not_product_specific'


def test_rejected_reviews_require_explicit_checks_and_notes(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('review_reject',ROOT/'scripts/review_eval_dataset.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    row=dict(query_id='q001',decision='rejected',reviewer='unit-fixture',notes='Evidence requires further review.')
    path=tmp_path/'decisions.jsonl'
    path.write_text(json.dumps(row)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Five explicit boolean'):
        module.apply_reviews(DATASET,path,tmp_path/'invalid','unit-rejected')
    row['checks']=dict(clarity=True,atomic_facts=True,evidence_sufficiency=False,answerability=True,split_integrity=True)
    path.write_text(json.dumps(row)+'\n',encoding='utf-8')
    module.apply_reviews(DATASET,path,tmp_path/'valid','unit-rejected')
    q=load_questions(tmp_path/'valid',allow_unreviewed=True)[0]
    assert q.annotation_status=='rejected'
    row['notes']=''
    path.write_text(json.dumps(row)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='explanatory notes'):
        module.apply_reviews(DATASET,path,tmp_path/'missing-notes','unit-rejected')


def test_official_scoring_rejects_unreviewed_labels():
    with pytest.raises(ValueError,match='pending human review'):load_questions(DATASET)
    q=question().model_copy(update={'annotation_status':'pending_review'})
    with pytest.raises(ValueError,match='Human-approved'):score_generation(q,judgment())


def test_section_hit_does_not_imply_fact_coverage():
    r=retrieval_metrics(question(),[dict(section_id='s1',ranges=[[0,2]])],5)
    assert r['hit_section']==1 and r['recall_section']==pytest.approx(1/3)
    assert r['required_fact_coverage']==0 and r['evidence_hit']==0


def test_equivalent_and_joint_evidence_and_chunk_union():
    r=retrieval_metrics(question(),[dict(section_id='s3',ranges=[[0,5]]),dict(section_id='s2',ranges=[[2,4]]),dict(section_id='s2',ranges=[[4,6]])],3)
    assert r['required_fact_coverage']==r['all_required_facts']==1
    assert r['recall_section']==pytest.approx(2/3)
    assert r['ndcg']<1 # duplicate Section consumes a rank; it has zero additional gain


def test_mrr_original_rank_not_collapsed_rank():
    ranked=[dict(section_id='x',ranges=[]),dict(section_id='x',ranges=[]),dict(section_id='s1',ranges=[[0,5]])]
    assert retrieval_metrics(question(),ranked,3)['mrr']==pytest.approx(1/3)
    assert retrieval_metrics(question(),ranked,2)['hit_section']==0
    assert retrieval_metrics(question(False),ranked,5) is None


def test_generation_correctness_faithfulness_citations():
    r=score_generation(question(),judgment())
    assert r['fact_precision']==r['fact_recall']==r['faithfulness']==r['citation_accuracy']==r['answer_success']==1
    j=judgment();j.claims.append(type(j.claims[0])(claim_id='wrong',text='Unsupported incorrect assertion',correct=False,supported_by_context=False))
    r=score_generation(question(),j)
    assert r['fact_precision']==r['faithfulness']==pytest.approx(2/3)
    assert r['fact_recall']==1 and r['fact_f1']==pytest.approx(.8) and r['answer_success']==0
    r=score_generation(question(),judgment(directly_answers_question=False))
    assert r['correctness_bucket']==r['answer_relevance']==r['answer_success']==0
    assert score_generation(question(),judgment(has_central_error=True))['correctness_bucket']==0


def test_empty_output_and_refusal_not_faithful_by_default():
    j=judgment(claims=[],citations=[],answer='',refused=True,abstention_appropriate=True)
    r=score_generation(question(False),j)
    assert r['faithfulness'] is None and r['fact_precision'] is None and r['answer_success']==0
    assert r['unsupported_answer']==0
    j.answer='This information is not available in the frozen corpus.'
    assert score_generation(question(False),j)['correct_abstention']==1
    assert score_generation(question(),j)['false_refusal']==1


def test_supported_explanation_does_not_make_appropriate_refusal_incorrect():
    j=judgment(claims=[dict(claim_id='c1',text='The docs only show an example password.',correct=True,
        supported_by_context=True,matches_fact_ids=[],evidence_refs=['ctx'])],
        answer='The documentation example is not your private password.',refused=True,abstention_appropriate=True)
    score=score_generation(question(False),j)
    assert score['correct_abstention']==1 and score['unsupported_answer']==0
    j.claims[0].supported_by_context=False
    assert score_generation(question(False),j)['correct_abstention']==0


def test_correctness_and_faithfulness_are_independent():
    j=judgment(claims=[dict(claim_id='c1',text='Wrong but present in context',correct=False,supported_by_context=True,matches_fact_ids=[],evidence_refs=['ctx'])])
    r=score_generation(question(),j)
    assert r['faithfulness']==1 and r['fact_precision']==0 and r['fact_recall']==0


def test_invalid_references_and_unknown_facts_rejected():
    j=judgment().model_dump();j['citations'][0]['context_id']='not-retrieved'
    with pytest.raises(ValidationError):GenerationJudgment.model_validate(j)
    j=judgment();j.claims[0].matches_fact_ids=['invented']
    with pytest.raises(ValueError,match='Unknown reference fact'):score_generation(question(),j)


def test_question_level_repeats_and_fixed_context():
    rows=[dict(query_id='q1',run_id='a',context_snapshot_hash='x',score=0),dict(query_id='q1',run_id='b',context_snapshot_hash='x',score=1),dict(query_id='q2',run_id='a',context_snapshot_hash='y',score=1)]
    result=aggregate_runs(rows,'score')
    assert result['bootstrap']['mean']==.75
    assert result['mean_within_question_std']==.5
    rows[1]['context_snapshot_hash']='changed'
    with pytest.raises(ValueError,match='confound'):aggregate_runs(rows,'score')


def test_bootstrap_pairing_and_calibration_known_answers():
    assert bootstrap([1,1,1])['ci95']==[1,1]
    assert bootstrap([0,1])==bootstrap([0,1])
    assert paired_bootstrap({'a':0,'b':0},{'a':1,'b':1})['ci95']==[1,1]
    assert mcnemar({'a':0,'b':0},{'a':1,'b':1})['exact_two_sided_p']==.5
    scores={'a':0,'b':.5,'c':1};r=calibration(scores,scores)
    assert r['kappa']==r['quadratic_weighted_kappa']==r['spearman']==1
    assert calibration({'a':1},{'a':1})['kappa'] is None
    with pytest.raises(ValueError):paired_bootstrap({'a':1},{'b':1})


def test_cluster_bootstrap_keeps_topic_variants_together():
    rows=[dict(query_id='a',topic_group='one',value=0),dict(query_id='b',topic_group='one',value=0),dict(query_id='c',topic_group='two',value=1)]
    r=cluster_bootstrap(rows,'value')
    assert r['questions']==3 and r['topic_groups']==2 and r['mean']==pytest.approx(1/3)


def test_human_review_import_is_explicit_and_versioned(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('review_import',ROOT/'scripts/review_eval_dataset.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    decisions=[json.loads(s) for s in (DATASET/'review_decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    path=tmp_path/'decisions.jsonl';decisions[0]['decision']='approved'
    path.write_text(''.join(json.dumps(r)+'\n' for r in decisions),encoding='utf-8')
    with pytest.raises(ValueError,match='reviewer'):m.apply_reviews(DATASET,path,tmp_path/'bad','unit-review')
    decisions[0].update(reviewer='unit-fixture',checks={k:True for k in decisions[0]['checks']})
    path.write_text(''.join(json.dumps(r)+'\n' for r in decisions),encoding='utf-8')
    m.apply_reviews(DATASET,path,tmp_path/'good','unit-review')
    qs=load_questions(tmp_path/'good',allow_unreviewed=True)
    assert sum(q.annotation_status=='approved' for q in qs)==1
    reviewed=json.loads((tmp_path/'good'/'validation_report.json').read_text(encoding='utf-8'))
    assert reviewed['manual_review_status']=='partial_human_review'
    assert sum(q.annotation_status=='approved' for q in load_questions(DATASET,allow_unreviewed=True))==0


@pytest.mark.parametrize('mode',['retrieval','generation'])
def test_cli_on_explicitly_synthetic_single_query(tmp_path,monkeypatch,mode):
    """Exercise the saved-results interface, never score the real pending set."""
    import importlib.util,hashlib
    spec=importlib.util.spec_from_file_location('eval_cli',ROOT/'scripts/evaluate_rag.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    q=load_questions(DATASET,allow_unreviewed=True)[0].model_copy(update={'annotation_status':'approved','reviewed_by':'unit-fixture','reviewed_at':'2026-11-15T00:00:00Z'})
    dataset=tmp_path/'unit-dataset';dataset.mkdir()
    (dataset/'questions.jsonl').write_text(q.model_dump_json()+'\n',encoding='utf-8')
    meta=json.loads((DATASET/'metadata.json').read_text(encoding='utf-8'));meta.pop('artifacts_sha256',None)
    (dataset/'metadata.json').write_text(json.dumps(meta),encoding='utf-8')
    manifest={k:None for k in ['chunking_config_hash','embedding_model_version','retriever_config','reranker_config','generator_model_version','generator_prompt_hash','temperature','seed','judge_model_version','judge_prompt_hash']}
    manifest.update({k:meta[k] for k in ['dataset_version','repository_commit','ingestion_config_hash']})
    manifest.update(index_version='unit-fixture',metric_code_version='unit-fixture',dataset_sha256=hashlib.sha256((dataset/'questions.jsonl').read_bytes()).hexdigest(),chunks_sha256=hashlib.sha256((ROOT/'data/processed/phase1-v3-final/chunks.jsonl').read_bytes()).hexdigest())
    manifest.update(retriever_config={'method':'unit-fixture'},generator_model_version='unit-fixture',generator_prompt_hash='b'*64)
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    results_path=tmp_path/'results.jsonl';contexts_path=tmp_path/'contexts.jsonl';output=tmp_path/'output.json'
    if mode=='retrieval':
        record=dict(query_id=q.query_id,results=[dict(section_id=q.gold_evidence[0].section_id,scope='section')])
    else:
        contexts=[dict(context_id='unit-context',content=q.gold_evidence[0].quote)]
        contexts_path.write_text(json.dumps(dict(query_id=q.query_id,run_id='unit-run',contexts=contexts))+'\n',encoding='utf-8')
        record=dict(query_id=q.query_id,run_id='unit-run',answer=q.reference_answer,refused=False,directly_answers_question=True,
                    claims=[dict(claim_id='c1',text=q.reference_facts[0].text,correct=True,supported_by_context=True,matches_fact_ids=['f1'],evidence_refs=['unit-context'])],
                    citations=[dict(claim_id='c1',context_id='unit-context',supports_claim=True)],context_ids=['unit-context'],context_snapshot_hash=cli.canonical(contexts),reviewer='unit-fixture',judgment_status='approved',reason='Synthetic interface test only')
    results_path.write_text(json.dumps(record)+'\n',encoding='utf-8')
    monkeypatch.setattr('sys.argv',['evaluate_rag.py','--dataset',str(dataset),'--results',str(results_path),'--mode',mode,'--manifest',str(manifest_path),'--output',str(output),*(['--contexts',str(contexts_path)] if mode=='generation' else [])])
    cli.main()
    report=json.loads(output.read_text(encoding='utf-8'))
    assert report['status']=='scored_reviewed_labels'
    assert report['input_records']==[record]
    assert report['results'][0]['required_fact_coverage' if mode=='retrieval' else 'answer_success']==1
    if mode=='generation':assert report['summary']['all']['correctness_bucket']['bootstrap']['mean']==1
