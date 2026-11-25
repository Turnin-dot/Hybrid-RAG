import pytest
from hybrid_rag.evaluation.generation import GenerationJudgment,score_generation
from hybrid_rag.evaluation.judge import validate_extraction,validate_assessment,cohort_statistics,calibration_report,evaluate_answer
from hybrid_rag.evaluation.schema import EvalQuestion
from hybrid_rag.retrieval.context_selection import select_contexts


def question(answerable=True):
    return EvalQuestion(query_id='synthetic',dataset_version='test',topic_group='family',question='How?',category='concept' if answerable else 'unanswerable',products=[],split='dev',answerable=answerable,reference_answer='A and B',reference_facts=[dict(fact_id='a',text='A'),dict(fact_id='b',text='B')] if answerable else [],gold_evidence=[dict(evidence_id='e',section_id='s',doc_id='d',source_path='p',heading_path=[],section_span=[0,1],evidence_span=[0,1],quote='A',supports=['a','b'],raw_sources=[])] if answerable else [],section_relevance={'s':3} if answerable else {},unanswerable_reason=None if answerable else 'Private information')


def judgment(**updates):
    obj=dict(query_id='synthetic',run_id='1',answer='A. Evidence is insufficient for B.',refused=False,abstention_appropriate=False,directly_answers_question=False,has_central_error=False,claims=[dict(claim_id='c',text='A',answer_quote='A',correct=True,supported_by_context=True,evidence_refs=['ctx1'],matches_fact_ids=['a'])],citations=[dict(claim_id='c',context_id='ctx1',supports_claim=True)],context_ids=['ctx1'],context_snapshot_hash='a'*64,reviewer='mock',reason='partial',rubric_version='grounded-v2',fact_assessments=[dict(fact_id='a',status='full',answer_quotes=['A'],reason='stated'),dict(fact_id='b',status='missing',reason='not stated')])
    obj.update(updates);return GenerationJudgment.model_validate(obj)


def test_answer_only_quote_and_locked_claims_reject_gold_hallucination():
    with pytest.raises(ValueError):validate_extraction('A.',dict(claims=[dict(claim_id='c',text='B',answer_quote='B')]))
    extraction=validate_extraction('A.',dict(claims=[dict(claim_id='c',text='A',answer_quote='A')]))
    obj=judgment().model_dump();obj['claims'].append(dict(claim_id='gold',text='B',answer_quote='B'))
    with pytest.raises(ValueError,match='locked'):validate_assessment(question(),dict(answer='A.'),extraction,obj)


def test_partial_answer_is_not_zero_for_incompleteness_or_local_refusal():
    for refused in [True,False]:
        score=score_generation(question(),judgment(refused=refused),True)
        assert score['correctness_bucket']==.5 and score['fact_recall']==.5 and score['false_refusal']==0


def test_true_core_error_stays_zero_and_unsupported_is_separate():
    j=judgment();j.claims[0].supported_by_context=False;j.claims[0].evidence_refs=[]
    score=score_generation(question(),j,True)
    assert score['correctness_bucket']==.5 and score['faithfulness']==0
    data=judgment().model_dump();data['claims'][0]['correct']=False;data.update(has_central_error=True,core_error_claim_ids=['c'])
    assert score_generation(question(),GenerationJudgment.model_validate(data),True)['correctness_bucket']==0
    with pytest.raises(ValueError,match='flag'):judgment(has_central_error=True)


def test_missing_gold_does_not_enter_claim_precision_or_faithfulness():
    score=score_generation(question(),judgment(),True)
    assert score['fact_precision']==score['faithfulness']==1 and score['completeness']==.5


def test_compound_partial_fact_and_no_answer_empty_output():
    j=judgment();j.fact_assessments[0].status='partial'
    score=score_generation(question(),j,True)
    assert score['fact_recall']==0 and score['completeness']==.25 and score['correctness_bucket']==.5
    j=judgment(answer='Cannot provide a private credential.',refused=True,claims=[],citations=[],fact_assessments=[],abstention_appropriate=True)
    assert score_generation(question(False),j,True)['correctness_bucket']==1
    j=judgment(answer='Cannot provide a private credential.',refused=True,claims=[dict(claim_id='c',text='Cannot provide a private credential.',answer_quote='Cannot provide a private credential.',correct=True,supported_by_context=False,evidence_refs=[],matches_fact_ids=[])],citations=[],fact_assessments=[],abstention_appropriate=True)
    separate=score_generation(question(False),j,True)
    assert separate['correct_abstention']==1 and separate['faithfulness']==0 and separate['answer_success']==0
    j.answer='';assert score_generation(question(False),j,True)['correctness_bucket']==0


def test_cohorts_and_confusion_counts_keep_denominators_separate():
    qs=[question(),question(False)];qs[1].query_id='private'
    rows=[dict(query_id='synthetic',correctness_bucket=.5),dict(query_id='private',correctness_bucket=1,correct_abstention=1)]
    result=cohort_statistics(qs,rows,[dict(query_id='synthetic',abstained=False),dict(query_id='private',abstained=True)])
    assert result['all']['n']==2 and result['all']['mean_score']==.75
    assert result['answerable']['full_score_rate']==0 and result['unanswerable']['full_score_rate']==1
    assert calibration_report({'synthetic':.5,'private':1},rows)['confusion_matrix']['counts']==[[0,0,0],[0,1,0],[0,0,1]]


def test_dynamic_has_no_gold_dependency_respects_order_budget_and_diversity():
    candidates=[dict(chunk_id=str(i),section_id=str(i//2),rank=i+1,text=f'unique_{i} '+('text '*i)) for i in range(12)]
    count=lambda t:len(t.split())
    result=select_contexts('How?',candidates,'dynamic',count,max_contexts=4,prompt_budget=5000)
    assert [r['rank'] for r in result]==[1,3,5,7]
    assert select_contexts('How?',candidates,'top5',count)==candidates[:5]
    assert select_contexts('How?',candidates,'dynamic',count,prompt_budget=1)==[]


def test_real_orchestrator_extraction_never_receives_gold_and_binds_anchors():
    import json
    from types import SimpleNamespace
    q=question();q.reference_answer='GOLD_SENSITIVE'
    saved=dict(answer='A. Evidence is insufficient for B.',run_id='1',abstained=False,citations=[],contexts=[dict(context_id='ctx1',text='A')],context_snapshot_hash='a'*64)
    calls=[]
    class Mock:
        def invoke(self,messages):
            calls.append(messages);payload=json.loads(messages[1]['content'])
            if len(calls)==1:
                assert 'GOLD_SENSITIVE' not in str(payload) and set(payload)=={'question','actual_answer'}
                obj=dict(claims=[dict(claim_id='c',text='A',answer_quote='A')])
            else:
                assert payload['reference_answer']=='GOLD_SENSITIVE'
                obj={k:v for k,v in judgment().model_dump().items() if k in ['directly_answers_question','has_central_error','core_error_claim_ids','abstention_appropriate','reason','claims','citations','fact_assessments']}
                del obj['claims'][0]['text'];del obj['claims'][0]['answer_quote']
            return SimpleNamespace(content=json.dumps(obj),usage_metadata={},response_metadata={})
    traces=[];result=evaluate_answer(q,saved,Mock(),traces)
    assert len(calls)==2 and result.claims[0].answer_quote=='A' and all(t['valid'] for t in traces)
