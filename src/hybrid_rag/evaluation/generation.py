"""Score structured human/Judge annotations, never infer truth by string match."""
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class Claim(BaseModel):
    claim_id: str
    text: str
    correct: bool
    supported_by_context: bool
    matches_fact_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    answer_quote: str | None = None


class FactAssessment(BaseModel):
    fact_id: str
    status: Literal['full','partial','missing','incorrect']
    answer_quotes: list[str] = Field(default_factory=list)
    reason: str


class Citation(BaseModel):
    claim_id: str
    context_id: str
    supports_claim: bool


class GenerationJudgment(BaseModel):
    query_id: str
    run_id: str
    answer: str
    refused: bool
    abstention_appropriate: bool = False
    directly_answers_question: bool
    has_central_error: bool = False
    claims: list[Claim]
    citations: list[Citation]
    context_ids: list[str]
    context_snapshot_hash: str = Field(pattern=r'^[a-f0-9]{64}$')
    reviewer: str
    judgment_status: Literal['pending_review','approved'] = 'pending_review'
    reason: str
    latency_ms: float | None = Field(default=None,ge=0)
    cost_usd: float | None = Field(default=None,ge=0)
    input_tokens: int | None = Field(default=None,ge=0)
    output_tokens: int | None = Field(default=None,ge=0)
    rubric_version: Literal['legacy','grounded-v2'] = 'legacy'
    fact_assessments: list[FactAssessment] = Field(default_factory=list)
    core_error_claim_ids: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def references(self):
        ids={c.claim_id for c in self.claims}
        if len(ids)!=len(self.claims):raise ValueError('Duplicate claim ID')
        if not self.reviewer.strip():raise ValueError('Reviewer identity required')
        for claim in self.claims:
            if not set(claim.evidence_refs)<=set(self.context_ids):raise ValueError('Evidence is not in actual retrieved context')
            if claim.supported_by_context and not claim.evidence_refs:raise ValueError('Supported claims require actual evidence references')
        for citation in self.citations:
            if citation.claim_id not in ids or citation.context_id not in self.context_ids:raise ValueError('Invalid citation reference')
        if self.rubric_version=='grounded-v2':
            for claim in self.claims:
                if not claim.answer_quote or claim.answer_quote not in self.answer:raise ValueError('Claim requires an exact actual-answer quote')
            for fact in self.fact_assessments:
                if any(not quote or quote not in self.answer for quote in fact.answer_quotes):raise ValueError('Fact attribution is not in the answer')
                if fact.status in ('full','partial') and not fact.answer_quotes:raise ValueError('Expressed facts require answer quotes')
            wrong_ids={c.claim_id for c in self.claims if not c.correct}
            if not set(self.core_error_claim_ids)<=wrong_ids:raise ValueError('Core error must identify an actual incorrect answer claim')
            if self.has_central_error!=bool(self.core_error_claim_ids):raise ValueError('Core error flag and claim IDs disagree')
        return self


def score_generation(question, judgment, allow_unreviewed=False):
    if not allow_unreviewed and (question.annotation_status!='approved' or judgment.judgment_status!='approved'):
        raise ValueError('Human-approved dataset and judgments required for official scoring')
    required={f.fact_id for f in question.reference_facts if f.required}
    known={f.fact_id for f in question.reference_facts}
    if any(not set(c.matches_fact_ids)<=known for c in judgment.claims):raise ValueError('Unknown reference fact')
    matched={fid for c in judgment.claims if c.correct for fid in c.matches_fact_ids}
    claims=len(judgment.claims)
    precision=sum(c.correct for c in judgment.claims)/claims if claims else None
    recall=len(required & matched)/len(required) if required else None
    faithful=sum(c.supported_by_context for c in judgment.claims)/claims if claims else None
    f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0 if question.answerable else None
    citation_accuracy=sum(c.supports_claim for c in judgment.citations)/len(judgment.citations) if judgment.citations else None
    wrong=sum(not c.correct for c in judgment.claims)
    completeness=recall
    if judgment.rubric_version=='grounded-v2':
        assessments={f.fact_id:f for f in judgment.fact_assessments}
        if len(assessments)!=len(judgment.fact_assessments) or set(assessments)!=known:raise ValueError('Assess every reference fact exactly once')
        full=sum(assessments[f].status=='full' for f in required)
        partial=sum(assessments[f].status=='partial' for f in required)
        recall=full/len(required) if required else None
        completeness=(full+.5*partial)/len(required) if required else None
        f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else 0 if question.answerable else None
    if question.answerable:
        bucket=(1 if recall==1 and wrong==0 else .5 if recall and recall>0 else 0) if judgment.directly_answers_question and not judgment.refused and not judgment.has_central_error else 0
        cited_claims={c.claim_id for c in judgment.citations if c.supports_claim}
        success=bool(not judgment.refused and not judgment.has_central_error and recall==1 and wrong==0 and faithful==1 and judgment.directly_answers_question
                     and citation_accuracy==1 and all(c.claim_id in cited_claims for c in judgment.claims))
        if judgment.rubric_version=='grounded-v2':
            # Missing facts/local evidence insufficiency are not false claims.
            bucket=0 if judgment.has_central_error or not judgment.answer.strip() else 1 if recall==1 and wrong==0 and not judgment.refused else .5 if completeness and completeness>0 else 0
            success=bool(bucket==1 and faithful==1 and citation_accuracy==1 and all(c.claim_id in cited_claims for c in judgment.claims))
    else:
        # A refusal may correctly explain a documentation placeholder. Judge
        # its appropriateness and support, rather than forbidding all claims.
        success=bool(judgment.answer.strip() and judgment.refused and judgment.abstention_appropriate
                     and not judgment.has_central_error and wrong==0 and all(c.supported_by_context for c in judgment.claims))
        bucket=float(success)
        if judgment.rubric_version=='grounded-v2':
            bucket=float(bool(judgment.answer.strip() and judgment.refused and judgment.abstention_appropriate and not judgment.has_central_error and wrong==0))
    return dict(fact_precision=precision,fact_recall=recall,fact_f1=f1,completeness=completeness,faithfulness=faithful,
                citation_accuracy=citation_accuracy,correctness_bucket=bucket,answer_relevance=float(judgment.directly_answers_question),answer_success=float(success),
                correct_abstention=float(bucket if judgment.rubric_version=='grounded-v2' else success) if not question.answerable else None,
                false_refusal=float(judgment.refused and not (judgment.rubric_version=='grounded-v2' and completeness and completeness>0)) if question.answerable else None,
                unsupported_answer=float(bool(not judgment.refused and judgment.answer.strip()) or any(not c.correct or not c.supported_by_context for c in judgment.claims)) if not question.answerable else None)
