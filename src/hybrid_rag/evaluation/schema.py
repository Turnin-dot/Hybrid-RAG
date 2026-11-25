from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class Fact(BaseModel):
    fact_id: str
    text: str
    required: bool = True


class Evidence(BaseModel):
    evidence_id: str
    section_id: str
    doc_id: str
    source_path: str
    heading_path: list[str]
    section_span: tuple[int, int] = Field(description='Exact quote range relative to Section.content, half-open character offsets; not the entire Section range')
    evidence_span: tuple[int, int] = Field(description='Exact quote range relative to cleaned Document.content, half-open character offsets')
    span_basis: Literal['cleaned_document_content'] = 'cleaned_document_content'
    quote: str
    supports: list[str]
    raw_sources: list[dict]

    @model_validator(mode='after')
    def spans(self):
        a,b=self.section_span;x,y=self.evidence_span
        if not 0<=a<b or not 0<=x<y or b-a!=len(self.quote) or y-x!=len(self.quote):
            raise ValueError('Evidence ranges must match the exact quote length')
        return self


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query_id: str
    dataset_version: str
    topic_group: str
    question: str
    category: Literal['api','concept','semantic_rewrite','multi_evidence','english_terminology','unanswerable']
    products: list[str]
    split: Literal['dev','test']
    answerable: bool
    reference_answer: str
    reference_facts: list[Fact]
    gold_evidence: list[Evidence]
    unanswerable_reason: str | None = None
    absence_check: dict | None = None
    annotation_status: Literal['pending_review','approved','rejected'] = 'pending_review'
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    section_relevance: dict[str, int] = Field(default_factory=dict)
    review_locations: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def grounded(self):
        if self.annotation_status=='approved' and (not self.reviewed_by or not self.reviewed_at):
            raise ValueError('Approved labels require human review attribution')
        if self.answerable:
            required={f.fact_id for f in self.reference_facts if f.required}
            if not required or not self.gold_evidence or not required <= {f for e in self.gold_evidence for f in e.supports}:
                raise ValueError('Every required fact must have evidence')
            if any(not 0<=g<=3 for g in self.section_relevance.values()) or not {e.section_id for e in self.gold_evidence}<={sid for sid,g in self.section_relevance.items() if g>0}:
                raise ValueError('Gold Sections require explicit relevance grades 1..3')
        elif self.reference_facts or self.gold_evidence or not self.unanswerable_reason:
            raise ValueError('Unanswerable cases need a reason, not invented evidence')
        if any(f not in {x.fact_id for x in self.reference_facts} for e in self.gold_evidence for f in e.supports):
            raise ValueError('Evidence references an unknown fact')
        return self
