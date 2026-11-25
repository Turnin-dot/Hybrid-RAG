import hashlib
import json
from pathlib import Path
import re
from .schema import EvalQuestion


def load_questions(directory,split=None,allow_unreviewed=False):
    directory=Path(directory)
    meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    for name,digest in meta.get('artifacts_sha256',{}).items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:raise ValueError('Dataset artifact changed: '+name)
    questions=[EvalQuestion.model_validate_json(s) for s in (directory/('pilot30.jsonl' if split=='pilot30' else 'questions.jsonl')).read_text(encoding='utf-8').splitlines()]
    if any(q.dataset_version!=meta['dataset_version'] for q in questions):raise ValueError('Mixed dataset versions')
    selected=questions if split=='pilot30' else [q for q in questions if split is None or q.split==split]
    if not selected:raise ValueError('Empty evaluation subset')
    if not allow_unreviewed and any(q.annotation_status!='approved' for q in selected):
        raise ValueError('Candidate labels are pending human review; official evaluation is blocked')
    return selected


def validate_dataset(directory,root):
    directory,root=Path(directory),Path(root)
    questions=load_questions(directory,allow_unreviewed=True)
    meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    sections={s['section_id']:s for s in map(json.loads,(root/'data/processed/phase1-v3-final/sections.jsonl').read_text(encoding='utf-8').splitlines())}
    docs={d['doc_id']:d for d in map(json.loads,(root/'data/processed/phase1-v3-final/documents.jsonl').read_text(encoding='utf-8').splitlines())}
    errors=[];seen={};section_splits={};groups={};locations=(directory/'review_all.md').read_text(encoding='utf-8').splitlines()
    for path,digest in meta['corpus_hashes'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:errors.append('Changed frozen corpus: '+path)
    for index,q in enumerate(questions,1):
        if q.query_id in seen:errors.append('Duplicate query ID')
        seen[q.query_id]=q
        if q.topic_group in groups and groups[q.topic_group]!=q.split:errors.append('Topic leakage: '+q.topic_group)
        groups[q.topic_group]=q.split
        if re.search(r'[\u4e00-\u9fff]',q.question+q.reference_answer):errors.append('Non-English Q/A: '+q.query_id)
        loc=q.review_locations
        if loc['questions_jsonl_line']!=index:errors.append('Wrong JSONL position')
        if locations[loc['question_line']-1]!=q.question or locations[loc['answer_line']-1]!=q.reference_answer:errors.append('Wrong review Markdown position: '+q.query_id)
        if 'pilot_markdown' in loc:
            pilot_lines=(directory/loc['pilot_markdown']).read_text(encoding='utf-8').splitlines()
            if pilot_lines[loc['pilot_question_line']-1]!=q.question or pilot_lines[loc['pilot_answer_line']-1]!=q.reference_answer:errors.append('Wrong pilot position: '+q.query_id)
        for e in q.gold_evidence:
            s,d=sections[e.section_id],docs[e.doc_id]
            a,b=e.section_span;x,y=e.evidence_span
            if not 0<=a<b<=len(s['content']) or s['content'][a:b]!=e.quote:errors.append('Section evidence mismatch: '+q.query_id)
            if s['doc_id']!=e.doc_id or (s['content_start']+a,s['content_start']+b)!=(x,y) or d['content'][x:y]!=e.quote:errors.append('Document evidence mismatch: '+q.query_id)
            if e.section_id in section_splits and section_splits[e.section_id]!=q.split:errors.append('Evidence leakage: '+e.section_id)
            section_splits[e.section_id]=q.split
            if not e.raw_sources:errors.append('Missing original source position')
            for raw in e.raw_sources:
                p=(directory/'source_snapshot'/raw['source_path']).resolve()
                if not p.is_relative_to((directory/'source_snapshot').resolve()):raise ValueError('Unsafe evidence source')
                data=p.read_bytes()
                if hashlib.sha256(data).hexdigest()!=raw['sha256']:errors.append('Source hash mismatch')
                if not 1<=raw['line_start']<=raw['line_end']<=len(data.decode('utf-8-sig').splitlines()):errors.append('Invalid raw line range')
    for filename,split in [('dev.jsonl','dev'),('test.jsonl','test')]:
        ids=[q.query_id for q in load_jsonl(directory/filename)]
        if ids!=[q.query_id for q in questions if q.split==split]:errors.append('Subset mismatch: '+filename)
    pilot=load_jsonl(directory/'pilot30.jsonl')
    if len(pilot)!=30 or any(q.split!='dev' for q in pilot):errors.append('Pilot must be 30 Dev questions')
    if len({q.category for q in pilot})!=6:errors.append('Pilot lacks category coverage')
    if any(q.model_dump()!=seen[q.query_id].model_dump() for q in pilot):errors.append('Pilot record mismatch')
    normalized=[re.sub(r'\W+',' ',q.question.lower()).strip() for q in questions]
    if len(set(normalized))!=len(normalized):errors.append('Duplicate question text')
    # This is an oracle union over every Chunk, not a measured retrieval score.
    from .retrieval import covered
    recoverability={}
    for path in meta['corpus_hashes']:
        if not path.endswith('chunks.jsonl'):continue
        ranges={}
        for c in map(json.loads,(root/path).read_text(encoding='utf-8').splitlines()):
            parts=c.get('content_parts')
            ranges.setdefault(c['section_id'],[]).extend([(p['section_start'],p['section_end']) for p in parts] if parts else [(c['section_start'],c['section_end'])])
        evidence=[e for q in questions for e in q.gold_evidence]
        hits=sum(covered(e.section_span,ranges.get(e.section_id,[])) for e in evidence)
        recoverability[path]=dict(evidence_annotations=len(evidence),fully_coverable=hits)
        if hits!=len(evidence):errors.append('Gold evidence cannot be fully reconstructed from Chunks: '+path)
    approved=sum(q.annotation_status=='approved' for q in questions)
    review_status='complete' if approved==len(questions) else 'partial_human_review' if any(q.annotation_status!='pending_review' for q in questions) else 'pending'
    return dict(status='failed' if errors else 'structural_checks_passed',manual_review_status=review_status,questions=len(questions),
                approved=sum(q.annotation_status=='approved' for q in questions),topic_groups=len(groups),
                evidence_spans=sum(len(q.gold_evidence) for q in questions),evidence_recoverability=recoverability,errors=errors)


def load_jsonl(path):
    return [EvalQuestion.model_validate_json(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
