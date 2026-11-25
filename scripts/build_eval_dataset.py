"""Build reviewable English candidates from explicitly authored Section topics."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hybrid_rag.evaluation.schema import EvalQuestion
from eval_topics import TOPICS, UNANSWERABLE

VERSION='hybridrag-eval-en-v2-candidate'


def sha(data): return hashlib.sha256(data).hexdigest()
def read(path): return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
def write(path,rows): path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


def build(output,version=VERSION):
    if not version.strip():raise ValueError('Dataset version must not be empty')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    corpus=ROOT/'data/processed/phase1-v3-final'
    docs={d['doc_id']:d for d in read(corpus/'documents.jsonl')}
    sections=read(corpus/'sections.jsonl'); lookup={(s['source_path'],s['section_index']):s for s in sections}
    ingestion=json.loads((corpus/'ingestion_report.json').read_text(encoding='utf-8'))
    snapshot={p:sha((ROOT/p).read_bytes()) for p in ['data/processed/phase1-v3-final/documents.jsonl','data/processed/phase1-v3-final/sections.jsonl','data/processed/phase1-v3-final/chunks.jsonl','data/processed/phase1-v3-final/ingestion_report.json','data/processed/chunk-comparison-v2/improved.chunks.jsonl']}
    def evidence(fact,index,fid):
        s=lookup[(fact['path'],fact['index'])];d=docs[s['doc_id']]
        anchor=fact['anchor'];position=s['content'].find(anchor)
        if position<0: raise ValueError(f"Missing anchor: {fact}")
        # Paragraph-sized gold spans, not full chunks or arbitrary sentence guesses.
        start=s['content'].rfind('\n\n',0,position)+2
        if start==1:start=0
        end=s['content'].find('\n\n',position+len(anchor))
        if end<0:end=len(s['content'])
        quote=s['content'][start:end]
        if fact.get('extent')=='line':
            start=s['content'].rfind('\n',0,position)+1
            end=s['content'].find('\n',position+len(anchor))
            if end<0:end=len(s['content'])
            quote=s['content'][start:end]
        elif fact.get('extent')=='lead':
            start=0;end=s['content'].find('```')
            if end<0:end=len(s['content'])
            quote=s['content'][start:end]
        a,b=s['content_start']+start,s['content_start']+end
        assert d['content'][a:b]==quote and anchor in quote
        raw={}
        for span in d['source_spans']:
            if span['start']<b and span['end']>a:
                p=span['source_path'];info=raw.setdefault(p,{'source_path':p,'line_start':span['source_line_start'],'line_end':span['source_line_end']})
                info['line_start']=min(info['line_start'],span['source_line_start']);info['line_end']=max(info['line_end'],span['source_line_end'])
        for p,info in raw.items():
            data=(ROOT/'docs/src'/p).read_bytes()
            info['sha256']=sha(data)
            expected=ingestion['input_sha256'].get(p) or next((x['content_hash'] for x in d['dependencies'] if x['source_path']==p),None)
            if expected and expected!=info['sha256']:raise ValueError('Source changed: '+p)
            target=output/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        return dict(evidence_id=f'e{index}',section_id=s['section_id'],doc_id=s['doc_id'],source_path=s['source_path'],heading_path=s['heading_path'],section_span=[start,end],evidence_span=[a,b],span_basis='cleaned_document_content',quote=quote,supports=[fid],raw_sources=list(raw.values()))
    questions=[];category_counts=Counter();group_counts=Counter();dev_counts={'api':4,'concept':4,'semantic_rewrite':3,'multi_evidence':4,'english_terminology':3,'unanswerable':2}
    topics=[*TOPICS,*[dict(category='unanswerable',questions=q,facts=[],reason=r,answer=a) for r,q,a in UNANSWERABLE]]
    for group_index,topic in enumerate(topics,1):
        category=topic['category']
        if category not in dev_counts:raise ValueError('Unsupported topic category: '+category)
        group_counts[category]+=1
        split='dev' if group_counts[category]<=dev_counts[category] else 'test'
        facts=[dict(fact_id=f'f{i}',text=f['text'],required=True) for i,f in enumerate(topic['facts'],1)]
        gold=[evidence(f,i,f'f{i}') for i,f in enumerate(topic['facts'],1)]
        # Explicit equivalent evidence for the persistence comparison, in
        # addition to its table. Alternatives satisfy the same atomic fact.
        if group_index==13:
            # Table is the primary evidence; introductory explanation is an
            # independent alternative, to be confirmed by the human reviewer.
            alternate=dict(path='oss/langgraph/persistence.mdx',index=0,anchor='LangGraph provides two complementary persistence systems:')
            # Include both bullets, rather than the bare introductory sentence.
            s=lookup[(alternate['path'],0)];a=s['content'].find('- **[Checkpointers]');b=s['content'].find('\n\nMost applications')
            if a>=0 and b>a:
                alternate['anchor']=s['content'][a:b]
                alternate_e=evidence(alternate,99,'f1')
                alternate_e['supports']=[f['fact_id'] for f in facts]
                gold.append(alternate_e)
            else:raise ValueError('Equivalent persistence evidence not found')
        for question in topic['questions']:
            category_counts[category]+=1
            record=dict(query_id=f'q{len(questions)+1:03d}',dataset_version=version,topic_group=f'topic-{group_index:02d}',question=question,category=category,
                products=sorted({f['path'].split('/')[1] for f in topic['facts']}),split=split,answerable=bool(facts),reference_answer=' '.join(f['text'] for f in facts) if facts else topic['answer'],
                reference_facts=facts,gold_evidence=gold,unanswerable_reason=topic.get('reason'),
                absence_check=None if facts else dict(method='scope_and_information_requirements_review',status='pending_human_confirmation',product_scope='not_product_specific',note='Empty products means no specific product attribution. No fabricated gold span. Absence cannot be proved by keyword matching alone.'),annotation_status='pending_review')
            if group_index in (7,8):record['topic_group']='family-custom-agent-state'
            if group_index in (14,36):record['topic_group']='family-checkpoint-definition'
            record['section_relevance']={sid:3 if {fid for e in gold if e['section_id']==sid for fid in e['supports']} >= {f['fact_id'] for f in facts} else 2 for sid in sorted({e['section_id'] for e in gold})}
            questions.append(EvalQuestion.model_validate(record).model_dump(mode='json'))
    assert len(questions)==120 and sum(q['split']=='dev' for q in questions)==40
    # No shared evidence Section may leak across Dev/Test.
    seen={}
    for q in questions:
        for e in q['gold_evidence']:
            if e['section_id'] in seen and seen[e['section_id']]!=q['split']:raise ValueError('Section evidence leaks across splits')
            seen[e['section_id']]=q['split']
    pilot=[]
    for category,count in {'api':6,'concept':6,'semantic_rewrite':4,'english_terminology':4,'multi_evidence':6,'unanswerable':4}.items():
        pilot.extend([q for q in questions if q['split']=='dev' and q['category']==category][:count])
    lines=['# English candidate evaluation set\n','All labels are pending human review. Questions are authored from evidence; they do not literally occur in the source. Source positions below support the answer facts.\n']
    for number,q in enumerate(questions,1):
        lines.extend([f"## {q['query_id']} | {q['split']} | {q['category']} | {q['topic_group']}\n",'### Question\n'])
        qline=len(lines)+1;lines.append(q['question']+'\n')
        lines.append('### Reference answer\n');aline=len(lines)+1;lines.append(q['reference_answer']+'\n')
        q['review_locations']={'questions_jsonl_line':number,'review_markdown':'review_all.md','question_line':qline,'answer_line':aline}
        lines.extend(['### Atomic reference facts\n',*['- '+f['fact_id']+': '+f['text']+'\n' for f in q['reference_facts']]])
        for e in q['gold_evidence']:
            lines.extend([f"### Evidence {e['evidence_id']} supports {', '.join(e['supports'])}\n",f"Section `{e['section_id']}`; {' > '.join(e['heading_path'])}\n",f"Cleaned Document characters [{e['evidence_span'][0]}, {e['evidence_span'][1]}); Section characters [{e['section_span'][0]}, {e['section_span'][1]}).\n"])
            for raw in e['raw_sources']:
                lines.append(f"Original/snapshot: `{raw['source_path']}`, lines {raw['line_start']}-{raw['line_end']} (inclusive).\n")
            marker='`'*max(4,max((len(m.group()) for m in re.finditer(r'`+',e['quote'])),default=0)+1)
            # Each list element is exactly one physical line, including quotes.
            lines.append(marker+'text\n');lines.extend(line+'\n' for line in e['quote'].splitlines());lines.append(marker+'\n')
        if not q['answerable']:lines.append('No gold evidence: '+q['unanswerable_reason']+'. Human confirmation required.\n')
        lines.append('\n')
    (output/'review_all.md').write_text(''.join(lines),encoding='utf-8')
    # Pilot contains all six categories, only from Dev, with its own exact locations.
    pilot_lines=['# Pilot 30: priority human review\n','All questions and answers are English; every label is pending review.\n']
    starts=[q['review_locations']['question_line']-3 for q in questions]+[len(lines)]
    for q in pilot:
        index=questions.index(q);block=lines[starts[index]:starts[index+1]]
        offset=len(pilot_lines)-starts[index]
        q['review_locations'].update(pilot_markdown='review_pilot30.md',pilot_question_line=q['review_locations']['question_line']+offset,pilot_answer_line=q['review_locations']['answer_line']+offset)
        pilot_lines.extend(block)
    (output/'review_pilot30.md').write_text(''.join(pilot_lines),encoding='utf-8')
    write(output/'questions.jsonl',questions)
    write(output/'dev.jsonl',[q for q in questions if q['split']=='dev']);write(output/'test.jsonl',[q for q in questions if q['split']=='test'])
    write(output/'pilot30.jsonl',pilot)
    write(output/'review_decisions.jsonl',[dict(query_id=q['query_id'],decision='pending_review',reviewer=None,checks={k:False for k in ['clarity','atomic_facts','evidence_sufficiency','answerability','split_integrity']},notes='') for q in questions])
    meta=dict(dataset_version=version,status='candidate_pending_human_review',created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),repository_commit=ingestion['repository_commit'],corpus_hashes=snapshot,
        ingestion_config_hash=sha(json.dumps(ingestion['config'],sort_keys=True).encode()),annotation_guidelines_version='en-evidence-v2',seed_sha256=sha((ROOT/'scripts/eval_topics.py').read_bytes()),builder_sha256=sha(Path(__file__).read_bytes()),
        count=len(questions),topic_groups=len({q['topic_group'] for q in questions}),category_counts=dict(category_counts),split_counts=dict(Counter(q['split'] for q in questions)),pilot_ids=[q['query_id'] for q in pilot],
        annotation_method='assistant_authored_from_frozen_source; pending_human_review',
        policy='English only; terminology replaces the planned mixed-language category. Pilot is a Dev-only subset. All 120 are draft candidates, not human gold.',
        raw_snapshot_hashes={str(p.relative_to(output/'source_snapshot')):sha(p.read_bytes()) for p in sorted((output/'source_snapshot').rglob('*')) if p.is_file()})
    meta['artifacts_sha256']={name:sha((output/name).read_bytes()) for name in ['questions.jsonl','dev.jsonl','test.jsonl','pilot30.jsonl','review_all.md','review_pilot30.md']}
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:meta[k] for k in ['status','count','topic_groups','category_counts','split_counts']},indent=2))
    return questions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/eval/hybridrag-en-v2');p.add_argument('--version',default=VERSION);a=p.parse_args();build(a.output,a.version)
