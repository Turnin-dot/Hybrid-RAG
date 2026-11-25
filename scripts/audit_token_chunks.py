"""Independently recount saved embedding_text and reconstruct every parent."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(source, export, tokenizer, repeat=None, maximum=512):
    source,export=Path(source),Path(export)
    def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
    def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
    old=rows(source/'chunks.jsonl');new=rows(export/'chunks.jsonl')
    old_by={c['chunk_id']:c for c in old};new_by={c['chunk_id']:c for c in new}
    report=json.loads((export/'ingestion_report.json').read_text(encoding='utf-8'))['token_refinement']
    original_report=json.loads((source/'ingestion_report.json').read_text(encoding='utf-8'))
    sections={s['section_id']:s for s in rows(source/'sections.jsonl')}
    errors=[]
    lengths=[]
    # A fresh batch Tokenizer invocation, not the refiner's cached statistics.
    for i in range(0,len(new),64):
        lengths.extend(map(len,tokenizer([c['embedding_text'] for c in new[i:i+64]],add_special_tokens=True,truncation=False,padding=False)['input_ids']))
    counts=dict(zip([c['chunk_id'] for c in new],lengths))
    if any(n>maximum for n in lengths):errors.append('embedding_text_over_hard_limit')
    if len(new_by)!=len(new):errors.append('duplicate_id')
    if any(not c['content'].strip() or not c['embedding_text'].strip() for c in new):errors.append('empty_chunk')
    for name in ['documents.jsonl','sections.jsonl']:
        if (source/name).read_bytes()!=(export/name).read_bytes():errors.append(name+'_changed')
    before_coverage={sid:bytearray(len(s['content'])) for sid,s in sections.items()}
    after_coverage={sid:bytearray(len(s['content'])) for sid,s in sections.items()}
    for records,masks in [(old,before_coverage),(new,after_coverage)]:
        for c in records:
            parts=c.get('content_parts') or [dict(start=0,end=len(c['content']),section_start=c['section_start'],section_end=c['section_end'])]
            cursor=0;expected_spans=[]
            for p in parts:
                a,b=p['section_start'],p['section_end']
                if p['start']!=cursor or c['content'][p['start']:p['end']]!=sections[c['section_id']]['content'][a:b]:errors.append('source_slice_or_order')
                for span in sections[c['section_id']]['source_spans']:
                    if span['start']<b and span['end']>a:
                        expected_spans.append(dict(span,start=p['start']+max(span['start'],a)-a,end=p['start']+min(span['end'],b)-a))
                cursor=p['end'];masks[c['section_id']][a:b]=b'\1'*(b-a)
            if cursor!=len(c['content']):errors.append('incomplete_part_mapping')
            if c['source_spans']!=expected_spans:errors.append('source_path_or_original_line_mapping_changed')
            cursor=0
            for span in c['source_spans']:
                if span['start']!=cursor or span['end']>len(c['content']):errors.append('source_span_offsets')
                if not 1<=span['source_line_start']<=span['source_line_end']:errors.append('source_line_range')
                cursor=span['end']
            if cursor!=len(c['content']):errors.append('source_span_coverage')
    total=sum(sum(mask) for mask in before_coverage.values())
    kept=sum(sum(bool(v) and bool(after_coverage[sid][i]) for i,v in enumerate(mask)) for sid,mask in before_coverage.items())
    if kept!=total:errors.append('source_content_lost')
    changes={r['parent_chunk_id']:r for r in report['parent_reassembly']}
    accounted=set();reassembled=0
    for parent in old:
        pid=parent['chunk_id']
        if pid not in changes:
            expected=dict(parent,embedding_text=report['config']['input_prefix']+parent['embedding_text'])
            if new_by.get(pid)!=expected:errors.append('untouched_parent_missing_or_changed')
            accounted.add(pid);continue
        relation=changes[pid];restored=['']*len(parent['content']);body_seen=set();last=-1
        for child in relation['children']:
            c=new_by[child['chunk_id']];accounted.add(c['chunk_id']);cursor=0
            if (c['doc_id'],c['section_id'])!=(parent['doc_id'],parent['section_id']):errors.append('parent_identity_changed')
            for p in child['parent_ranges']:
                a,b=p['start'],p['end'];piece=c['content'][cursor:cursor+b-a];cursor+=b-a
                if piece!=parent['content'][a:b]:errors.append('parent_slice_changed')
                restored[a:b]=piece
                if p['role']=='body':
                    fresh=[i for i in range(a,b) if i not in body_seen]
                    if fresh and fresh[0]<last:errors.append('parent_body_reordered')
                    if fresh:last=fresh[-1]
                    body_seen.update(range(a,b))
            if cursor!=len(c['content']):errors.append('lineage_does_not_cover_child')
        text=''.join(restored)
        if text!=parent['content'] or digest(text)!=relation['original_sha256']:errors.append('parent_reassembly_failed')
        reassembled+=1
    if accounted!=set(new_by):errors.append('orphan_or_skipped_chunks')
    previous=[]
    for unresolved in original_report.get('token_refinement',{}).get('unresolved',[]):
        pid=unresolved['chunk_id'];relation=changes.get(pid)
        children=[c['chunk_id'] for c in relation['children']] if relation else [pid]
        previous.append(dict(original_chunk_id=pid,source_path=unresolved['source_path'],previous_m3_tokens=unresolved['tokens'],
            outcome='split' if relation else 'already_fits_target_tokenizer',children=[dict(chunk_id=k,tokens=counts[k]) for k in children],
            maximum=max(counts[k] for k in children)))
    repeated=None
    if repeat:
        repeated={name:(export/name).read_bytes()==(Path(repeat)/name).read_bytes() for name in ['documents.jsonl','sections.jsonl','chunks.jsonl','ingestion_report.json']}
        if not all(repeated.values()):errors.append('four_file_repeat_mismatch')
    ordered=sorted(lengths)
    return dict(status='failed' if errors else 'passed',tokenizer='BAAI/bge-base-en-v1.5',add_special_tokens=True,truncation=False,
        field='embedding_text',input_prefix=report['config']['input_prefix'],chunks=len(new),over_512=sum(n>maximum for n in lengths),
        distribution={f'p{p}':ordered[int((len(ordered)-1)*p/100)] for p in (0,50,90,95,99,100)},
        covered_source_characters=kept,total_source_characters=total,source_coverage=kept/total if total else 1,
        parents_reassembled=reassembled,parents_untouched=len(old)-reassembled,previous_46=previous,four_file_repeat=repeated,errors=errors)


if __name__=='__main__':
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,default=Path(__file__).resolve().parents[1]/'data/models/bge-base-en-v1.5-tokenizer')
    parser.add_argument('--repeat',type=Path);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
    result=audit(args.source,args.export,tokenizer,args.repeat)
    if args.report.exists():raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='previous_46'},indent=2));raise SystemExit(int(bool(result['errors'])))
