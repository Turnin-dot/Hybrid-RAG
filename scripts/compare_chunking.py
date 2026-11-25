"""Compare frozen sections with baseline and optional structural optimization.

No Cleaner rerun, model inference or network calls. A local BGE-M3 tokenizer
is required; truncation is disabled for both content and embedding_text.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from transformers import AutoTokenizer
from hybrid_rag.schemas.document import Document, Section, Chunk
from hybrid_rag.ingestion.chunker import Chunker, CHUNKER_VERSION
from hybrid_rag.ingestion.improved_chunker import ImprovedChunker, IMPROVED_VERSION
from hybrid_rag.ingestion.validator import validate_records


def load(path, model):
    return [model.model_validate_json(s) for s in path.read_text(encoding='utf-8').splitlines()]


def distribution(values):
    ordered = sorted(values)
    return {**{f'p{p}':ordered[int((len(ordered)-1)*p/100)] if ordered else 0 for p in (0, 5, 50, 90, 95, 99, 100)},
            'mean':sum(values)/len(values) if values else 0}


def run(source, output, tokenizer_path, size=800, overlap=120, options=None):
    source, output, tokenizer_path = Path(source), Path(output), Path(tokenizer_path)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    documents = load(source/'documents.jsonl', Document)
    sections = load(source/'sections.jsonl', Section)
    original = load(source/'chunks.jsonl', Chunk)
    options = options or dict(python=True,tables=True,lists=True,captions=True)
    metadata = tokenizer_path/'.cache/huggingface/download/tokenizer.json.metadata'
    report = {'status':'pending', 'retrieval_quality':'not_evaluated', 'size':size, 'overlap':overlap, 'improved_options':options,
        'strategy_versions':{'baseline':CHUNKER_VERSION,'improved':IMPROVED_VERSION},
        'implementation_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
             [ROOT/'src/hybrid_rag/ingestion/chunker.py',ROOT/'src/hybrid_rag/ingestion/improved_chunker.py',ROOT/'src/hybrid_rag/ingestion/validator.py',ROOT/'src/hybrid_rag/ingestion/structured_validation.py',ROOT/'src/hybrid_rag/schemas/document.py',Path(__file__)]},
        'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.jsonl')},
        'tokenizer':{'name':'BAAI/bge-m3','revision':metadata.read_text().splitlines()[0] if metadata.exists() else None,
                     'model_card':'https://huggingface.co/BAAI/bge-m3',
                     'local_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tokenizer_path.glob('*')) if p.is_file()},
                     'add_special_tokens':True,'truncation':False,'lengths':['content','embedding_text'],
                     'versions':{p:version(p) for p in ('transformers','tokenizers')}}, 'strategies':{}}
    section_groups = defaultdict(list)
    for s in sections: section_groups[s.doc_id].append(s)
    samples = []
    for name, splitter in [('baseline', Chunker(size, overlap)), ('improved', ImprovedChunker(size, overlap, **options))]:
        chunks = [c for section in sections for c in splitter.split(section)]
        repeated = [c for section in sections for c in splitter.split(section)]
        encoded = ''.join(c.model_dump_json()+'\n' for c in chunks)
        repeatable = encoded == ''.join(c.model_dump_json()+'\n' for c in repeated)
        (output/(name+'.chunks.jsonl')).write_text(encoded, encoding='utf-8', newline='\n')
        errors = []
        grouped = defaultdict(list)
        for c in chunks: grouped[c.doc_id].append(c)
        for d in documents:
            errors.extend({'source_path':d.source_path,'check':check} for check in validate_records(d, section_groups[d.doc_id], grouped[d.doc_id], size))
        token_lengths = {}; metrics = []
        for field in ('content','embedding_text'):
            lengths = []
            for i in range(0,len(chunks),64):
                texts = [getattr(c,field) for c in chunks[i:i+64]]
                lengths.extend(map(len,tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)['input_ids']))
            token_lengths[field] = lengths
        for i,c in enumerate(chunks):
            metrics.append({'chunk_id':c.chunk_id,'characters':len(c.content),'content_tokens':token_lengths['content'][i],
                            'embedding_tokens':token_lengths['embedding_text'][i], 'source_path':c.source_path,'split_method':c.split_method})
        (output/(name+'.metrics.jsonl')).write_text(''.join(json.dumps(m)+'\n' for m in metrics),encoding='utf-8')
        duplicate_text = Counter(c.content for c in chunks)
        locations = defaultdict(list)
        for c in chunks: locations[c.content].append({'source_path':c.source_path,'section_id':c.section_id,'start':c.section_start,'end':c.section_end,'chunk_id':c.chunk_id})
        duplicate_groups = [v for k,v in locations.items() if len(v)>1]
        source_locations = Counter((c.section_id, tuple((p.section_start,p.section_end) for p in c.content_parts if p.role=='body')
                                   if c.content_parts else ((c.section_start,c.section_end),)) for c in chunks)
        (output/(name+'.duplicates.json')).write_text(json.dumps(duplicate_groups,ensure_ascii=False,indent=2),encoding='utf-8')
        # Overlap statistics are source-position based: intentional text overlap
        # and repeated headers/wrappers must not be called anomalous duplicates.
        covered_total = total = 0
        for s in sections:
            coverage = bytearray(len(s.content))
            for c in grouped[s.doc_id]:
                if c.section_id != s.section_id:continue
                ranges = [(p.section_start,p.section_end) for p in c.content_parts] if c.content_parts else [(c.section_start,c.section_end)]
                for a,b in ranges: coverage[a:b] = b'\1'*(b-a)
            total += sum(not x.isspace() for x in s.content)
            covered_total += sum(bool(coverage[i]) for i,x in enumerate(s.content) if not x.isspace())
        summary = {'chunks':len(chunks),'oversized':sum(c.oversized for c in chunks),
            'oversized_ratio':sum(c.oversized for c in chunks)/len(chunks), 'characters':distribution([len(c.content) for c in chunks]),
            'content_tokens':distribution(token_lengths['content']), 'embedding_tokens':distribution(token_lengths['embedding_text']),
            'embedding_over_8192':sum(n>8192 for n in token_lengths['embedding_text']),
            'short_under_100_chars':sum(len(c.content)<100 for c in chunks),
            'short_under_20_content_tokens':sum(n<20 for n in token_lengths['content']),
            'oversized_by_types':dict(Counter(','.join(c.block_types) for c in chunks if c.oversized)),
            'split_methods':dict(Counter(c.split_method or 'unchanged' for c in chunks)),
            'duplicate_id_count':len(chunks)-len({c.chunk_id for c in chunks}),
            'duplicate_body_location_count':sum(n-1 for n in source_locations.values() if n>1),
            'repeated_table_header_characters':sum(p.end-p.start for c in chunks for p in c.content_parts if p.role=='context'),
            'total_content_tokens':sum(token_lengths['content']), 'total_embedding_tokens':sum(token_lengths['embedding_text']),
            'exact_text_duplicate_groups':len(duplicate_groups),
            'duplicate_text_extra_occurrences':sum(n-1 for n in duplicate_text.values() if n>1),
            'coverage':covered_total/total if total else 1,'covered_nonwhitespace':covered_total,'total_nonwhitespace':total,
            'repeatable':repeatable,'validation_errors':errors}
        if name == 'baseline':
            summary['matches_frozen_baseline_ids_and_content'] = [(c.chunk_id,c.content) for c in chunks] == [(c.chunk_id,c.content) for c in original]
        report['strategies'][name]=summary
        by_section = defaultdict(list)
        for c in chunks: by_section[c.section_id].append(c)
        if name=='improved':
            methods = Counter()
            for c in chunks:
                method=c.split_method or 'retained_oversized'
                if (c.split_method or c.oversized) and methods[method]<4:
                    methods[method]+=1
                    samples.append({'method':method,'section_id':c.section_id,'source_path':c.source_path,
                                    'heading_path':c.heading_path,'chunk':c.model_dump(mode='json')})
    report['status']='failed' if any(s['validation_errors'] or not s['repeatable'] or s['coverage']!=1 or s['duplicate_id_count'] or s['duplicate_body_location_count'] for s in report['strategies'].values()) or not report['strategies']['baseline']['matches_frozen_baseline_ids_and_content'] else 'structural_checks_passed'
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/processed/phase1-v3-final')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,default=ROOT/'data/models/bge-m3-tokenizer')
    p.add_argument('--chunk-size',type=int,default=800)
    p.add_argument('--chunk-overlap',type=int,default=120)
    p.add_argument('--no-python',action='store_true')
    p.add_argument('--no-tables',action='store_true')
    p.add_argument('--no-lists',action='store_true')
    p.add_argument('--no-captions',action='store_true')
    a=p.parse_args()
    raise SystemExit(int(run(a.source,a.output,a.tokenizer,a.chunk_size,a.chunk_overlap,
        dict(python=not a.no_python,tables=not a.no_tables,lists=not a.no_lists,captions=not a.no_captions))['status']=='failed'))
