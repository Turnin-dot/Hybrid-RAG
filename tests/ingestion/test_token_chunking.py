import ast
import json
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from hybrid_rag.ingestion.chunker import Chunker
from hybrid_rag.ingestion.token_chunking import TokenRefiner
from hybrid_rag.ingestion.validator import validate_records
from hybrid_rag.schemas.document import Document, Section, SourceSpan

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def tokenizer():
    transformers=pytest.importorskip('transformers')
    path=ROOT/'data/models/bge-base-en-v1.5-tokenizer'
    if not path.exists():pytest.skip('Local BGE-base-en tokenizer not installed')
    return transformers.AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)


def records(text,heading=None):
    spans=[];cursor=0
    for i,line in enumerate(text.splitlines(True),1):
        spans.append(SourceSpan(start=cursor,end=cursor+len(line),source_path='sample.mdx',source_line_start=i,source_line_end=i))
        cursor+=len(line)
    shared=dict(doc_id='doc',revision_id='rev',source_path='sample.mdx',source_spans=spans)
    document=Document(**shared,source_revision_id='raw',title='Page',content=text,cleaner_version='fixture')
    section=Section(**shared,section_id='section',heading_path=heading or ['Page','Section'],heading_level=2,section_index=0,
                    content=text,content_start=0,content_end=len(text))
    return document,section


def exercise(tokenizer,text,*,target=100,maximum=140,overlap=12,heading=None):
    doc,section=records(text,heading)
    base=Chunker(100000,0)
    originals=base.split(section)
    worker=TokenRefiner(tokenizer,target_tokens=target,max_tokens=maximum,overlap=overlap,chunk_size=100000)
    chunks=base.refine_tokens(section,originals,worker)
    assert validate_records(doc,[section],chunks,100000)==[]
    again=base.refine_tokens(section,originals,TokenRefiner(tokenizer,target_tokens=target,max_tokens=maximum,overlap=overlap,chunk_size=100000))
    assert [c.model_dump_json() for c in chunks]==[c.model_dump_json() for c in again]
    assert len({c.chunk_id for c in chunks})==len(chunks)
    return section,originals,chunks,worker


def test_small_chunks_completely_unchanged(tokenizer):
    _,before,after,worker=exercise(tokenizer,'Short paragraph.\n\n```python\nx=1 # comment\n```\n')
    assert before==after and before[0] is after[0] and not worker.findings


def test_complete_embedding_text_including_heading_and_special_tokens(tokenizer):
    heading=['Long heading '+('metadata ' * 60)]
    section,before,after,worker=exercise(tokenizer,'word '*100,target=110,maximum=145,heading=heading)
    assert worker.count(before[0].content)<=145<worker.count(before[0].embedding_text)
    assert len(after)>1 and all(worker.count(c.embedding_text)<=145 for c in after)
    assert all(c.embedding_text.startswith(' > '.join(section.heading_path)+'\n\n') for c in after)


def test_python_functions_decorators_comments_and_ast_not_executed(tokenizer,tmp_path):
    sentinel=tmp_path/'must-not-exist'
    body=f'from pathlib import Path\nPath({str(sentinel)!r}).touch()\n\n'+''.join(
        f'@decorator\nasync def function_{i}():\n    # retain this comment\n    return {i}\n\n' for i in range(35))
    section,originals,chunks,worker=exercise(tokenizer,'```python Example\n'+body+'```\n')
    assert len(chunks)>1 and all(c.split_method=='token_python_ast' for c in chunks)
    assert not sentinel.exists() and not worker.findings
    for c in chunks:
        fence,=[t for t in MarkdownIt().parse(c.content) if t.type=='fence']
        ast.parse(fence.content)
        assert worker.count(c.embedding_text)<=140
    owned=''.join(section.content[p.section_start:p.section_end] for c in chunks for p in c.content_parts if p.role=='body')
    assert owned==body


def test_table_rows_headers_and_overlap_are_explicit(tokenizer):
    table='| Name | Value |\n| --- | --- |\n'+''.join(f'| row_{i} | escaped \\| cell |\n' for i in range(55))
    _,_,chunks,worker=exercise(tokenizer,table,overlap=20)
    assert len(chunks)>1 and not worker.findings
    assert all(c.content.startswith('| Name | Value |') for c in chunks)
    assert all(worker.count(c.embedding_text)<=140 for c in chunks)
    assert any(p.role=='context' for c in chunks for p in c.content_parts)


def test_lists_keep_nested_code_and_whole_items(tokenizer):
    text=''
    for i in range(1,25):
        margin=' '*len(f'{i}. ')
        text+=f'{i}. Complete item {i}\n\n{margin}- Nested detail\n\n{margin}```python\n{margin}x={i}\n{margin}```\n\n'
    _,_,chunks,worker=exercise(tokenizer,text)
    assert len(chunks)>1 and all(c.split_method=='token_list_items' for c in chunks)
    assert not worker.findings
    assert all(len([t for t in MarkdownIt().parse(c.content) if t.type=='fence'])>=1 for c in chunks)


def test_logs_split_at_message_events(tokenizer):
    body=''.join(f'================ Ai Message ================\nEvent {i}: action completed.\n\n' for i in range(45))
    _,_,chunks,worker=exercise(tokenizer,'``` Result expandable\n'+body+'```\n')
    assert len(chunks)>1 and all(c.split_method=='token_log_events' for c in chunks)
    assert not worker.findings


@pytest.mark.parametrize('text',[
    ' '.join(f'word_{i}' for i in range(500)),
    '你好🙂 café e\u0301 数据。'*350,
    'same repeated phrase '*500,
])
def test_recursive_token_splitter_exact_source_and_overlap(tokenizer,text):
    section,_,chunks,worker=exercise(tokenizer,text)
    assert len(chunks)>1 and not worker.findings
    assert all(worker.count(c.embedding_text)<=140 for c in chunks)
    assert any(p.role=='context' for c in chunks[1:] for p in c.content_parts)
    for c in chunks:
        repeated=''.join(section.content[p.section_start:p.section_end] for p in c.content_parts if p.role=='context')
        assert worker.count(repeated,False)<=12


@pytest.mark.parametrize('text',[
    '```python\ndef one_giant_function():\n'+''.join(f'    x_{i}={i}\n' for i in range(150))+'```\n',
    '| Name | Value |\n| --- | --- |\n| one | '+('huge ' * 500)+' |\n',
    '```tsx\n'+('const component = <Component />;\n'*100)+'```\n',
    '```python\ndef broken(:\n'+(' bad syntax\n'*100)+'```\n',
])
def test_giant_code_rows_and_invalid_syntax_are_lossless_continuations(tokenizer,text):
    _,originals,chunks,worker=exercise(tokenizer,text)
    assert len(chunks)>1 and not worker.findings
    assert all(worker.count(c.embedding_text)<=140 for c in chunks)
    assert all(c.split_method=='token_source_continuation' for c in chunks)
    assert worker.reconstructions[0]['original_sha256']==worker.reconstructions[0]['reconstructed_sha256']


def test_unusable_heading_budget_is_explicit(tokenizer):
    _,before,after,worker=exercise(tokenizer,'Body text.\n',heading=['metadata '*200])
    assert before==after and worker.findings[0]['reason']=='unresolved_heading_budget'


def test_single_huge_log_line_and_actual_prefix_are_bounded(tokenizer):
    body='================ Ai Message ================\n'+json.dumps({'content':' '.join(f'value_{i}' for i in range(1800))})+'\n'
    doc,section=records('```Result\n'+body+'```\n')
    original=Chunker(100000,0).split(section)
    worker=TokenRefiner(tokenizer,target_tokens=100,max_tokens=140,overlap=12,chunk_size=100000,input_prefix='passage: ')
    chunks=worker.refine(section,original)
    assert len(chunks)>2 and not worker.findings
    assert all(c.embedding_text.startswith('passage: ') and worker.count(c.embedding_text)<=140 for c in chunks)
    assert all('code' in c.block_types for c in chunks)
    assert validate_records(doc,[section],chunks,100000)==[]
    assert worker.reconstructions[0]['original_sha256']==worker.reconstructions[0]['reconstructed_sha256']


def test_impossible_heading_fails_export_without_dropping_chunk(tokenizer,tmp_path):
    from hybrid_rag.ingestion.pipeline import refine_export
    doc,s=records('Keep this text unchanged.',heading=['metadata '*200])
    original=Chunker(800,120).split(s)
    source=tmp_path/'source';source.mkdir()
    for name,rows in [('documents',[doc]),('sections',[s]),('chunks',original)]:
        (source/(name+'.jsonl')).write_text(''.join(r.model_dump_json()+'\n' for r in rows),encoding='utf-8')
    (source/'ingestion_report.json').write_text(json.dumps(dict(config=dict(chunk_size=800,chunk_overlap=120),chunker_version='fixture')),encoding='utf-8')
    result=refine_export(source,tmp_path/'out',ROOT/'data/models/bge-base-en-v1.5-tokenizer',target_tokens=100,max_tokens=140,token_overlap=12)
    assert result['status']=='failed' and result['token_refinement']['hard_limit_acceptance']=='failed'
    assert len((tmp_path/'out/chunks.jsonl').read_text(encoding='utf-8').splitlines())==1
    assert result['token_refinement']['unresolved'][0]['reason']=='unresolved_heading_budget'


def test_config_validation(tokenizer):
    for options in [dict(target_tokens=0),dict(target_tokens=600),dict(overlap=384),dict(overlap=-1)]:
        with pytest.raises(ValueError):TokenRefiner(tokenizer,**options)


def test_unchanged_sibling_identity_and_index(tokenizer):
    doc,section=records('Short before.\n\n```python\n'+''.join(f'def f{i}():\n    return {i}\n\n' for i in range(60))+'```\n\nShort after.\n')
    base=Chunker(100,0);originals=base.split(section)
    worker=TokenRefiner(tokenizer,target_tokens=100,max_tokens=140,overlap=12,chunk_size=100)
    chunks=base.refine_tokens(section,originals,worker)
    by={c.chunk_id:c for c in chunks}
    assert all(by[c.chunk_id]==c for c in originals if worker.count(c.embedding_text)<=140)
    assert len({c.chunk_index for c in chunks})==len(chunks)
    assert validate_records(doc,[section],chunks,100)==[]


def test_existing_prose_overlap_preserved_without_duplicate_ownership(tokenizer):
    doc,section=records('数据与来源映射🙂。'*500)
    base=Chunker(800,120);originals=base.split(section)
    worker=TokenRefiner(tokenizer,target_tokens=100,max_tokens=140,overlap=12,chunk_size=800)
    chunks=base.refine_tokens(section,originals,worker)
    assert len(chunks)>len(originals) and not worker.findings
    assert validate_records(doc,[section],chunks,800)==[]


def test_four_file_export_repeats_and_original_bytes(tokenizer,tmp_path):
    from hybrid_rag.ingestion.pipeline import refine_export
    doc,s=records('First paragraph.\n\n```python\n'+''.join(f'def f{i}():\n    return {i}\n\n' for i in range(80))+'```\n\nLast paragraph.\n')
    originals=Chunker(100,0).split(s)
    source=tmp_path/'source';source.mkdir()
    for filename,rows in [('documents',[doc]),('sections',[s]),('chunks',originals)]:
        (source/(filename+'.jsonl')).write_text(''.join(r.model_dump_json()+'\n' for r in rows),encoding='utf-8')
    (source/'ingestion_report.json').write_text(json.dumps(dict(config=dict(chunk_size=100,chunk_overlap=0),chunker_version='fixture',cleaning_findings=[],failures=[])),encoding='utf-8')
    kwargs=dict(target_tokens=100,max_tokens=140,token_overlap=12)
    a,b=tmp_path/'a',tmp_path/'b'
    report=refine_export(source,a,ROOT/'data/models/bge-base-en-v1.5-tokenizer',**kwargs)
    refine_export(source,b,ROOT/'data/models/bge-base-en-v1.5-tokenizer',**kwargs)
    assert report['quality_errors']==0 and report['token_refinement']['repeatable']
    assert report['token_refinement']['after']['over_max']==0
    assert {p.name for p in a.iterdir()}=={'documents.jsonl','sections.jsonl','chunks.jsonl','ingestion_report.json'}
    assert all(p.read_bytes()==(b/p.name).read_bytes() for p in a.iterdir())
    for name in ['documents.jsonl','sections.jsonl']:assert (source/name).read_bytes()==(a/name).read_bytes()
    assert report['token_refinement']['small_chunks_unchanged']
    with pytest.raises(FileExistsError):refine_export(source,a,ROOT/'data/models/bge-base-en-v1.5-tokenizer',**kwargs)
