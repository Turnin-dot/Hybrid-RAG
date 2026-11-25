import ast
import pytest
from markdown_it import MarkdownIt
from hybrid_rag.ingestion.cleaner import MDXCleaner
from hybrid_rag.ingestion.parser import SectionParser
from hybrid_rag.ingestion.chunker import Chunker
from hybrid_rag.ingestion.improved_chunker import ImprovedChunker
from hybrid_rag.ingestion.validator import validate_records


def exercise(tmp_path, body, size=150, **options):
    path=tmp_path/'page.mdx'
    text='---\ntitle: Test\n---\n## Section\n'+body
    path.write_text(text,encoding='utf-8')
    cleaner=MDXCleaner(tmp_path)
    doc=cleaner.clean(cleaner.from_text(text,source_path='page.mdx'))
    sections=SectionParser().parse(doc)
    chunker=ImprovedChunker(size,20,**options)
    chunks=[c for s in sections for c in chunker.split(s)]
    assert not validate_records(doc,sections,chunks,size)
    assert [c.model_dump_json() for c in chunks]==[c.model_dump_json() for s in sections for c in chunker.split(s)]
    return sections,chunks


def test_python_decorators_async_and_classes(tmp_path):
    body='import math\n\n'+''.join(f'@decorator\nasync def f{i}():\n    # retain comment\n    return {i}\n\n' for i in range(8))+'class C:\n    def m(self):\n        return 1\n'
    _,chunks=exercise(tmp_path,'```python Example\n'+body+'```\n')
    assert len(chunks)>1 and all(c.split_method=='python_ast' for c in chunks)
    reconstructed=''
    for chunk in chunks:
        token,=[t for t in MarkdownIt().parse(chunk.content) if t.type=='fence']
        ast.parse(token.content)
        reconstructed+=token.content
    assert reconstructed==body


@pytest.mark.parametrize('body', ['def bad(:\n    ???\n'*30, 'class Giant:\n'+''.join(f'    def m{i}(self):\n        return {i}\n' for i in range(20))])
def test_invalid_or_single_giant_class_retained(tmp_path,body):
    sections,chunks=exercise(tmp_path,'```python\n'+body+'```\n')
    assert len(chunks)==1 and chunks[0].oversized and not chunks[0].content_parts


def test_table_headers_repeated_rows_unchanged(tmp_path):
    table='| Name | Value |\n| --- | --- |\n'+''.join(f'| row{i} | escaped \\| cell |\n' for i in range(15))
    _,chunks=exercise(tmp_path,table)
    assert len(chunks)>1 and all(c.split_method=='table_rows' for c in chunks)
    assert all('| Name | Value |' in c.content for c in chunks)
    assert sum(p.role=='context' for c in chunks for p in c.content_parts)==len(chunks)-1


def test_huge_row_stays_whole(tmp_path):
    table='| Name | Value |\n| --- | --- |\n| huge | '+'x'*500+' |\n| small | ok |\n'
    _,chunks=exercise(tmp_path,table)
    assert any(c.oversized and 'x'*500 in c.content for c in chunks)


def test_nested_list_items_and_code_are_not_cut(tmp_path):
    body=''.join(f'{i}. Item {i}:\n\n   - Nested explanation\n\n   ```python\n   value = {i}\n   ```\n\n' for i in range(1,7))
    _,chunks=exercise(tmp_path,body,size=200)
    assert len(chunks)>1 and all(c.split_method=='list_items' for c in chunks)


def test_disabled_optimizations_preserve_baseline(tmp_path):
    sections,chunks=exercise(tmp_path,'```python\n'+''.join(f'def f{i}():\n    return {i}\n' for i in range(10))+'```\n',python=False,tables=False,lists=False,captions=False)
    baseline=[c for s in sections for c in Chunker(150,20).split(s)]
    assert [c.model_dump_json() for c in chunks]==[c.model_dump_json() for c in baseline]


def test_corruption_is_caught(tmp_path):
    sections,chunks=exercise(tmp_path,'| a | b |\n| --- | --- |\n'+''.join(f'| {i} | value |\n' for i in range(20)))
    from hybrid_rag.ingestion.structured_validation import table_values
    assert table_values(chunks[0].content)[0]==['a','b']
    chunks[0].content=chunks[0].content.replace('value','LOST',1)
    # Exact source-part checks catch even syntactically valid corruption.
    cleaner=MDXCleaner(tmp_path)
    doc=cleaner.clean(cleaner.from_text((tmp_path/'page.mdx').read_text(encoding='utf-8'),source_path='page.mdx'))
    assert 'chunk_part_source_slice' in validate_records(doc,sections,chunks,150)


def test_ast_analysis_does_not_execute_source(tmp_path):
    sentinel=tmp_path/'never-execute.txt'
    body=f'from pathlib import Path\nPath({str(sentinel)!r}).write_text("unsafe")\n'+''.join(f'def f{i}():\n    return {i}\n' for i in range(10))
    exercise(tmp_path,'```python\n'+body+'```\n')
    assert not sentinel.exists()


def test_short_caption_stays_with_related_code(tmp_path):
    body='**Sync**\n\n```python\n'+''.join(f'def f{i}():\n    return {i}\n' for i in range(8))+'```\n'
    _,chunks=exercise(tmp_path,body)
    assert chunks[0].content.startswith('**Sync**') and 'code' in chunks[0].block_types
    assert len(chunks)>1


def test_pipeline_strategy_is_opt_in(tmp_path):
    from hybrid_rag.ingestion.pipeline import run_pipeline
    root=tmp_path/'docs/src'
    directory=root/'oss/langgraph'
    directory.mkdir(parents=True)
    (directory/'page.mdx').write_text('---\ntitle: Test\n---\n```python\n'+''.join(f'def f{i}():\n    return {i}\n\n' for i in range(20))+'```\n',encoding='utf-8')
    baseline=run_pipeline(root,tmp_path/'baseline',products=('langgraph',),chunk_size=150,chunk_overlap=20)
    improved=run_pipeline(root,tmp_path/'improved',products=('langgraph',),chunk_size=150,chunk_overlap=20,strategy='improved')
    assert baseline['config']['strategy']=='baseline'
    assert baseline['chunks_generated']==1
    assert improved['chunks_generated']>1
    assert baseline['quality_errors']==improved['quality_errors']==0
