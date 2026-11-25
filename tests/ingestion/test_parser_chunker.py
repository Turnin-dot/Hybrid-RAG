from pathlib import Path
import json
import pytest
from markdown_it import MarkdownIt
from hybrid_rag.ingestion.cleaner import MDXCleaner
from hybrid_rag.ingestion.parser import SectionParser
from hybrid_rag.ingestion.chunker import Chunker
from hybrid_rag.ingestion.pipeline import run_pipeline
from hybrid_rag.ingestion.validator import validate_records

ROOT = Path(__file__).parents[2]


def parsed(tmp_path, body):
    path = tmp_path / "page.mdx"
    text = "---\ntitle: Page\n---\n" + body
    path.write_text(text, encoding="utf-8")
    cleaner = MDXCleaner(tmp_path)
    doc = cleaner.clean(cleaner.from_text(text, source_path="page.mdx"))
    return doc, SectionParser().parse(doc)


def test_hierarchy_duplicate_titles_jumps_and_code(tmp_path):
    doc, sections = parsed(tmp_path, 'Intro\n\n## Setup\nDirect\n\n#### Same\nA\n\n#### Same\n```python\n# Not a heading\n```\n## Other\nB\n')
    assert [s.heading_level for s in sections] == [0, 2, 4, 4, 2]
    assert sections[2].parent_section_id == sections[1].section_id == sections[3].parent_section_id
    assert sections[2].heading_path == ["Page", "Setup", "Same"]
    assert sections[2].section_id != sections[3].section_id
    assert "A" not in sections[1].content
    chunks = [c for s in sections for c in Chunker().split(s)]
    assert not validate_records(doc, sections, chunks, 800)


def test_matching_h1_and_empty_parent(tmp_path):
    doc, sections = parsed(tmp_path, '# Page\n## Parent\n### Child\nBody\n')
    assert sections[0].heading_path == ["Page"]
    assert not Chunker().split(sections[1])
    assert sections[2].parent_section_id == sections[1].section_id


def test_atomic_code_table_and_recursive_overlap(tmp_path):
    table = "| Field | Value |\n| --- | --- |\n" + "| x | y |\n" * 25
    code = '```python\n' + '# comment\n' * 40 + '```\n'
    doc, sections = parsed(tmp_path, '## Test\n' + ('Repeated prose with useful details. ' * 60) + '\n\n' + code + '\n' + table)
    chunks = Chunker(200, 30).split(sections[0])
    assert any(c.oversized and "code" in c.block_types for c in chunks)
    assert any(c.oversized and "table" in c.block_types for c in chunks)
    assert sum(code in c.content for c in chunks) == 1
    assert sum(table.strip() in c.content for c in chunks) == 1
    assert any(a.section_end > b.section_start for a, b in zip(chunks, chunks[1:]))
    assert not validate_records(doc, sections, chunks, 200)
    assert [c.model_dump_json() for c in chunks] == [c.model_dump_json() for c in Chunker(200, 30).split(sections[0])]


@pytest.mark.parametrize("relative", ['oss/langgraph/persistence.mdx', 'oss/langgraph/overview.mdx', 'oss/deepagents/subagents.mdx'])
def test_real_three_document_pipeline(tmp_path, relative):
    report = run_pipeline(ROOT / 'docs/src', tmp_path / 'out', paths=[relative])
    assert not report['failures'] and report['quality_errors'] == 0, report
    assert report['sections_generated'] > 0 and report['chunks_generated'] > 0
    assert len(list((tmp_path / 'out').iterdir())) == 4


def test_four_file_repeatability(tmp_path):
    root = tmp_path / 'docs/src'
    for product in ('langchain', 'langgraph', 'deepagents'):
        directory = root / 'oss' / product
        directory.mkdir(parents=True)
        (directory / 'page.mdx').write_text('---\ntitle: Page\n---\nIntro\n\n## Setup\n```python\n# example\n```\n', encoding='utf-8')
    for name in ('a', 'b'):
        report = run_pipeline(root, tmp_path / name)
        assert report['quality_errors'] == 0
    for name in ('documents.jsonl', 'sections.jsonl', 'chunks.jsonl', 'ingestion_report.json'):
        assert (tmp_path / 'a' / name).read_bytes() == (tmp_path / 'b' / name).read_bytes()


def test_blockquote_code_remains_atomic(tmp_path):
    code = '> ```python\n' + '> value = 1\n' * 30 + '> ```\n'
    # Feed clean Markdown directly: the chunker must protect nested code too.
    doc, sections = parsed(tmp_path, '## Quote\n' + code)
    chunks = Chunker(100, 10).split(sections[0])
    assert len(chunks) == 1 and chunks[0].oversized
    assert "code" in chunks[0].block_types


def test_expanded_snippet_source_survives_all_layers(tmp_path):
    (tmp_path / 'snippet.mdx').write_text('```python\n# from snippet\nvalue = 1\n```\n', encoding='utf-8')
    doc, sections = parsed(tmp_path, 'import Demo from "./snippet.mdx";\n## Setup\n<Demo />\n')
    chunks = [c for s in sections for c in Chunker().split(s)]
    code, = [c for c in chunks if '# from snippet' in c.content]
    assert any(span.source_path == 'snippet.mdx' for span in code.source_spans)
    assert code.dependencies[0].source_path == 'snippet.mdx'
    assert not validate_records(doc, sections, chunks, 800)
