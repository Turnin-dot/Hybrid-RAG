from pathlib import Path
import hashlib
import re
import shutil

import pytest
from pydantic import ValidationError
from markdown_it import MarkdownIt

from hybrid_rag.ingestion.cleaner import MDXCleaner
from hybrid_rag.schemas.document import Chunk, Document, RawDocument, Section


FIXTURES = Path(__file__).parents[1] / "fixtures" / "cleaner"
DOCS = Path(__file__).parents[2] / "docs" / "src"


@pytest.fixture
def root(tmp_path):
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    return tmp_path


def run(root, text, *, path="page.mdx", **options):
    cleaner = MDXCleaner(root, **options)
    return cleaner.clean(cleaner.from_text(text, source_path=path, language=cleaner.language))


def codes(result):
    return {warning.code for warning in result.warnings}


def assert_balanced(text):
    # Independent parser: do not verify cleaner behavior with its own lexer.
    lines = text.splitlines()
    for token in MarkdownIt().parse(text):
        if token.type == "fence":
            last = lines[token.map[1] - 1].strip()
            assert re.fullmatch(re.escape(token.markup[0]) + "{" + str(len(token.markup)) + ",}", last)


def test_frontmatter_and_no_invented_h1(root):
    cleaner = MDXCleaner(root)
    raw = cleaner.from_text((root / "page.mdx").read_text(), source_path="page.mdx")
    assert raw.frontmatter == {"title": "Fixture page", "description": "Plain metadata", "keywords": ["retrieval", "code"], "boost": 2}
    assert raw.body_start_line == 7
    result = cleaner.clean(raw)
    assert result.title == "Fixture page"
    assert result.frontmatter == raw.frontmatter
    assert "# Fixture page" not in result.content
    assert "##  Install" in result.content
    assert isinstance(result.to_document(), Document)


def test_code_tables_links_and_component_metadata(root):
    result = run(root, (root / "page.mdx").read_text())
    content = result.content
    assert '# This is not a heading\nimport os\ntext = "<Info> :::js </Info>"' in content
    assert "const excluded" not in content
    assert "**Extra details**" in content
    assert "Important **body** and [reference](/reference)." in content
    assert "[Guide](/guide)" in content
    assert "Card explanation." in content
    assert "| Field | Meaning |\n| --- | --- |\n| value | Keep this table |" in content
    assert "<CodeGroup>" not in content
    assert not result.warnings
    assert_balanced(content)


def test_loader_adapter_preserves_metadata_and_version(root):
    raw = RawDocument.from_loader({"source_path": "page.mdx", "body": "No heading\n", "metadata": {"title": "Loader title", "custom": [1, 2]}, "version": "abc"})
    result = MDXCleaner(root).clean(raw)
    assert result.content == "No heading\n"
    assert result.frontmatter["custom"] == [1, 2]
    assert result.repository_commit == "abc"
    with pytest.raises(ValueError, match="Conflicting"):
        RawDocument.from_loader({"source_path": "page.mdx", "body": "x", "content": "y"})


def test_nested_language_blocks_and_indented_list(root):
    text = "shared\n   :::python\nkeep\n:::js\nremove\n:::\nkeep again\n   :::\nafter\n"
    result = run(root, text)
    assert result.content == "shared\nkeep\nkeep again\nafter\n"
    assert "unclosed_directive" not in codes(result)


def test_directives_inside_fences_are_not_control_syntax(root):
    text = ":::python\n````python\n# heading\n:::js\nimport Something from '/secret.mdx';\n<Info>literal</Info>\n```\n````\n:::\n"
    result = run(root, text)
    assert result.content == text.removeprefix(":::python\n").removesuffix(":::\n")
    assert_balanced(result.content)


@pytest.mark.parametrize("text,code", [
    (":::js\nuncertain content\n", "unclosed_directive"),
    ("::: \nordinary content\n", "unmatched_directive"),
    (":::custom\nordinary content\n:::\n", "unsupported_directive"),
])
def test_malformed_or_unknown_directives_preserve_content(root, text, code):
    result = run(root, text)
    assert result.content == text
    assert code in codes(result)


def test_snippet_expansion_hash_and_source_map(root):
    result = run(root, (root / "page.mdx").read_text())
    assert 'return "snippet value"' in result.content
    assert "<Example />" not in result.content
    assert "import Example" not in result.content
    dep, = result.dependencies
    assert dep.source_path == "snippets/example.mdx"
    assert dep.content_hash == hashlib.sha256((root / dep.source_path).read_bytes()).hexdigest()
    assert dep.referenced_by == ["page.mdx"]
    assert result.source_spans[0].start == 0
    assert result.source_spans[-1].end == len(result.content)
    assert all(a.end == b.start for a, b in zip(result.source_spans, result.source_spans[1:]))
    code = "".join(result.content[s.start:s.end] for s in result.source_spans if s.source_path == dep.source_path)
    assert code == (root / dep.source_path).read_text()


def test_recursive_snippet_and_dependency_revision(root):
    (root / "snippets/a.mdx").write_text("import B from './b.mdx';\n<Aaa />\n<B />\n")
    (root / "snippets/b.mdx").write_text("Nested content\n")
    text = "import A from '/snippets/a.mdx';\n<A />\n"
    first = run(root, text)
    assert "Nested content" in first.content
    assert len(first.dependencies) == 2
    (root / "snippets/b.mdx").write_text("Updated content\n")
    second = run(root, text)
    assert first.doc_id == second.doc_id
    assert first.source_revision_id == second.source_revision_id
    assert first.revision_id != second.revision_id


def test_missing_unused_and_excluded_snippets(root):
    text = "import Missing from '/snippets/missing.mdx';\n<Missing />\n"
    result = run(root, text)
    assert "[unresolved snippet:" in result.content
    assert "missing_snippet" in codes(result)
    excluded = run(root, "import Missing from '/snippets/missing.mdx';\n:::js\n<Missing />\n:::\n")
    assert "missing_snippet" not in codes(excluded)
    assert not excluded.dependencies


def test_cycles_preserve_placeholder(root):
    (root / "snippets/a.mdx").write_text("import B from './b.mdx';\nA text\n<B />\n")
    (root / "snippets/b.mdx").write_text("import A from './a.mdx';\nB text\n<A />\n")
    result = run(root, "import A from '/snippets/a.mdx';\n<A />\n")
    assert "snippet_cycle" in codes(result)
    assert "A text" in result.content and "B text" in result.content
    assert "[unresolved snippet:" in result.content


@pytest.mark.parametrize("reference", ["../../secret.mdx", "/../secret.mdx", "C:/secret.mdx", "https://example.com/file.mdx", "//server/file.mdx", "/snippets/data.py"])
def test_untrusted_paths(root, reference):
    result = run(root, f"import Unsafe from '{reference}';\n<Unsafe />\n")
    assert "unsafe_snippet_path" in codes(result)
    assert "[unresolved snippet:" in result.content
    assert not result.dependencies


def test_depth_and_expansion_limits(root):
    text = "import Demo from '/snippets/example.mdx';\n<Demo />\n<Demo />\n"
    result = run(root, text, max_expansions=1)
    assert "snippet_limit" in codes(result)
    assert "snippet value" in result.content
    assert "snippet_limit" in codes(run(root, text, max_depth=1))
    assert "snippet_limit" in codes(run(root, text, max_snippet_bytes=1))


def test_unknown_dynamic_and_inline_code_are_not_lost(root):
    text = 'import Demo from "/snippets/example.mdx";\n<Unknown value={x}>Keep me</Unknown>\n<Demo value={x} />\n`<Icon />` and `<Demo />`\n'
    result = run(root, text)
    assert "[unresolved component: Unknown] <Unknown value={x}>Keep me</Unknown>" in result.content
    assert "`<Icon />` and `<Demo />`" in result.content
    assert "dynamic_snippet" in codes(result)
    assert "unsupported_component" in codes(result)
    assert not result.dependencies


def test_multiline_components_and_static_links(root):
    text = '<Card\n title="Multi-line guide" description="Useful description"\n href="/guide?limit=2>1"\n>\nBody\n</Card>\n'
    result = run(root, text)
    assert "[Multi-line guide](/guide?limit=2>1)" in result.content
    assert "Body" in result.content
    assert "Useful description" in result.content
    assert not {"unsupported_component", "mismatched_component", "malformed_tag"} & codes(result)
    assert any(span.source_line_start == 1 and span.source_line_end == 4 for span in result.source_spans)


def test_multiline_inline_code_and_longer_ticks(root):
    text = "``literal ` and\n<Icon /> and <Unknown />``\n<Icon />\n"
    result = run(root, text)
    assert result.content == "``literal ` and\n<Icon /> and <Unknown />``\n\n"
    assert "unsupported_component" not in codes(result)


@pytest.mark.parametrize("token", ["<Card title={title}>", "<Card href={target}>", "<Card {...props}>"])
def test_dynamic_component_metadata_is_preserved(root, token):
    result = run(root, token + "Keep body</Card>\n")
    assert token in result.content
    assert "Keep body" in result.content
    assert "dynamic_component" in codes(result)


def test_nested_dynamic_snippet_props_not_executed(root):
    result = run(root, "import Demo from '/snippets/example.mdx';\n<Demo data={{value: 1}} />\n")
    assert "dynamic_snippet" in codes(result)
    assert not result.dependencies
    assert "data={{value: 1}}" in result.content


def test_unknown_import_and_exports_are_not_discarded(root):
    text = "import { Component } from '/snippets/example.mdx';\nexport const value = 1;\n<Component />\n"
    result = run(root, text)
    assert result.content == text
    assert {"unsupported_import", "unsupported_export", "unsupported_component"} <= codes(result)


def test_unclosed_comments_and_bad_component_closure(root):
    result = run(root, "<Info>body</Card>\n{/* never closed\nvaluable text\n")
    assert "valuable text" in result.content
    assert "</Card>" in result.content
    assert {"unclosed_comment", "mismatched_component", "unclosed_component"} <= codes(result)


def test_indented_snippet_retains_list_context(root):
    result = run(root, "import Demo from '/snippets/example.mdx';\n1. Example:\n\n   <Demo />\n")
    assert "   ```python\n   def example():" in result.content
    assert_balanced(result.content)


def test_layout_indent_is_not_mistaken_for_markdown_code(root):
    (root / "snippets/group.mdx").write_text('<CodeGroup>\n    ```python\n    def demo():\n        return 1\n    ```\n</CodeGroup>\n')
    result = run(root, 'import Group from "/snippets/group.mdx";\n<Columns>\n    <Card title="Guide" href="/guide">\n        Useful prose\n    </Card>\n</Columns>\n<Group />\n')
    tokens = MarkdownIt().parse(result.content)
    assert not any(token.type == "code_block" for token in tokens)
    fence, = [token for token in tokens if token.type == "fence"]
    assert fence.content == "def demo():\n    return 1\n"
    assert "[Guide](/guide)" in result.content
    assert_balanced(result.content)


def test_code_string_at_column_zero_does_not_break_wrapper_layout(root):
    text = '<Tabs>\n  <Tab title="Store">\n    Description\n\n    ```python\n    def demo():\n        value = """first\nzero column text\n    last"""\n        return value\n    ```\n  </Tab>\n</Tabs>\n'
    result = run(root, text)
    tokens = MarkdownIt().parse(result.content)
    assert not any(token.type == "code_block" for token in tokens)
    code, = [token.content for token in tokens if token.type == "fence"]
    assert code == 'def demo():\n    value = """first\nzero column text\nlast"""\n    return value\n'
    assert "Description" in result.content
    assert_balanced(result.content)


def test_code_fence_closing_margin_and_unknown_wrapper(root):
    result = run(root, '<UnknownField name="value">\n    Description\n\n      ```python\n      def demo():\n          return 1\n    ```\n</UnknownField>\n')
    tokens = MarkdownIt().parse(result.content)
    code, = [token for token in tokens if token.type == "fence"]
    assert code.content == "def demo():\n    return 1\n"
    assert "unsupported_component" in codes(result)
    assert_balanced(result.content)


def test_code_nested_in_list_preserves_list_and_code_indent(root):
    text = '1. Example:\n\n   <Tabs>\n       <Tab title="Sync">\n         ```python\n         def demo():\n             return 1\n       ```\n       </Tab>\n   </Tabs>\n'
    result = run(root, text)
    tokens = MarkdownIt().parse(result.content)
    types = [token.type for token in tokens]
    assert types.index("list_item_open") < types.index("fence") < types.index("list_item_close")
    code, = [token.content for token in tokens if token.type == "fence"]
    assert code == "def demo():\n    return 1\n"
    assert_balanced(result.content)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_official_adjacent_close_open_and_close_with_text(root, newline):
    text = 'shared\n:::python\npython body\n::::::js\njavascript body\n:::remaining shared text\nend\n'.replace("\n", newline)
    result = run(root, text)
    assert result.content == "shared\npython body\nremaining shared text\nend\n"
    assert not {"unclosed_directive", "unmatched_directive"} & codes(result)


def test_directive_variants_inside_fence_remain_literal(root):
    text = ':::python\n```python\n# :::js\n::::::js\n:::literal\n```\n:::\n'
    result = run(root, text)
    assert result.content == '```python\n# :::js\n::::::js\n:::literal\n```\n'


def test_escaped_directives_and_close_table_row(root):
    result = run(root, '\\:::python\nliteral\n\\:::\n:::python\n| value | keep |\n::: | shared | row |\n')
    assert result.content == '\\:::python\nliteral\n\\:::\n| value | keep |\n | shared | row |\n'


@pytest.mark.parametrize("noindex", ["true", "false"])
def test_redirect_classification_uses_empty_original_body_and_url(root, noindex):
    result = run(root, f'---\ntitle: Link\nurl: /oss/langchain/mcp\nnoindex: {noindex}\n---\n')
    assert result.document_kind == "redirect"
    assert result.redirect_url == "/oss/langchain/mcp"
    assert result.content == ""
    assert result.to_document().document_kind == "redirect"


def test_noindex_or_url_does_not_skip_real_body(root):
    result = run(root, '---\ntitle: Content\nurl: /somewhere\nnoindex: true\n---\nKeep this body\n')
    assert result.document_kind == "content"
    assert result.redirect_url is None
    assert "Keep this body" in result.content


def test_field_prompt_and_literal_array_attributes(root):
    result = run(root, '<ParamField body="model" type="string" required>Choose model.</ParamField>\n<ResponseField name="tokens" type="int" post={["optional"]}>Count.</ResponseField>\n<Prompt description="Configure" actions={["copy"]}>Keep prompt.</Prompt>\n')
    assert "**`model`** (string; required)" in result.content
    assert "**`tokens`** (int; optional)" in result.content
    assert "Keep prompt." in result.content and "Configure" in result.content
    assert "<ParamField" not in result.content and "<Prompt" not in result.content


def test_unmatched_backtick_does_not_hide_following_components(root):
    result = run(root, 'A stray `\n\n<Info>Important.</Info>\n')
    assert "<Info>" not in result.content
    assert "Important." in result.content
    assert "unmatched_backtick" in codes(result)


def test_symlink_cannot_escape_root(root, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.mdx")
    outside.write_text("MUST NOT BE READ")
    link = root / "snippets/escape.mdx"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit unprivileged symlink creation")
    result = run(root, "import Escape from '/snippets/escape.mdx';\n<Escape />\n")
    assert "unsafe_snippet_path" in codes(result)
    assert "MUST NOT BE READ" not in result.content


def test_javascript_target_and_language_mismatch(root):
    result = run(root, "common\n:::python\npython only\n:::\n:::js\njs only\n:::\n", target_language="javascript")
    assert result.content == "common\njs only\n"
    with pytest.raises(ValueError, match="language"):
        MDXCleaner(root).clean(RawDocument(source_path="page.mdx", content="body", language="javascript"))


def test_unclosed_fence_is_reported_not_repaired(root):
    result = run(root, "```python\n# unfinished\n<Info>literal\n")
    assert result.content == "```python\n# unfinished\n<Info>literal\n"
    assert "unclosed_code_fence" in codes(result)


def test_invalid_frontmatter_is_preserved_and_reported(root):
    text = "---\ntitle: [\n---\nbody\n"
    result = run(root, text)
    assert result.content == text
    assert "invalid_frontmatter" in codes(result)


def test_json_roundtrip_and_repeatability(root):
    cleaner = MDXCleaner(root)
    raw = cleaner.from_text((root / "page.mdx").read_text(), source_path="page.mdx")
    first = cleaner.clean(raw)
    second = cleaner.clean(raw)
    assert first.model_dump_json() == second.model_dump_json()
    assert Document.model_validate_json(first.to_document().model_dump_json()) == first.to_document()


def test_crlf_and_lf_have_identical_language_selection(root):
    text = (root / "page.mdx").read_text()
    lf = run(root, text)
    crlf = run(root, text.replace("\n", "\r\n"))
    assert lf.content == crlf.content
    assert lf.dependencies == crlf.dependencies
    assert lf.warnings == crlf.warnings


def test_schema_hierarchy_and_mixed_chunk():
    shared = dict(doc_id="d", revision_id="r", source_path="page.mdx", heading_path=["Page", "Setup"])
    section = Section(section_id="s", parent_section_id="parent", heading_level=2, section_index=0, content="", **shared)
    assert section.parent_section_id == "parent"
    chunk = Chunk(chunk_id="c", section_id="s", chunk_index=0, content="text and code", embedding_text="Page\ntext and code", block_types=["paragraph", "code"], **shared)
    assert chunk.block_types == ["paragraph", "code"]
    with pytest.raises(ValidationError):
        Chunk(**{**chunk.model_dump(), "content": ""})


@pytest.mark.parametrize("relative", ["oss/langgraph/persistence.mdx", "oss/langgraph/overview.mdx", "oss/deepagents/subagents.mdx"])
def test_real_document_regression(relative):
    assert (DOCS / relative).is_file(), "Real-document corpus must be available"
    text = (DOCS / relative).read_bytes().decode("utf-8-sig")
    cleaner = MDXCleaner(DOCS)
    raw = cleaner.from_text(text, source_path=relative)
    result = cleaner.clean(raw)
    assert result.model_dump_json() == cleaner.clean(raw).model_dump_json()
    assert result.frontmatter == raw.frontmatter
    assert not result.warnings, [w.model_dump() for w in result.warnings]
    assert_balanced(result.content)
    assert ":::python" not in result.content and ":::js" not in result.content
    if relative.endswith("persistence.mdx"):
        assert result.title == "Persistence"
        assert "# Persistence\n" not in result.content
        assert "| Persists | Graph state snapshots |" in result.content
        assert "# Consider adding a cron job to delete checkpoints older than N days" in result.content
        assert 'import { MemorySaver' not in result.content
        assert "Agent Server handles persistence automatically" in result.content
    elif relative.endswith("overview.mdx"):
        assert "##  Install" in result.content
        assert "pip install -U langgraph" in result.content
        assert "npm install" not in result.content
        assert "[LangChain](/oss/langchain/overview)" in result.content
        assert "Deploy and scale stateful agents" in result.content
        assert not any(token.type == "code_block" for token in MarkdownIt().parse(result.content))
    else:
        assert "def internet_search(" in result.content
        assert "# Optional override, defaults to main agent model" in result.content
        assert len(result.dependencies) >= 20
        assert all(not d.source_path.endswith("-js.mdx") for d in result.dependencies)
        assert "<SubagentBasicPy />" not in result.content
        assert "| `name` | `str` |" in result.content
