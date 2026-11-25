"""Test the batch harness with a tiny synthetic corpus, not the user's full run."""

import importlib.util
import json
from pathlib import Path

import pytest

from hybrid_rag.ingestion.cleaner import MDXCleaner
from hybrid_rag.schemas.document import SourceSpan

SCRIPT = Path(__file__).parents[2] / "scripts/test_full_cleaning.py"
spec = importlib.util.spec_from_file_location("full_cleaning_harness", SCRIPT)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "docs/src"
    for product in harness.PRODUCTS:
        (root / "oss" / product).mkdir(parents=True)
    (root / "snippets").mkdir()
    (root / "oss/langchain/page.mdx").write_text('---\ntitle: Chain\n---\n<Unknown>Keep content</Unknown>\n', encoding="utf-8")
    (root / "oss/langgraph/page.mdx").write_bytes(b'---\r\ntitle: Graph\r\n---\r\n:::python\r\n```python\r\n# keep\r\nx = 1\r\n```\r\n:::\r\n:::js\r\nremove this\r\n:::\r\n')
    (root / "oss/langgraph/extra.md").write_text('---\ntitle: Extra\n---\nAdditional content\n', encoding="utf-8")
    (root / "oss/deepagents/page.mdx").write_text('---\ntitle: Deep\n---\nimport Example from "/snippets/example.mdx";\n<Example />\n', encoding="utf-8")
    (root / "snippets/example.mdx").write_text('```python\n# snippet comment\nvalue = 42\n```\n', encoding="utf-8")
    return root


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def report(path):
    return json.loads((path / "ingestion_report.json").read_text(encoding="utf-8"))


def test_default_run_exports_all_evidence(corpus, tmp_path, capsys):
    before = {path: path.read_bytes() for path in corpus.rglob("*") if path.is_file()}
    out = tmp_path / "output"
    code = harness.run_batch(corpus, out, list(harness.PRODUCTS), quiet=True)
    assert code == 0
    result = report(out)
    assert result["documents_discovered"] == 3
    assert result["documents_completed"] == 3
    assert result["documents_failed"] == result["quality_errors"] == 0
    assert result["status"] == "review_required"
    assert result["manual_review_status"] == "pending"
    assert result["documents_with_warnings"] == 1
    assert len(jsonl(out / "documents.jsonl")) == 3
    assert len(jsonl(out / "manifest.jsonl")) == 3
    assert len(jsonl(out / "review_queue.jsonl")) == 3
    assert len(jsonl(out / "warnings.jsonl")) >= 1
    assert not jsonl(out / "failures.jsonl")
    assert (out / "before/snippets/example.mdx").read_bytes() == (corpus / "snippets/example.mdx").read_bytes()
    graph = (out / "after/oss/langgraph/page.mdx.cleaned.md").read_text(encoding="utf-8")
    assert "# keep" in graph and "remove this" not in graph
    assert all(path.read_bytes() == data for path, data in before.items())
    rows = jsonl(out / "manifest.jsonl")
    assert all(row["repeatable"] for row in rows)


def test_include_md_is_explicit(corpus, tmp_path, capsys):
    out = tmp_path / "with-md"
    harness.run_batch(corpus, out, list(harness.PRODUCTS), include_md=True, quiet=True)
    assert report(out)["documents_discovered"] == 4
    assert (out / "after/oss/langgraph/extra.md.cleaned.md").exists()


def test_selected_product_and_clean_status(corpus, tmp_path, capsys):
    out = tmp_path / "graph-only"
    code = harness.run_batch(corpus, out, ["langgraph", "langgraph"], quiet=True)
    assert code == 0
    result = report(out)
    assert result["documents_discovered"] == 1
    assert result["status"] == "automated_checks_passed"
    assert result["manual_review_status"] == "pending"


def test_warnings_can_fail_strict_mode(corpus, tmp_path, capsys):
    assert harness.run_batch(corpus, tmp_path / "strict", ["langchain"], strict_warnings=True, quiet=True) == 2


def test_bad_file_does_not_stop_remaining_documents(corpus, tmp_path, capsys):
    (corpus / "oss/langchain/broken.mdx").write_bytes(b"\xff\xfe\xfa")
    out = tmp_path / "failure"
    assert harness.run_batch(corpus, out, list(harness.PRODUCTS), quiet=True) == 1
    result = report(out)
    assert result["documents_failed"] == 1
    assert result["documents_completed"] == 3
    assert result["status"] == "failed"
    failure, = jsonl(out / "failures.jsonl")
    assert failure["error_type"] == "UnicodeDecodeError"
    assert failure["traceback"]


def test_existing_output_is_never_overwritten(corpus, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        harness.run_batch(corpus, out, ["langgraph"], quiet=True)
    assert (out / "sentinel").read_text() == "keep"


def test_output_cannot_be_in_docs(corpus):
    with pytest.raises(ValueError, match="overlap"):
        harness.run_batch(corpus, corpus.parent / "output", ["langgraph"], quiet=True)


def test_empty_or_missing_scope_does_not_pass(corpus, tmp_path):
    (corpus / "oss/langchain/page.mdx").unlink()
    with pytest.raises(ValueError, match="No documents"):
        harness.run_batch(corpus, tmp_path / "empty", ["langchain"], quiet=True)
    with pytest.raises(ValueError, match="Missing product"):
        harness.discover(corpus, ["nonexistent"], False)


def test_audit_detects_broken_offsets_and_dependency_hash(corpus):
    cleaner = MDXCleaner(corpus)
    relative = "oss/deepagents/page.mdx"
    raw = cleaner.from_text((corpus / relative).read_text(), source_path=relative)
    result = cleaner.clean(raw)
    (corpus / "snippets/example.mdx").write_text("changed\n")
    result.source_spans = [SourceSpan(start=0, end=len(result.content) + 1, source_path=relative, source_line_start=1, source_line_end=1)]
    findings, _ = harness.audit_document(raw, result, corpus)
    checks = {finding["check"] for finding in findings}
    assert {"invalid_source_offsets", "incomplete_source_map", "dependency_hash_mismatch"} <= checks


def test_audit_does_not_treat_code_literals_as_residuals(corpus):
    cleaner = MDXCleaner(corpus)
    raw = cleaner.from_text('---\ntitle: Literals\n---\n```python\n# comment\ns = "<Unknown />"\nimport os\n```\n', source_path="oss/langgraph/page.mdx")
    result = cleaner.clean(raw)
    findings, _ = harness.audit_document(raw, result, corpus)
    assert not {"residual_jsx", "residual_import_export", "unclosed_output_fence"} & {f["check"] for f in findings}


def test_non_repeatable_result_sets_failure_status(corpus, tmp_path, monkeypatch, capsys):
    original = MDXCleaner.clean
    calls = 0

    def unstable(self, raw):
        nonlocal calls
        calls += 1
        result = original(self, raw)
        if calls % 2 == 0:
            result.title += " changed"
        return result

    monkeypatch.setattr(MDXCleaner, "clean", unstable)
    out = tmp_path / "unstable"
    assert harness.run_batch(corpus, out, ["langgraph"], quiet=True) == 1
    assert report(out)["quality_check_counts"]["not_repeatable"] == 1


def test_js_code_is_a_review_clue_not_automatic_rejection(corpus):
    cleaner = MDXCleaner(corpus)
    raw = cleaner.from_text('---\ntitle: Shared\n---\n```typescript\nconst value = 1;\n```\n', source_path="oss/langgraph/page.mdx")
    result = cleaner.clean(raw)
    findings, _ = harness.audit_document(raw, result, corpus)
    clue, = [item for item in findings if item["check"] == "javascript_fence_in_python_output"]
    assert clue["severity"] == "review"


def test_cli_accepts_user_options(corpus, tmp_path, monkeypatch, capsys):
    out = tmp_path / "cli"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--docs-root", str(corpus), "--output", str(out),
                                    "--products", "langgraph", "--include-md", "--quiet"])
    assert harness.main() == 0
    assert report(out)["documents_discovered"] == 2


def test_redirect_is_recorded_and_excluded_from_search_documents(corpus, tmp_path, capsys):
    (corpus / "oss/langgraph/link.mdx").write_text('---\ntitle: Link\nurl: /oss/langchain/mcp\nnoindex: true\n---\n', encoding="utf-8")
    out = tmp_path / "redirect"
    assert harness.run_batch(corpus, out, ["langgraph"], quiet=True) == 0
    summary = report(out)
    assert summary["documents_discovered"] == summary["documents_processed"] == 2
    assert summary["documents_completed"] == summary["documents_skipped"] == 1
    assert summary["quality_errors"] == 0
    assert len(jsonl(out / "documents.jsonl")) == 1
    redirect, = jsonl(out / "redirects.jsonl")
    assert redirect["redirect_url"] == "/oss/langchain/mcp"
    row, = [row for row in jsonl(out / "manifest.jsonl") if row["status"] == "skipped"]
    assert row["skip_reason"] == "redirect_only"
    assert (out / row["before_file"]).exists()
    assert all(item["source_path"] != row["source_path"] for item in jsonl(out / "review_queue.jsonl"))


def test_genuinely_empty_page_still_fails(corpus, tmp_path, capsys):
    (corpus / "oss/langgraph/empty.mdx").write_text('---\ntitle: Empty\n---\n', encoding="utf-8")
    out = tmp_path / "empty-body"
    assert harness.run_batch(corpus, out, ["langgraph"], quiet=True) == 1
    assert report(out)["quality_check_counts"]["empty_content"] == 1
    assert report(out)["documents_skipped"] == 0


def test_audit_detects_balanced_but_indented_fences(corpus):
    relative = "oss/langgraph/layout.mdx"
    text = '---\ntitle: Code\n---\n    ```python\n    value = 1\n    ```\n'
    (corpus / relative).write_text(text, encoding="utf-8")
    cleaner = MDXCleaner(corpus)
    raw = cleaner.from_text(text, source_path=relative)
    result = cleaner.clean(raw)
    # Simulate the old output, with a complete, internally consistent map.
    result.content = raw.content
    result.source_spans = []
    offset = 0
    for number, line in enumerate(raw.content.splitlines(True), raw.body_start_line):
        result.source_spans.append(SourceSpan(start=offset, end=offset + len(line), source_path=relative, source_line_start=number, source_line_end=number))
        offset += len(line)
    findings, _ = harness.audit_document(raw, result, corpus)
    assert "code_fence_not_recognized" in {finding["check"] for finding in findings}


LAYOUT_REGRESSIONS = [
    "oss/deepagents/code/config-file.mdx", "oss/deepagents/code/configuration.mdx",
    "oss/deepagents/code/providers.mdx", "oss/langchain/context-engineering.mdx",
    "oss/langchain/human-in-the-loop.mdx", "oss/langchain/models.mdx",
    "oss/langgraph/add-memory.mdx", "oss/langgraph/sql-agent.mdx", "oss/langgraph/streaming.mdx",
]


@pytest.mark.parametrize("relative", LAYOUT_REGRESSIONS)
def test_real_failed_layout_regression(relative):
    root = SCRIPT.parents[1] / "docs/src"
    cleaner = MDXCleaner(root)
    raw = cleaner.from_text((root / relative).read_bytes().decode("utf-8-sig"), source_path=relative)
    result = cleaner.clean(raw)
    findings, metrics = harness.audit_document(raw, result, root)
    assert not [finding for finding in findings if finding["severity"] == "error"], findings
    assert metrics["source_fences_emitted"] == metrics["source_fences_recognized"]
    assert result.model_dump_json() == cleaner.clean(raw).model_dump_json()


@pytest.mark.parametrize("relative", ["oss/deepagents/event-streaming.mdx", "oss/deepagents/frontend/sandbox.mdx"])
def test_real_official_conditional_variants(relative):
    root = SCRIPT.parents[1] / "docs/src"
    cleaner = MDXCleaner(root)
    raw = cleaner.from_text((root / relative).read_bytes().decode("utf-8-sig"), source_path=relative)
    result = cleaner.clean(raw)
    assert not {"unclosed_directive", "unmatched_directive"} & {warning.code for warning in result.warnings}
    assert "::::::js" not in result.content
    if relative.endswith("event-streaming.mdx"):
        assert "| `tool_calls` |" in result.content
        assert "| `taskInput` |" not in result.content
    else:
        assert "get_or_create_sandbox_for_thread" in result.content
        assert ":::agent's backend:" not in result.content


@pytest.mark.parametrize("relative", [
    "oss/deepagents/a2a.mdx", "oss/deepagents/changelog-js.mdx", "oss/deepagents/changelog-py.mdx",
    "oss/deepagents/code/changelog.mdx", "oss/deepagents/mcp.mdx", "oss/deepagents/openwiki.mdx",
    "oss/langchain/academy.mdx", "oss/langchain/changelog-js.mdx", "oss/langchain/changelog-py.mdx",
    "oss/langchain/retrieval.mdx", "oss/langgraph/changelog-js.mdx", "oss/langgraph/changelog-py.mdx",
])
def test_real_redirect_pages_are_not_quality_failures(relative):
    root = SCRIPT.parents[1] / "docs/src"
    cleaner = MDXCleaner(root)
    raw = cleaner.from_text((root / relative).read_bytes().decode("utf-8-sig"), source_path=relative)
    result = cleaner.clean(raw)
    assert not raw.content.strip()
    assert result.document_kind == "redirect"
    assert result.redirect_url == raw.frontmatter["url"]
    findings, _ = harness.audit_document(raw, result, root)
    assert not findings
