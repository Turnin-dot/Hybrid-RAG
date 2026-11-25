"""Shared cleaning audit and pipeline integrity validation."""
from pathlib import Path
from collections import Counter
from bisect import bisect_right
import hashlib
import re
from markdown_it import MarkdownIt


def validate_records(document, sections, chunks, chunk_size):
    """Check links, exact slices, non-whitespace coverage and atomic structures."""
    errors = []
    by_id = {s.section_id: s for s in sections}
    if len(by_id) != len(sections) or len({c.chunk_id for c in chunks}) != len(chunks):
        errors.append("duplicate_section_or_chunk_id")
    for section in sections:
        if section.doc_id != document.doc_id or section.revision_id != document.revision_id:
            errors.append("section_document_identity")
        if document.content[section.content_start:section.content_end] != section.content:
            errors.append("section_source_slice")
        if section.parent_section_id:
            parent = by_id.get(section.parent_section_id)
            if parent is None or parent.heading_level >= section.heading_level:
                errors.append("section_parent")
        local = [c for c in chunks if c.section_id == section.section_id]
        covered = bytearray(len(section.content))
        for chunk in local:
            a, b = chunk.section_start, chunk.section_end
            if chunk.content_parts:
                position = 0
                for part in chunk.content_parts:
                    if part.start != position or chunk.content[part.start:part.end] != section.content[part.section_start:part.section_end]:
                        errors.append("chunk_part_source_slice")
                    position = part.end
                    covered[part.section_start:part.section_end] = b"\x01" * (part.section_end-part.section_start)
                if position != len(chunk.content):
                    errors.append("chunk_part_coverage")
            elif chunk.content != section.content[a:b]:
                errors.append("chunk_source_slice_or_empty")
            else:
                covered[a:b] = b"\x01" * (b - a)
            if not chunk.content.strip():
                errors.append("chunk_source_slice_or_empty")
            if chunk.doc_id != document.doc_id or chunk.revision_id != document.revision_id:
                errors.append("chunk_document_identity")
            if chunk.oversized != (len(chunk.content) > chunk_size):
                errors.append("oversized_flag")
            cursor = 0
            for span in chunk.source_spans:
                if span.start != cursor or span.end > len(chunk.content):
                    errors.append("chunk_source_map")
                cursor = span.end
            if cursor != len(chunk.content):
                errors.append("chunk_source_map_coverage")
        if any(not covered[i] and not char.isspace() for i, char in enumerate(section.content)):
            errors.append("chunk_content_lost")
        offsets = [0]
        for line in section.content.splitlines(True):
            offsets.append(offsets[-1] + len(line))
        for token in MarkdownIt().enable("table").parse(section.content):
            if token.type in {"fence", "table_open"} and token.map:
                a, b = offsets[token.map[0]], offsets[token.map[1]]
                method = "python_ast" if token.type == "fence" else "table_rows"
                if any(c.split_method and c.split_method.startswith('token_') and any(p.section_start < b and p.section_end > a for p in c.content_parts) for c in local):
                    from .structured_validation import validate_token_structure
                    errors.extend(validate_token_structure(section, local, token, a, b))
                elif any(c.split_method == method and any(p.section_start < b and p.section_end > a for p in c.content_parts) for c in local):
                    from .structured_validation import validate_structure
                    errors.extend(validate_structure(section, local, token, a, b))
                elif sum(any(p.role == 'body' and p.section_start <= a and p.section_end >= b for p in c.content_parts)
                         if c.content_parts else c.section_start <= a and c.section_end >= b for c in local) != 1:
                    errors.append("code_or_table_split_or_duplicated")
            if token.type == 'list_item_open' and token.map and any(c.split_method in {'list_items','token_list_items'} for c in local):
                a, b = offsets[token.map[0]], offsets[token.map[1]]
                if any(c.split_method=='token_source_continuation' and any(p.section_start<b and p.section_end>a for p in c.content_parts) for c in local):continue
                if sum(any(p.role == 'body' and p.section_start <= a and p.section_end >= b for p in c.content_parts)
                       if c.content_parts else c.section_start <= a and c.section_end >= b for c in local) != 1:
                    errors.append('list_item_split_or_duplicated')
        # Repeated wrappers/headers are explicit; body characters must not be
        # duplicated by structural splitting (ordinary baseline prose overlap is allowed).
        body_counts = Counter()
        for chunk in local:
            for part in chunk.content_parts:
                if part.role == "body":
                    body_counts.update(range(part.section_start, part.section_end))
        if any(value > 1 for value in body_counts.values()):
            errors.append("duplicated_structural_body")
    if any(chunk.section_id not in by_id for chunk in chunks):
        errors.append("unknown_section")
    # All document body characters outside actual Markdown headings must be in sections.
    covered = bytearray(len(document.content))
    for section in sections:
        covered[section.content_start:section.content_end] = b"\x01" * len(section.content)
    offsets = [0]
    for line in document.content.splitlines(True):
        offsets.append(offsets[-1] + len(line))
    for token in MarkdownIt().parse(document.content):
        if token.type == "heading_open":
            a, b = offsets[token.map[0]], offsets[token.map[1]]
            covered[a:b] = b"\x01" * (b - a)
    if any(not covered[i] and not char.isspace() for i, char in enumerate(document.content)):
        errors.append("section_content_lost")
    return sorted(set(errors))

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def trusted_path(root: Path, relative: str) -> Path:
    """Validate before reading, including symlinks and cross-platform paths."""
    if "\\" in relative or relative.startswith("/") or re.match(r"^[A-Za-z][\w+.-]*:", relative):
        raise ValueError(f"Not a root-relative source path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Source path escapes docs root: {relative}")
    return path


def source_fences(text: str, start_line: int = 1) -> dict[int, str]:
    """Extract source fence bodies for comparison, independent of the Cleaner.

    Source JSX permits deeper fence margins than Markdown. Remove that margin
    from code text for comparison, including short/blank lines, without
    removing the code's relative indentation.
    """
    blocks = {}
    active = None
    body = []
    for number, line in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").splitlines(True), start_line):
        match = re.match(r"^([ \t]*)(`{3,}|~{3,})(.*?)(?:\n)?$", line)
        if active is None:
            if match:
                active = (number, match[2], len(match[1]))
                body = []
        elif match and match[2][0] == active[1][0] and len(match[2]) >= len(active[1]) and not match[3].strip():
            blocks[active[0]] = "".join(body)
            active = None
        else:
            leading = len(line) - len(line.lstrip(" \t"))
            body.append(line[min(leading, active[2]):])
    return blocks


def audit_document(raw, result, root: Path) -> tuple[list[dict], dict]:
    """Independent structural checks plus explicitly labelled review heuristics."""
    findings = []

    def add(check: str, severity: str, detail: str, **evidence):
        findings.append({"check": check, "severity": severity, "detail": detail, **evidence})

    text = result.content
    before = raw.content.replace("\r\n", "\n").replace("\r", "\n")
    redirect = result.document_kind == "redirect"
    if redirect and (raw.content.strip() or not result.redirect_url or result.redirect_url != raw.frontmatter.get("url") or text.strip()):
        add("invalid_redirect_classification", "error", "Only originally empty pages with a preserved URL may be skipped")
    if not text.strip() and not redirect:
        add("empty_content", "error", "Cleaned body is empty or whitespace-only")
    if raw.frontmatter != result.frontmatter:
        add("frontmatter_changed", "error", "Cleaner changed extracted frontmatter")
    title = raw.frontmatter.get("title")
    if isinstance(title, str) and title.strip() and result.title != title:
        add("title_changed", "error", "Cleaner did not preserve the static page title")
    if result.source_path != raw.source_path or result.doc_id != raw.doc_id:
        add("identity_changed", "error", "Source path or document ID changed")

    sources = {raw.source_path}
    for dependency in result.dependencies:
        sources.add(dependency.source_path)
        if re.search(r"-(?:js|javascript)\.mdx?$", dependency.source_path):
            add("javascript_dependency_in_python_output", "review", "Check whether a JS-specific snippet belongs in the Python corpus", dependency=dependency.source_path)
        try:
            data = trusted_path(root, dependency.source_path).read_bytes()
            if digest(data) != dependency.content_hash:
                add("dependency_hash_mismatch", "error", "Dependency bytes differ from recorded hash", dependency=dependency.source_path)
        except (OSError, ValueError) as exc:
            add("invalid_dependency", "error", str(exc), dependency=dependency.source_path)

    source_lengths = {}
    original_blocks = {}
    for relative in sorted(sources):
        try:
            original_text = trusted_path(root, relative).read_text(encoding="utf-8-sig")
            source_lengths[relative] = len(original_text.splitlines())
            original_blocks[relative] = source_fences(raw.content, raw.body_start_line) if relative == raw.source_path else source_fences(original_text)
        except (OSError, ValueError, UnicodeError) as exc:
            add("invalid_source_file", "error", str(exc), source=relative)
    cursor = 0
    for index, span in enumerate(result.source_spans):
        if span.start != cursor or span.end <= span.start or span.end > len(text):
            add("invalid_source_offsets", "error", "Source spans must cover content contiguously", span_index=index)
        cursor = span.end
        if span.source_path not in source_lengths:
            add("unknown_span_source", "error", "Span points outside the page/dependency set", span_index=index, source=span.source_path)
        elif not 1 <= span.source_line_start <= span.source_line_end <= source_lengths[span.source_path]:
            add("invalid_source_lines", "error", "Source line range is out of bounds", span_index=index, source=span.source_path)
    if cursor != len(text):
        add("incomplete_source_map", "error", "Source spans do not cover the full cleaned body")

    # markdown-it is independent of the cleaner's internal fence lexer.
    tokens = MarkdownIt().parse(text)
    lines = text.splitlines()
    code_lines = set()
    fences = 0
    line_offsets = [0]
    for line in text.splitlines(True):
        line_offsets.append(line_offsets[-1] + len(line))
    span_ends = [span.end for span in result.source_spans]
    emitted = Counter()
    recognized = Counter()
    for span in result.source_spans:
        if span.source_line_start == span.source_line_end and span.source_line_start in original_blocks.get(span.source_path, {}):
            if re.match(r"^[ \t]*(?:`{3,}|~{3,})", text[span.start:span.end]):
                emitted[(span.source_path, span.source_line_start)] += 1
    for token in tokens:
        if token.type not in {"fence", "code_block"} or token.map is None:
            continue
        code_lines.update(range(token.map[0], token.map[1]))
        if token.type == "fence":
            fences += 1
            if result.source_spans:
                opener = lines[token.map[0]]
                position = line_offsets[token.map[0]] + max(0, opener.find(token.markup))
                span = result.source_spans[min(bisect_right(span_ends, position), len(span_ends) - 1)]
                source_key = (span.source_path, span.source_line_start)
                expected = original_blocks.get(source_key[0], {}).get(source_key[1])
                if expected is None:
                    add("mispaired_output_fence", "error", "Output fence does not map to a source opening fence", output_line=token.map[0] + 1, source=source_key[0], source_line=source_key[1])
                else:
                    recognized[source_key] += 1
                    if expected.strip("\n") != token.content.strip("\n"):
                        add("code_content_changed", "error", "Parsed code differs from source after removing layout margin", output_line=token.map[0] + 1, source=source_key[0], source_line=source_key[1])
            language = token.info.split()[0].lower() if token.info.split() else ""
            if language in {"javascript", "js", "typescript", "ts", "jsx", "tsx"}:
                add("javascript_fence_in_python_output", "review", "May be shared content or language leakage; inspect before accepting", output_line=token.map[0] + 1, language=language)
            last = lines[token.map[1] - 1].lstrip()
            # List/blockquote containers may precede closing fences.
            last = re.sub(r"^(?:>\s*)+", "", last).strip()
            pattern = re.escape(token.markup[0]) + "{" + str(len(token.markup)) + ",}"
            if not re.fullmatch(pattern, last):
                add("unclosed_output_fence", "error", "Markdown parser found a fence without a closing marker", output_line=token.map[0] + 1)
        elif any(re.match(r"\s*(?:`{3,}|~{3,})", line) for line in token.content.splitlines()):
            add("possibly_indented_fence", "review", "A fence may have become an indented code block", output_line=token.map[0] + 1)

    missing = emitted - recognized
    if missing:
        add("code_fence_not_recognized", "error", "Emitted source fences were not parsed as their corresponding code blocks", occurrences=sum(missing.values()),
            examples=[{"source": key[0], "source_line": key[1], "count": count} for key, count in sorted(missing.items())[:12]])

    # These are review clues, not proof of loss, leakage, or corruption.
    residuals = Counter()
    examples = []
    for index, line in enumerate(lines):
        if index in code_lines:
            continue
        # Avoid flagging simple inline-code examples as real component syntax.
        visible = re.sub(r"(`+).*?\1", "", line)
        for check, pattern in (
            ("residual_directive", r"^\s*:::(?:python|js|javascript)\s*$"),
            ("residual_import_export", r"^\s*(?:import|export)\b"),
            ("residual_jsx", r"</?[A-Z][\w.:-]*(?=[\s/>])"),
            ("unresolved_placeholder", r"\[unresolved (?:snippet|component)"),
        ):
            if re.search(pattern, visible):
                residuals[check] += 1
                if len(examples) < 12:
                    examples.append({"check": check, "output_line": index + 1, "excerpt": line[:200]})
    for check, count in residuals.items():
        add(check, "review", "Potential residual syntax; inspect source and cleaned output", occurrences=count,
            examples=[item for item in examples if item["check"] == check])
    ratio = len(text) / len(before) if before else None
    if ratio is not None and (ratio < 0.2 or ratio > 5):
        add("unusual_length_ratio", "review", "Large length change may be valid after language filtering or snippet expansion", ratio=round(ratio, 4), dependencies=len(result.dependencies))
    return findings, {"body_chars_before": len(before), "body_chars_after": len(text),
                      "length_ratio": ratio, "output_fences": fences,
                      "source_fences_emitted": sum(emitted.values()), "source_fences_recognized": sum(recognized.values()),
                      "dependencies": len(result.dependencies), "source_spans": len(result.source_spans)}
