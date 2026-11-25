"""Conservative static MDX cleaner. No JSX evaluation or code execution.

Fences are recognized before directives/imports/tags. Unsupported syntax stays
visible, with warnings. This is intentionally not a JavaScript/MDX compiler.
"""

from __future__ import annotations

import hashlib
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hybrid_rag.schemas.document import (
    CleanedDocument, CleaningWarning, RawDocument, SnippetDependency, SourceSpan,
    stable_hash,
)

CLEANER_VERSION = "mdx-static-v3"
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
_IMPORT = re.compile(r'''^\s*import\s+([\w$]+)\s+from\s+(['"])([^'"]+)\2\s*;?\s*$''')
_TAG_NAME = re.compile(r"</?([A-Za-z][\w.:-]*)(?=[\s/>])")
_ATTR = re.compile(r'''([\w:-]+)\s*=\s*("[^"]*"|'[^']*'|\{[^{}]*\})''', re.S)
_WRAPPERS = {
    "Info", "Note", "Tip", "Warning", "CodeGroup", "Expandable", "Card",
    "Columns", "CardGroup", "Tabs", "Tab", "Accordion", "AccordionGroup",
    "Steps", "Step", "Frame", "div", "span", "p", "section",
    "ParamField", "ResponseField", "Prompt", "Update", "Tooltip", "Callout",
    "Tree", "Tree.Folder", "Tree.File", "code", "ul", "li",
}
_DECORATIVE = {"Icon", "a"}


@dataclass
class _Line:
    text: str
    path: str
    number: int
    code: bool = False
    end_line: int | None = None
    fence_role: str | None = None


@dataclass
class _Context:
    warnings: list[CleaningWarning] = field(default_factory=list)
    dependencies: dict[str, SnippetDependency] = field(default_factory=dict)
    expansions: int = 0


def _mark_fences(lines: list[_Line]) -> bool:
    fence: tuple[str, int] | None = None
    for line in lines:
        match = _FENCE.match(line.text)
        line.code = fence is not None
        if match:
            marker, suffix = match.groups()
            if fence is None:
                fence = (marker[0], len(marker))
                line.code = True
                line.fence_role = "open"
            elif marker[0] == fence[0] and len(marker) >= fence[1] and not suffix.strip():
                fence = None
                line.fence_role = "close"
    return fence is None


class MDXCleaner:
    def __init__(self, docs_root: str | Path, *, target_language: str = "python",
                 max_depth: int = 20, max_expansions: int = 256,
                 max_snippet_bytes: int = 1_000_000):
        """docs_root is the trusted content root, normally docs/src."""
        self.root = Path(docs_root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("docs_root must be a directory")
        if target_language not in {"python", "javascript"}:
            raise ValueError("Unsupported target language")
        if min(max_depth, max_expansions, max_snippet_bytes) < 1:
            raise ValueError("Expansion limits must be positive")
        self.language = target_language
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.max_snippet_bytes = max_snippet_bytes

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _resolve(self, path: str, parent: Path | None = None) -> Path:
        # Reject URL schemes, UNC paths and Windows drive paths on every OS.
        if "\\" in path or re.match(r"^[A-Za-z][\w+.-]*:", path) or path.startswith("//"):
            raise ValueError("Only document-root paths are allowed")
        base = self.root if path.startswith("/") or parent is None else parent.parent
        resolved = (base / path.lstrip("/")).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Path escapes trusted document root")
        if resolved.suffix.lower() not in {".md", ".mdx"}:
            raise ValueError("Only static .md/.mdx files are allowed")
        return resolved

    @staticmethod
    def _warn(ctx: _Context, code: str, message: str, line: _Line) -> None:
        ctx.warnings.append(CleaningWarning(
            code=code, message=message, source_path=line.path, source_line=line.number,
        ))

    def from_text(self, text: str, *, source_path: str, **metadata: Any) -> RawDocument:
        """Parse raw file text. Already-parsed loaders should construct RawDocument."""
        self._resolve(source_path)
        lines = text.splitlines(keepends=True)
        frontmatter: dict[str, Any] = {}
        body_start_line = 1
        warnings: list[CleaningWarning] = []
        if lines and lines[0].strip().lstrip("\ufeff") == "---":
            closing = next((i for i in range(1, len(lines)) if lines[i].strip() in {"---", "..."}), None)
            if closing is None:
                warnings.append(CleaningWarning(code="invalid_frontmatter", message="Unclosed frontmatter retained verbatim", source_path=source_path, source_line=1))
            else:
                try:
                    parsed = yaml.safe_load("".join(lines[1:closing]))
                    if parsed is not None and not isinstance(parsed, dict):
                        raise ValueError("Frontmatter must be a mapping")
                    frontmatter = parsed or {}
                    body_start_line = closing + 2
                    text = "".join(lines[closing + 1:])
                except (yaml.YAMLError, ValueError) as exc:
                    warnings.append(CleaningWarning(code="invalid_frontmatter", message=f"Frontmatter retained verbatim: {exc}", source_path=source_path, source_line=1))
        return RawDocument(source_path=source_path, content=text, frontmatter=frontmatter,
                           body_start_line=body_start_line, warnings=warnings, **metadata)

    def clean(self, raw: RawDocument) -> CleanedDocument:
        if raw.language != self.language:
            raise ValueError("RawDocument language must match cleaner target language")
        path = self._resolve(raw.source_path)
        ctx = _Context(warnings=list(raw.warnings))
        fragments = self._render(raw.content, path, raw.body_start_line, (path,), ctx)
        content = "".join(line.text for line in fragments)
        spans: list[SourceSpan] = []
        offset = 0
        for line in fragments:
            if not line.text:
                continue
            end = offset + len(line.text)
            spans.append(SourceSpan(start=offset, end=end, source_path=line.path,
                                    source_line_start=line.number,
                                    source_line_end=line.end_line or line.number))
            offset = end
        deps = [ctx.dependencies[key] for key in sorted(ctx.dependencies)]
        fm = raw.frontmatter
        title = fm.get("title")
        if not isinstance(title, str) or not title.strip():
            title = path.stem
            self._warn(ctx, "missing_title", "Missing static frontmatter title; using filename", _Line("", self._relative(path), 1))
        description = fm.get("description")
        target = fm.get("url")
        redirect = not raw.content.strip() and isinstance(target, str) and bool(target.strip())
        return CleanedDocument(
            doc_id=raw.doc_id, source_revision_id=raw.revision_id,
            revision_id=stable_hash([raw.revision_id, raw.repository_commit, self.language,
                                     CLEANER_VERSION, self.max_depth, self.max_expansions,
                                     self.max_snippet_bytes, [d.model_dump() for d in deps]]),
            repository_commit=raw.repository_commit, source_path=raw.source_path,
            source_url=raw.source_url, product=raw.product, language=raw.language,
            title=title, description=description if isinstance(description, str) else None,
            frontmatter=fm, content=content, cleaner_version=CLEANER_VERSION,
            dependencies=deps, warnings=ctx.warnings, source_spans=spans,
            document_kind="redirect" if redirect else "content",
            redirect_url=target if redirect else None,
        )

    def _language_lines(self, lines: list[_Line], ctx: _Context) -> list[_Line]:
        """Pair first: malformed excluded regions remain visible, never silently lost."""
        # The official source also uses close+open on one line (::::::js),
        # and text after a closer (:::agent's backend:). Tokenize those events
        # without moving their source line or interpreting fenced code.
        events: list[tuple[str, str, _Line]] = []
        for line in lines:
            prefix = re.match(r"[ \t]*", line.text)[0]
            rest = line.text[len(prefix):]
            if line.code or not rest.startswith(":::"):
                events.append(("text", "", line))
                continue
            while rest.startswith(":::"):
                opening = re.fullmatch(r":::([\w-]+)[ \t]*\n?", rest)
                if opening:
                    events.append(("open", opening[1], _Line(prefix + rest, line.path, line.number)))
                    rest = ""
                    break
                tail = rest[3:]
                token = prefix + ":::" + (tail if not tail.strip() else "")
                events.append(("close", "", _Line(token, line.path, line.number)))
                rest = tail if tail.strip() else ""
            if rest:
                events.append(("text", "", _Line(prefix + rest, line.path, line.number)))
        stack: list[int] = []
        pairs: dict[int, int] = {}
        for i, (kind, _, line) in enumerate(events):
            if kind == "open":
                stack.append(i)
            elif kind == "close" and stack:
                start = stack.pop()
                pairs[start] = i
            elif kind == "close":
                self._warn(ctx, "unmatched_directive", "Closing directive without opener retained", line)
        for i in stack:
            self._warn(ctx, "unclosed_directive", "Unclosed directive and its content retained", events[i][2])
        result: list[_Line] = []
        active: list[tuple[int, bool, bool]] = []
        languages = {"python": "python", "js": "javascript", "javascript": "javascript"}
        for i, (_, label, line) in enumerate(events):
            enabled = all(item[1] for item in active)
            if i in pairs:
                known = label in languages
                if not known and enabled:
                    self._warn(ctx, "unsupported_directive", f"Directive {label!r} retained", line)
                if enabled and not known:
                    result.append(line)
                active.append((pairs[i], not known or languages[label] == self.language, known))
            elif active and i == active[-1][0]:
                _, _, known = active.pop()
                if all(item[1] for item in active) and not known:
                    result.append(line)
            elif enabled:
                result.append(line)
        return result

    def _render(self, text: str, path: Path, start_line: int,
                chain: tuple[Path, ...], ctx: _Context) -> list[_Line]:
        # read_text() normalizes newlines; byte-based loaders often do not.
        # Keep identical semantics for both, including Windows CRLF directives.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [_Line(value, self._relative(path), i) for i, value in enumerate(text.splitlines(keepends=True), start_line)]
        if not _mark_fences(lines):
            self._warn(ctx, "unclosed_code_fence", "Original unclosed code fence preserved; no closing fence invented", lines[-1])
        selected = self._language_lines(lines, ctx)
        imports: dict[str, str] = {}
        body: list[_Line] = []
        for line in selected:
            if not line.code and re.match(r"^\s*import\s", line.text):
                match = _IMPORT.match(line.text)
                if match:
                    if match[1] in imports and imports[match[1]] != match[3]:
                        self._warn(ctx, "duplicate_import", "Conflicting component binding retained", line)
                        body.append(line)
                    else:
                        imports[match[1]] = match[3]
                else:
                    self._warn(ctx, "unsupported_import", "Non-default or multiline import retained verbatim", line)
                    body.append(line)
            else:
                if not line.code and re.match(r"^\s*export\b", line.text):
                    self._warn(ctx, "unsupported_export", "Export retained verbatim; not evaluated", line)
                body.append(line)
        body = self._normalize_fence_layout(self._dedent_wrappers(self._join_multiline_tags(body)))
        output: list[_Line] = []
        tags: list[tuple[str, str, _Line]] = []
        comment = False
        inline_run: str | None = None
        for body_index, line in enumerate(body):
            if line.code:
                if line.fence_role == "open":
                    # A retained/unknown HTML wrapper can otherwise swallow a
                    # fence as HTML. A blank boundary also separates adjacent
                    # examples without touching any code body characters.
                    tail = "".join(item.text for item in output[-8:])
                    if tail.strip() and not re.search(r"\n[ \t]*\n[ \t]*$", tail):
                        output.append(_Line("\n" if tail.endswith("\n") else "\n\n", line.path, line.number))
                output.append(line)
                continue
            # Comments are layout only. Do not treat comment-looking text in code as comments.
            text = line.text
            position = 0
            while position < len(text):
                if inline_run is not None:
                    closing_tick = re.search(r"(?<!`)" + re.escape(inline_run) + r"(?!`)", text[position:])
                    end = len(text) if closing_tick is None else position + closing_tick.end()
                    output.append(_Line(text[position:end], line.path, line.number, end_line=line.end_line))
                    position = end
                    if closing_tick is not None:
                        inline_run = None
                    continue
                if comment:
                    end = text.find("*/}", position)
                    if end < 0:
                        break
                    comment = False
                    position = end + 3
                    continue
                if text.startswith("{/*", position):
                    comment = True
                    position += 3
                    continue
                if text[position] == "`":
                    run = re.match(r"`+", text[position:])[0]
                    output.append(_Line(run, line.path, line.number, end_line=line.end_line))
                    position += len(run)
                    paragraph = text[position:]
                    for following in body[body_index + 1:]:
                        if following.code or not following.text.strip():
                            break
                        paragraph += following.text
                    if re.search(r"(?<!`)" + re.escape(run) + r"(?!`)", paragraph):
                        inline_run = run
                    else:
                        self._warn(ctx, "unmatched_backtick", "Unmatched backtick retained as literal within its paragraph", line)
                    continue
                if text[position] == "<":
                    parsed = self._tag(text, position)
                    if parsed:
                        end, name, closing, self_closing, attrs, dynamic = parsed
                        token = text[position:end]
                        replacement = self._component(name, closing, self_closing, attrs, dynamic,
                                                      token, line, imports, path, chain, ctx, tags)
                        output.extend(replacement)
                        position = end
                        continue
                    if _TAG_NAME.match(text, position):
                        self._warn(ctx, "malformed_tag", "Incomplete tag retained verbatim", line)
                if text[position] == "{" and not text.startswith("{/*", position):
                    self._warn(ctx, "unsupported_expression", "JSX expression retained verbatim; not evaluated", line)
                # Keep Markdown, inline expressions and unknown syntax visible.
                end = position + 1
                while end < len(text) and text[end] not in "<`" and not text.startswith("{/*", end):
                    end += 1
                output.append(_Line(text[position:end], line.path, line.number, end_line=line.end_line))
                position = end
        if comment and body:
            self._warn(ctx, "unclosed_comment", "Unclosed JSX comment; original text retained below", body[-1])
            # Conservative recovery, including all text that may have been hidden.
            output.append(_Line("\n[unclosed JSX comment: original body follows]\n" + "".join(x.text for x in body), self._relative(path), start_line,
                                end_line=body[-1].number))
        if inline_run is not None and body:
            self._warn(ctx, "unclosed_inline_code", "Unclosed inline code retained conservatively", body[-1])
        for name, _, origin in tags:
            self._warn(ctx, "unclosed_component", f"Unclosed component {name}", origin)
        return output

    @classmethod
    def _dedent_wrappers(cls, lines: list[_Line]) -> list[_Line]:
        """Remove common JSX layout indentation, preserving relative code indent.

        Four-space-indented Card text and CodeGroup fences would otherwise
        become Markdown indented code. Only balanced, standalone
        wrappers participate. Ignore code *bodies* when choosing the component
        margin: multiline string contents may deliberately begin at column 0.
        Unknown wrappers keep their tags and warnings, but their layout margin
        must not turn contained fenced examples into indented Markdown code.
        """
        stack: list[tuple[str, int, int | None]] = []
        pairs: list[tuple[int, int, int | None]] = []
        list_stack: list[tuple[int, int]] = []
        for i, line in enumerate(lines):
            if line.code:
                continue
            start = len(line.text) - len(line.text.lstrip())
            if line.text.strip():
                while list_stack and start < list_stack[-1][1]:
                    list_stack.pop()
                marker = re.match(r"[ \t]*(?:[-+*]|\d+[.)])[ \t]+", line.text)
                if marker:
                    list_stack.append((i, marker.end()))
            parsed = cls._tag(line.text, start)
            if not parsed:
                continue
            end, name, closing, self_closing, _, _ = parsed
            if self_closing or line.text[end:].strip():
                continue
            if not closing:
                stack.append((name, i, list_stack[-1][0] if list_stack else None))
            elif stack and stack[-1][0] == name:
                _, begin, owner = stack.pop()
                pairs.append((begin, i, owner))
        for begin, end, owner in sorted(pairs, key=lambda pair: pair[0]):
            region = lines[begin + 1:end]
            indents = [len(line.text) - len(line.text.lstrip(" ")) for line in region
                       if line.text.strip() and (not line.code or line.fence_role == "open")]
            marker = re.match(r"[ \t]*(?:[-+*]|\d+[.)])[ \t]+", lines[owner].text) if owner is not None else None
            list_margin = marker.end() if marker else 0
            common = max(0, min(indents, default=0) - list_margin)
            if common:
                for line in region:
                    trim = min(common, len(line.text) - len(line.text.lstrip(" ")))
                    line.text = line.text[trim:]
        return lines

    @classmethod
    def _normalize_fence_layout(cls, lines: list[_Line]) -> list[_Line]:
        """Give each code block a valid Markdown margin, preserving list owners.

        Remove only the opening fence's layout indentation from its body;
        preserve indentation *within* the code. Normalize the closing fence
        independently, as source JSX often uses a different closing margin.
        """
        lists: list[int] = []
        code_margin: tuple[int, int] | None = None
        for line in lines:
            prefix = re.match(r"[ \t]*", line.text)[0]
            indent = len(prefix)
            if line.fence_role == "open":
                while lists and indent < lists[-1]:
                    lists.pop()
                owner = lists[-1] if lists else 0
                code_margin = (indent, owner)
                line.text = " " * owner + line.text[indent:]
            elif line.code and code_margin is not None:
                removed, owner = code_margin
                if line.fence_role == "close":
                    line.text = " " * owner + line.text[indent:]
                    code_margin = None
                else:
                    line.text = " " * owner + line.text[min(removed, indent):]
            elif not line.code and line.text.strip():
                while lists and indent < lists[-1]:
                    lists.pop()
                marker = re.match(r"[ \t]*(?:[-+*]|\d+[.)])([ \t]+)", line.text)
                if marker:
                    lists.append(marker.end())
                else:
                    tag = cls._tag(line.text, indent)
                    if tag and not line.text[tag[0]:].strip():
                        owner = lists[-1] if lists else 0
                        line.text = " " * owner + line.text[indent:]
        return lines

    @classmethod
    def _join_multiline_tags(cls, lines: list[_Line]) -> list[_Line]:
        """Join only incomplete JSX tokens, retaining their inclusive line range."""
        result: list[_Line] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            text = line.text
            if not line.code:
                # Inline-code spans must remain opaque, including apparent tags.
                opaque = False
                for pos, char in enumerate(text):
                    if char == "`":
                        opaque = not opaque
                    if char == "<" and not opaque and _TAG_NAME.match(text, pos) and cls._tag(text, pos) is None:
                        j = i + 1
                        while j < len(lines) and not lines[j].code:
                            text += lines[j].text
                            if cls._tag(text, pos) is not None:
                                line = _Line(text, line.path, line.number, end_line=lines[j].number)
                                i = j
                                break
                            j += 1
                        break
            result.append(line)
            i += 1
        return result

    @staticmethod
    def _tag(text: str, start: int) -> tuple | None:
        match = _TAG_NAME.match(text, start)
        if not match:
            return None
        quote: str | None = None
        braces = 0
        for i in range(match.end(), len(text)):
            char = text[i]
            if quote:
                if char == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "{":
                braces += 1
            elif char == "}":
                braces -= 1
            elif char == ">" and braces == 0:
                attr_text = text[match.end():i].rstrip("/ ")
                attrs = {m[1]: m[2][1:-1] for m in _ATTR.finditer(attr_text) if m[2][0] in "\"'"}
                for attribute in _ATTR.finditer(attr_text):
                    if attribute[2].startswith("{"):
                        try:
                            value = ast.literal_eval(attribute[2][1:-1])
                            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                                attrs[attribute[1]] = ", ".join(value)
                        except (ValueError, SyntaxError):
                            pass
                without_static = _ATTR.sub(lambda m: "" if m[2][0] in "\"'" else m[0], attr_text)
                dynamic = "{" in without_static
                return i + 1, match[1], text.startswith("</", start), text[start:i].rstrip().endswith("/"), attrs, dynamic
        return None

    def _component(self, name: str, closing: bool, self_closing: bool, attrs: dict[str, str],
                   dynamic: bool, token: str, line: _Line, imports: dict[str, str],
                   path: Path, chain: tuple[Path, ...], ctx: _Context,
                   tags: list[tuple[str, str, _Line]]) -> list[_Line]:
        def fragment(value: str) -> list[_Line]:
            return [_Line(value, line.path, line.number, end_line=line.end_line)] if value else []

        if name in imports:
            if not self_closing or closing or dynamic or attrs:
                self._warn(ctx, "dynamic_snippet", f"Snippet {name} requires unsupported invocation/props", line)
                return fragment(f"[unresolved snippet: {name}] {token}")
            return self._expand(imports[name], name, line, path, chain, ctx)
        known = name in _WRAPPERS | _DECORATIVE | {"br", "img", "strong", "b", "em", "i"}
        if not known:
            self._warn(ctx, "unsupported_component", f"Component {name} retained verbatim", line)
            if dynamic:
                self._warn(ctx, "dynamic_component", f"Dynamic component {name} not evaluated", line)
                return fragment(f"[unresolved component: {name}] {token}")
            return fragment(token)
        semantic_dynamic = any(key not in attrs and re.search(r"\b" + key + r"\s*=", token)
                               for key in ("title", "href", "src", "alt", "description"))
        if dynamic and (semantic_dynamic or "{..." in token):
            self._warn(ctx, "dynamic_component", f"Dynamic metadata on {name} retained", line)
            return fragment(f"[unresolved component metadata: {token}]")
        if name == "Icon":
            return fragment(attrs.get("alt", ""))
        if name == "br":
            return fragment("<br />" if not closing else "")  # preserve Markdown table rows
        if name == "img":
            return fragment(f"![{attrs.get('alt', '')}]({attrs.get('src', '')})")
        if closing:
            if tags and tags[-1][0] == name:
                return fragment(tags.pop()[1])
            self._warn(ctx, "mismatched_component", f"Unexpected closing {name} retained", line)
            return fragment(token)
        title = attrs.get("title", "")
        href = attrs.get("href", "")
        opening, ending = "", ""
        if name == "a" and href:
            opening, ending = "[", f"]({href})"
        elif title:
            opening = f"[{title}]({href})\n" if href else f"**{title}**\n"
        elif href:
            opening = f"[{href}]({href})\n"
        elif name in {"strong", "b", "em", "i"}:
            opening = ending = "**" if name in {"strong", "b"} else "*"
        if attrs.get("description"):
            opening += attrs["description"] + "\n"
        if name in {"ParamField", "ResponseField"}:
            field_name = next((attrs[key] for key in ("name", "body", "path", "query", "header") if key in attrs), "Field")
            details = [attrs[key] for key in ("type", "default", "post") if key in attrs]
            if re.search(r"\srequired(?:\s|/|>)", token):
                details.append("required")
            opening = f"**`{field_name}`**" + (" (" + "; ".join(details) + ")" if details else "") + "\n\n"
        elif name == "Tooltip" and attrs.get("tip"):
            ending += f" ({attrs['tip']})"
        elif name == "Update" and attrs.get("label"):
            opening = f"**{attrs['label']}**\n" + opening
        elif name == "code":
            opening = ending = "`"
        if not self_closing:
            tags.append((name, ending, line))
        return fragment(opening + (ending if self_closing else ""))

    def _expand(self, reference: str, name: str, line: _Line, path: Path,
                chain: tuple[Path, ...], ctx: _Context) -> list[_Line]:
        def failure(code: str, reason: str) -> list[_Line]:
            self._warn(ctx, code, reason, line)
            return [_Line(f"[unresolved snippet: {name}; {code}]", line.path, line.number)]

        try:
            target = self._resolve(reference, path)
        except (ValueError, OSError, RuntimeError) as exc:
            return failure("unsafe_snippet_path", str(exc))
        if target in chain:
            return failure("snippet_cycle", " -> ".join(self._relative(p) for p in (*chain, target)))
        if len(chain) >= self.max_depth or ctx.expansions >= self.max_expansions:
            return failure("snippet_limit", "Snippet depth or expansion limit reached")
        try:
            if target.stat().st_size > self.max_snippet_bytes:
                return failure("snippet_limit", "Snippet file size limit reached")
            data = target.read_bytes()
            text = data.decode("utf-8-sig")
        except (OSError, UnicodeError) as exc:
            return failure("missing_snippet", f"Cannot read {reference}: {exc}")
        relative = self._relative(target)
        dep = ctx.dependencies.setdefault(relative, SnippetDependency(
            source_path=relative, content_hash=hashlib.sha256(data).hexdigest(),
        ))
        if line.path not in dep.referenced_by:
            dep.referenced_by.append(line.path)
            dep.referenced_by.sort()
        ctx.expansions += 1
        snippet = self.from_text(text, source_path=relative, language=self.language)
        ctx.warnings.extend(snippet.warnings)
        expanded = self._render(snippet.content, target, snippet.body_start_line, (*chain, target), ctx)
        prefix = re.match(r"[ \t]*", line.text)[0]
        if prefix and line.text.lstrip().startswith(f"<{name}"):
            indented: list[_Line] = []
            at_start = True
            for item in expanded:
                for part in item.text.splitlines(keepends=True):
                    indented.append(_Line((prefix if at_start else "") + part,
                                          item.path, item.number, code=item.code,
                                          end_line=item.end_line))
                    at_start = part.endswith("\n")
            expanded = indented
        # Fence-only snippets must start/end on their own lines, even inline uses.
        return [_Line("\n", line.path, line.number), *expanded, _Line("\n", line.path, line.number)]
