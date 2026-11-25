"""One-to-one retrieval adapters for validated ingestion Chunks.

The dense and sparse indexes share the same retrieval unit, so a stable
``unit_id`` and identical source metadata are used by both. Over-budget inputs
are rejected and sent back to the existing ingestion TokenRefiner. The legacy
split_content utilities remain for compatibility, not index preparation.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable

from pydantic import BaseModel, Field
from hybrid_rag.schemas.document import ChunkPart, SourceSpan


class RetrievalUnit(BaseModel):
    unit_id: str
    chunk_id: str
    doc_id: str
    section_id: str
    source_path: str
    heading_path: list[str]
    text: str
    section_start: int = Field(ge=0)
    section_end: int = Field(ge=0)
    split_index: int = Field(default=0, ge=0)
    parent_count: int = Field(default=1, ge=1)
    content_parts: list[ChunkPart] = Field(default_factory=list)
    source_spans: list[SourceSpan] = Field(default_factory=list)

    @property
    def ranges(self) -> list[list[int]]:
        return [[p.section_start,p.section_end] for p in self.content_parts] or [[self.section_start, self.section_end]]


_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_CLOSE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*$")


def _heading_prefix(heading_path: list[str]) -> str:
    return " > ".join(heading_path) + "\n\n"


def iter_blocks(text: str):
    """Yield ``(start, end, block)`` that tile ``text`` exactly; fenced code stays whole.

    The concatenation of all returned blocks equals ``text``, so offsets map
    one-to-one onto the source content.
    """
    lines = text.splitlines(keepends=True)
    blocks: list[tuple[int, int, str]] = []
    cur: list[str] = []
    cur_start = 0
    pos = 0
    in_fence = False
    fence_marker = ""
    for line in lines:
        stripped = line.lstrip()
        if in_fence:
            cur.append(line)
            if _CLOSE_RE.match(stripped) and stripped.lstrip().startswith(fence_marker):
                in_fence = False
                blocks.append((cur_start, pos + len(line), "".join(cur)))
                cur = []
        else:
            m = _FENCE_RE.match(stripped)
            if m:
                if cur:
                    blocks.append((cur_start, pos, "".join(cur)))
                cur = [line]
                cur_start = pos
                in_fence = True
                fence_marker = m.group(1)
            elif line.strip() == "":
                if not cur:
                    cur_start = pos
                cur.append(line)
                blocks.append((cur_start, pos + len(line), "".join(cur)))
                cur = []
            else:
                if not cur:
                    cur_start = pos
                cur.append(line)
        pos += len(line)
    if cur:
        blocks.append((cur_start, pos, "".join(cur)))
    return blocks


def _hard_split(text: str, budget: int, token_count: Callable[[str], int]):
    """Hard-split a single oversized run (no usable line boundary) by token budget."""
    pieces = []
    remaining = text
    while remaining:
        if token_count(remaining) <= budget:
            pieces.append(remaining)
            break
        lo, hi = 0, len(remaining)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if token_count(remaining[:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        lo = max(lo, 1)
        pieces.append(remaining[:lo])
        remaining = remaining[lo:]
    return pieces


def _split_text(text: str, budget: int, token_count: Callable[[str], int]):
    """Split ``text`` into ``(sub_text, rel_start, rel_end)`` pieces, each <= budget tokens.

    Prefers line boundaries; falls back to a hard split for a single oversized line.
    """
    out: list[tuple[str, int, int]] = []
    buf: list[str] = []
    buf_tokens = 0
    pos = 0
    for line in text.splitlines(keepends=True):
        t = token_count(line)
        if t > budget:
            if buf:
                out.append(("".join(buf), pos - sum(len(x) for x in buf), pos))
                buf = []
                buf_tokens = 0
            for sub in _hard_split(line, budget, token_count):
                out.append((sub, pos, pos + len(sub)))
                pos += len(sub)
            continue
        if buf_tokens + t > budget:
            out.append(("".join(buf), pos - sum(len(x) for x in buf), pos))
            buf = []
            buf_tokens = 0
        buf.append(line)
        buf_tokens += t
        pos += len(line)
    if buf:
        out.append(("".join(buf), pos - sum(len(x) for x in buf), pos))
    return out


def _pack(blocks, budget: int, token_count: Callable[[str], int]):
    """Greedily pack blocks into pieces <= budget; hard-split oversized blocks."""
    pieces: list[tuple[str, int, int]] = []
    buf: list[tuple[str, int, int]] = []
    buf_tokens = 0
    for start, end, block in blocks:
        t = token_count(block)
        if t > budget:
            if buf:
                pieces.append((_join(buf), buf[0][1], buf[-1][2]))
                buf = []
                buf_tokens = 0
            for sub, rel_s, rel_e in _split_text(block, budget, token_count):
                pieces.append((sub, start + rel_s, start + rel_e))
            continue
        if buf_tokens + t > budget:
            pieces.append((_join(buf), buf[0][1], buf[-1][2]))
            buf = []
            buf_tokens = 0
        buf.append((block, start, end))
        buf_tokens += t
    if buf:
        pieces.append((_join(buf), buf[0][1], buf[-1][2]))
    return pieces


def _join(buf) -> str:
    return "".join(b[0] for b in buf)


def split_content(content: str, body_budget: int, token_count: Callable[[str], int]):
    """Split ``content`` into sub-pieces each within ``body_budget`` tokens.

    Returns ``(sub_text, rel_start, rel_end)`` with offsets relative to ``content``.
    """
    blocks = iter_blocks(content)
    pieces = _pack(blocks, body_budget, token_count)
    return [(t, s, e) for t, s, e in pieces]


def build_units(chunks: list[dict], tokenize: Callable[[str], int], max_tokens: int = 512):
    """A lossless one-to-one adapter for already validated ingestion Chunks.

    Token splitting belongs to the existing ingestion TokenRefiner. Reject an
    incompatible export here instead of maintaining a second splitting policy.
    """
    units: list[RetrievalUnit] = []
    for chunk in chunks:
        total = tokenize(chunk["embedding_text"])
        if total>max_tokens:raise ValueError(f"Chunk {chunk['chunk_id']} has {total}>{max_tokens} Tokens; refine the ingestion export first")
        if not chunk['content'].strip():raise ValueError('Empty Chunk')
        parts=[ChunkPart.model_validate(p) for p in chunk.get('content_parts',[])]
        cursor=0
        for p in parts:
            if p.start!=cursor or p.end-p.start!=p.section_end-p.section_start or p.end<=p.start:raise ValueError('Invalid Chunk part mapping')
            cursor=p.end
        if parts and cursor!=len(chunk['content']):raise ValueError('Incomplete Chunk part mapping')
        if not parts and chunk['section_end']-chunk['section_start']!=len(chunk['content']):raise ValueError('Noncontiguous Chunk requires content_parts')
        units.append(RetrievalUnit(unit_id=chunk['chunk_id'],chunk_id=chunk['chunk_id'],doc_id=chunk['doc_id'],
            section_id=chunk['section_id'],source_path=chunk['source_path'],heading_path=chunk['heading_path'],text=chunk['embedding_text'],
            section_start=chunk['section_start'],section_end=chunk['section_end'],content_parts=parts,source_spans=chunk.get('source_spans',[])))
    if len({u.unit_id for u in units})!=len(units):raise ValueError('Duplicate Chunk ID')
    return units


def unit_id_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
