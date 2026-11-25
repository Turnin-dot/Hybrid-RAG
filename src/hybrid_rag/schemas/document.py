"""Data contracts; section parsing and chunk construction are deliberately separate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


class CleaningWarning(BaseModel):
    code: str
    message: str
    source_path: str
    source_line: int = Field(ge=1)
    severity: Literal["warning", "error"] = "warning"


class SnippetDependency(BaseModel):
    source_path: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    referenced_by: list[str] = Field(default_factory=list)


class SourceSpan(BaseModel):
    """Half-open offsets in cleaned content, mapped to inclusive raw source lines.

    Generated labels/placeholders map to their originating component line.
    Offsets are Python string character offsets, not UTF-8 byte offsets.
    """

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    source_path: str
    source_line_start: int = Field(ge=1)
    source_line_end: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self) -> SourceSpan:
        if self.end < self.start or self.source_line_end < self.source_line_start:
            raise ValueError("Source span ranges must be ordered")
        return self


class RawDocument(BaseModel):
    """Loader contract: content is the body; frontmatter was already extracted.

    Use MDXCleaner.from_text for an unparsed MDX file, or from_loader for a
    mapping with explicit body/content and metadata aliases.
    """

    model_config = ConfigDict(extra="forbid")
    source_path: str
    content: str
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    doc_id: str = ""
    revision_id: str | None = None
    repository_commit: str | None = None
    product: str | None = None
    language: Literal["python", "javascript"] = "python"
    source_url: str | None = None
    body_start_line: int = Field(default=1, ge=1)
    warnings: list[CleaningWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def assign_ids(self) -> RawDocument:
        if not self.doc_id:
            self.doc_id = stable_hash([self.source_path, self.language])
        if self.revision_id is None:
            self.revision_id = stable_hash([self.frontmatter, self.content])
        return self

    @classmethod
    def from_loader(cls, record: Mapping[str, Any]) -> RawDocument:
        """Explicit adapter, with conflicting aliases rejected instead of lost."""
        values = dict(record)
        for old, new in (("body", "content"), ("metadata", "frontmatter"),
                         ("git_commit", "repository_commit"), ("version", "repository_commit")):
            if old in values:
                if new in values and values[new] != values[old]:
                    raise ValueError(f"Conflicting loader fields: {old}, {new}")
                values[new] = values.pop(old)
        return cls.model_validate(values)


class Document(BaseModel):
    """One cleaned logical page, with title kept out of the Markdown body."""

    model_config = ConfigDict(extra="forbid")
    doc_id: str
    revision_id: str
    source_revision_id: str
    repository_commit: str | None = None
    source_path: str
    source_url: str | None = None
    product: str | None = None
    language: Literal["python", "javascript"] = "python"
    title: str
    description: str | None = None
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    content: str
    cleaner_version: str
    document_kind: Literal["content", "redirect"] = "content"
    redirect_url: str | None = None
    dependencies: list[SnippetDependency] = Field(default_factory=list)
    warnings: list[CleaningWarning] = Field(default_factory=list)
    source_spans: list[SourceSpan] = Field(default_factory=list)


class CleanedDocument(Document):
    """Typed cleaner output, sharing the persisted Document contract."""

    def to_document(self) -> Document:
        return Document.model_validate(self.model_dump())


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str
    doc_id: str
    revision_id: str
    parent_section_id: str | None = None
    heading_path: list[str]
    heading_level: int = Field(ge=0, le=6, description="0 denotes the untitled preamble")
    section_index: int = Field(ge=0)
    content_start: int = Field(default=0, ge=0)
    content_end: int = Field(default=0, ge=0)
    content: str = Field(description="Direct body only; excludes child section bodies")
    source_path: str
    source_url: str | None = None
    repository_commit: str | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    dependencies: list[SnippetDependency] = Field(default_factory=list)
    warnings: list[CleaningWarning] = Field(default_factory=list)


class ChunkPart(BaseModel):
    """Exact output slice mapped to a Section slice; repeated context is explicit."""
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    section_start: int = Field(ge=0)
    section_end: int = Field(ge=0)
    role: Literal["body", "context", "wrapper"] = "body"


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    doc_id: str
    revision_id: str
    section_id: str
    chunk_index: int = Field(ge=0)
    section_start: int = Field(default=0, ge=0)
    section_end: int = Field(default=0, ge=0)
    heading_path: list[str]
    content: str = Field(min_length=1)
    embedding_text: str = Field(min_length=1)
    block_types: list[Literal["paragraph", "heading", "code", "list", "table", "image", "other"]] = Field(min_length=1)
    source_path: str
    source_url: str | None = None
    repository_commit: str | None = None
    source_spans: list[SourceSpan] = Field(default_factory=list)
    dependencies: list[SnippetDependency] = Field(default_factory=list)
    warnings: list[CleaningWarning] = Field(default_factory=list)
    oversized: bool = False
    content_parts: list[ChunkPart] = Field(default_factory=list)
    split_method: str | None = None
