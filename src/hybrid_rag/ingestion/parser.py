"""Heading hierarchy with offsets into the unmodified cleaned document."""
from markdown_it import MarkdownIt
from hybrid_rag.schemas.document import Document, Section, stable_hash
from .provenance import slice_spans

PARSER_VERSION = "markdown-sections-v1"


class SectionParser:
    def parse(self, document: Document) -> list[Section]:
        if document.document_kind == "redirect":
            return []
        offsets = [0]
        for line in document.content.splitlines(True):
            offsets.append(offsets[-1] + len(line))
        tokens = MarkdownIt().enable("table").parse(document.content)
        headings = [(offsets[t.map[0]], offsets[t.map[1]], int(t.tag[1:]), tokens[i + 1].content)
                    for i, t in enumerate(tokens) if t.type == "heading_open"]
        records = []
        if not headings or document.content[:headings[0][0]].strip():
            records.append((0, headings[0][0] if headings else len(document.content), 0, ""))
        for index, (_, body_start, level, title) in enumerate(headings):
            records.append((body_start, headings[index + 1][0] if index + 1 < len(headings) else len(document.content), level, title))
        sections = []
        stack = []
        for start, end, level, title in records:
            while stack and stack[-1].heading_level >= level:
                stack.pop()
            parent = stack[-1] if stack and level else None
            path = list(parent.heading_path) if parent else [document.title]
            if level and not (level == 1 and not parent and title == document.title):
                path.append(title)
            index = len(sections)
            section = Section(section_id=stable_hash([PARSER_VERSION, document.revision_id, document.doc_id, index, path]),
                doc_id=document.doc_id, revision_id=document.revision_id,
                parent_section_id=parent.section_id if parent else None, heading_path=path,
                heading_level=level, section_index=index, content_start=start, content_end=end,
                content=document.content[start:end], source_path=document.source_path,
                source_url=document.source_url, repository_commit=document.repository_commit,
                source_spans=slice_spans(document.source_spans, start, end),
                dependencies=document.dependencies, warnings=document.warnings)
            sections.append(section)
            if level:
                stack.append(section)
        return sections
