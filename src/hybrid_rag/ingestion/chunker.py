"""Contiguous structure-aware chunks; code, tables and lists are atomic."""
from markdown_it import MarkdownIt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hybrid_rag.schemas.document import Section, Chunk, stable_hash
from .provenance import slice_spans

CHUNKER_VERSION = "structural-recursive-v1"


class Chunker:
    def refine_tokens(self, section, chunks, refiner):
        """Refine existing chunks with a configured TokenRefiner, leaving small ones intact."""
        return refiner.refine(section, chunks)

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require chunk_size > chunk_overlap >= 0")
        self.size, self.overlap = chunk_size, chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            add_start_index=True, strip_whitespace=False, keep_separator=True)

    def split(self, section: Section) -> list[Chunk]:
        text = section.content
        if not text.strip():
            return []
        offsets = [0]
        for line in text.splitlines(True):
            offsets.append(offsets[-1] + len(line))
        tokens = MarkdownIt().enable("table").parse(text)
        kinds = {"fence": "code", "code_block": "code", "table_open": "table", "bullet_list_open": "list", "ordered_list_open": "list", "heading_open": "heading"}
        starts = {}
        for token in tokens:
            if token.level == 0 and token.map and token.nesting != -1:
                starts.setdefault(offsets[token.map[0]], kinds.get(token.type, "paragraph"))
        points = sorted(starts)
        if not points:
            points = [0]
        blocks = [(0 if i == 0 else start, points[i + 1] if i + 1 < len(points) else len(text), starts.get(start, "paragraph"))
                  for i, start in enumerate(points)]
        ranges = []
        pending = []

        def emit(group):
            if group:
                ranges.append((group[0][0], group[-1][1], list(dict.fromkeys(b[2] for b in group))))

        for block in blocks:
            start, end, kind = block
            if end - start > self.size:
                emit(pending); pending = []
                contains_protected = any(token.type in {"fence", "code_block", "table_open"} and token.map
                    and offsets[token.map[0]] >= start and offsets[token.map[1]] <= end for token in tokens)
                if kind in {"code", "table", "list"} or contains_protected:
                    ranges.append((start, end, [kind]))
                else:
                    for piece in self.splitter.create_documents([text[start:end]]):
                        begin = start + piece.metadata["start_index"]
                        finish = begin + len(piece.page_content)
                        if begin < start or text[begin:finish] != piece.page_content:
                            raise ValueError("Recursive splitter returned an invalid source offset")
                        ranges.append((begin, finish, [kind]))
                continue
            if pending and end - pending[0][0] > self.size:
                if kind == "code" and pending[-1][2] == "paragraph" and end - pending[-1][0] <= self.size:
                    emit(pending[:-1]); pending = pending[-1:]
                else:
                    emit(pending); pending = []
            pending.append(block)
        emit(pending)
        result = []
        for start, end, types in ranges:
            content = text[start:end]
            if not content.strip():
                continue
            for token in tokens:
                if token.map and offsets[token.map[0]] < end and offsets[token.map[1]] > start:
                    kind = kinds.get(token.type, "paragraph" if token.type == "paragraph_open" else None)
                    if kind and kind not in types:
                        types.append(kind)
                    if token.children and any(child.type == "image" for child in token.children) and "image" not in types:
                        types.append("image")
            result.append(Chunk(chunk_id=stable_hash([CHUNKER_VERSION, section.section_id, self.size, self.overlap, start, end, content]),
                doc_id=section.doc_id, revision_id=section.revision_id, section_id=section.section_id,
                chunk_index=len(result), section_start=start, section_end=end, heading_path=section.heading_path,
                content=content, embedding_text=" > ".join(section.heading_path) + "\n\n" + content,
                block_types=types, source_path=section.source_path, source_url=section.source_url,
                repository_commit=section.repository_commit, source_spans=slice_spans(section.source_spans, start, end),
                dependencies=section.dependencies, warnings=section.warnings, oversized=len(content) > self.size))
        return result
