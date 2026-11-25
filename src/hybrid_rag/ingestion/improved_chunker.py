"""Optional optimization layered on the unchanged baseline Chunker.

Only oversized top-level Python fences, Markdown tables and Markdown lists
participate. Unsupported syntax and indivisible structures retain the baseline.
"""
import ast
from markdown_it import MarkdownIt
from hybrid_rag.schemas.document import ChunkPart, stable_hash
from .chunker import Chunker
from .provenance import slice_spans

IMPROVED_VERSION = "oversized-structures-v2"


class ImprovedChunker(Chunker):
    def __init__(self, chunk_size=800, chunk_overlap=120, *, python=True, tables=True, lists=True, captions=True):
        super().__init__(chunk_size, chunk_overlap)
        self.options = dict(python=python, tables=tables, lists=lists, captions=captions)

    def _pack(self, ranges, budget):
        groups = []
        current = []
        for start, end in ranges:
            if current and end - current[0][0] > budget:
                groups.append((current[0][0], current[-1][1])); current = []
            current.append((start, end))
        if current:
            groups.append((current[0][0], current[-1][1]))
        return groups

    def _plan(self, chunk):
        text = chunk.content
        tokens = MarkdownIt().enable("table").parse(text)
        top = [t for t in tokens if t.level == 0 and t.map and t.nesting != -1]
        if len(top) != 1:
            return None
        token = top[0]
        offsets = [0]
        for line in text.splitlines(True):
            offsets.append(offsets[-1] + len(line))
        if token.type == "fence" and self.options['python'] and token.info.split() and token.info.split()[0] in {"py", "python"}:
            begin, end = offsets[token.map[0] + 1], offsets[token.map[1] - 1]
            body = text[begin:end]
            if body != token.content:
                return None
            try:
                tree = ast.parse(body)
            except (SyntaxError, ValueError, RecursionError):
                return None
            lines = [0]
            for line in body.splitlines(True):
                lines.append(lines[-1] + len(line))
            if len(tree.body) < 2:
                return None  # a single giant function/class stays intact
            cuts = [0]
            for node in tree.body[1:]:
                first = min([node.lineno, *[d.lineno for d in getattr(node, 'decorator_list', [])]])
                cuts.append(lines[first - 1])
            cuts.append(len(body))
            groups = self._pack(list(zip(cuts, cuts[1:])), max(1, self.size - begin - (len(text) - end)))
            if len(groups) < 2:
                return None
            for a, b in groups:
                try:
                    ast.parse(body[a:b])
                except (SyntaxError, ValueError, RecursionError):
                    return None
            return "python_ast", [[(0, begin, "wrapper"), (begin+a, begin+b, "body"), (end, len(text), "wrapper")] for a, b in groups]
        if token.type == "table_open" and self.options['tables']:
            # markdown-it provides row boundaries; never split cells on '|'.
            rows = [t for t in tokens if t.type == "tr_open" and t.map]
            if len(rows) < 3:
                return None
            header_end = offsets[rows[1].map[0]]
            starts = [offsets[t.map[0]] for t in rows[1:]]
            ends = starts[1:] + [len(text)]
            groups = self._pack(list(zip(starts, ends)), max(1, self.size-header_end))
            if len(groups) < 2:
                return None
            return "table_rows", [[(0, header_end, "body" if i == 0 else "context"), (a, b, "body")] for i, (a, b) in enumerate(groups)]
        if token.type in {"bullet_list_open", "ordered_list_open"} and self.options['lists']:
            items = [t for t in tokens if t.type == "list_item_open" and t.level == 1 and t.map]
            if len(items) < 2:
                return None
            starts = [0] + [offsets[t.map[0]] for t in items[1:]]
            groups = self._pack(list(zip(starts, starts[1:] + [len(text)])), self.size)
            if len(groups) > 1:
                return "list_items", [[(a, b, "body")] for a, b in groups]
        return None

    def split(self, section):
        output = []
        for original in super().split(section):
            caption = (output[-1] if output and self.options['captions'] and original.oversized
                       and original.block_types == ['code'] and output[-1].block_types == ['paragraph']
                       and len(output[-1].content) < 100 and output[-1].section_end == original.section_start else None)
            plan = self._plan(original) if original.oversized else None
            if plan is None:
                if caption is not None:
                    output.pop()
                    content = caption.content + original.content
                    start, end = caption.section_start, original.section_end
                    output.append(original.model_copy(update=dict(
                        chunk_id=stable_hash([IMPROVED_VERSION, section.section_id, self.options, self.size, self.overlap, 'code_with_caption', start, end, content]),
                        content=content, embedding_text=' > '.join(section.heading_path)+'\n\n'+content,
                        section_start=start, source_spans=slice_spans(section.source_spans,start,end),
                        block_types=['paragraph','code'], split_method='code_with_caption', oversized=len(content)>self.size)))
                else:
                    output.append(original)
                continue
            method, groups = plan
            for group_index, group in enumerate(groups):
                pieces, spans, parts = [], [], []
                cursor = 0
                absolute = [(original.section_start+a, original.section_start+b, role) for a,b,role in group]
                merged = caption is not None and group_index == 0 and method == 'python_ast'
                if merged:
                    output.pop()
                    absolute.insert(0,(caption.section_start,caption.section_end,'body'))
                for a, b, role in absolute:
                    if a == b:
                        continue
                    piece = section.content[a:b]
                    pieces.append(piece)
                    parts.append(ChunkPart(start=cursor, end=cursor+len(piece), section_start=a, section_end=b, role=role))
                    spans.extend(s.model_copy(update={"start": s.start+cursor, "end": s.end+cursor}) for s in slice_spans(section.source_spans, a, b))
                    cursor += len(piece)
                content = ''.join(pieces)
                output.append(original.model_copy(update=dict(
                    chunk_id=stable_hash([IMPROVED_VERSION, section.section_id, self.size, self.overlap, self.options, method, [p.model_dump() for p in parts], content]),
                    content=content, embedding_text=' > '.join(section.heading_path)+'\n\n'+content,
                    content_parts=parts, split_method=method, source_spans=spans,
                    block_types=['paragraph',*original.block_types] if merged else original.block_types,
                    section_start=min(p.section_start for p in parts), section_end=max(p.section_end for p in parts),
                    oversized=len(content)>self.size)))
        for i, chunk in enumerate(output):
            chunk.chunk_index = i
        return output
