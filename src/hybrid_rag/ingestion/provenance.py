"""Slice and rebase source maps; source line ranges stay conservatively inclusive."""
from hybrid_rag.schemas.document import SourceSpan


def slice_spans(spans: list[SourceSpan], start: int, end: int) -> list[SourceSpan]:
    return [span.model_copy(update={"start": max(span.start, start) - start,
                                    "end": min(span.end, end) - start})
            for span in spans if span.start < end and span.end > start]
