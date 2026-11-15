"""Shared result type and row materialisation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    text: str
    score: float
    page_number: int | None = None
    heading_path: str | None = None
    chunk_index: int = 0
    parent_id: uuid.UUID | None = None
    window_text: str | None = None
    source: str = "vector"
    raw_score: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        location = f", page {self.page_number}" if self.page_number else ""
        heading = f" · {self.heading_path}" if self.heading_path else ""
        return f"{self.filename}{location}{heading}"

    def as_dict(self, include_text: bool = True) -> dict[str, Any]:
        payload = {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "filename": self.filename,
            "score": round(float(self.score), 4),
            "page_number": self.page_number,
            "heading_path": self.heading_path,
            "citation": self.citation,
        }
        if include_text:
            payload["text"] = self.text
        return payload


def hit_from_row(
    chunk: Any,
    filename: str,
    parent_text: str | None,
    *,
    score: float,
    source: str,
    raw_score: float | None = None,
) -> SearchHit:
    display_text = (parent_text or chunk.window_text or chunk.text or "").strip()
    return SearchHit(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        filename=filename,
        text=display_text,
        score=float(score),
        page_number=chunk.page_number,
        heading_path=chunk.heading_path,
        chunk_index=chunk.chunk_index,
        parent_id=chunk.parent_id,
        window_text=chunk.window_text,
        source=source,
        raw_score=raw_score,
        extra=chunk.extra or {},
    )


def dedupe_by_parent(hits: list[SearchHit]) -> list[SearchHit]:
    """Keep the best child per parent so one parent cannot occupy several slots."""
    seen: set[uuid.UUID] = set()
    result: list[SearchHit] = []
    for hit in sorted(hits, key=lambda item: item.score, reverse=True):
        key = hit.parent_id or hit.chunk_id
        if key in seen:
            continue
        seen.add(key)
        result.append(hit)
    return result


def normalise_scores(hits: list[SearchHit]) -> list[SearchHit]:
    """Show relative relevance for rank-fusion scores; keep cosine scores intact."""
    if not hits:
        return hits
    if hits[0].source in {"vector", "rerank"}:
        for hit in hits:
            hit.score = max(0.0, min(1.0, hit.score))
        return hits
    best = max(hit.score for hit in hits) or 1.0
    for hit in hits:
        hit.score = max(0.0, min(1.0, hit.score / best))
    return hits
