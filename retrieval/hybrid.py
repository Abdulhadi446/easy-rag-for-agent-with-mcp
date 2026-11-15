"""Reciprocal Rank Fusion: merge ranked lists without calibrating their scores."""

from __future__ import annotations

import uuid

from retrieval.base import SearchHit


def reciprocal_rank_fusion(
    result_lists: list[list[SearchHit]],
    *,
    k: int = 60,
) -> list[SearchHit]:
    scores: dict[uuid.UUID, float] = {}
    best: dict[uuid.UUID, SearchHit] = {}
    sources: dict[uuid.UUID, set[str]] = {}

    for results in result_lists:
        for rank, hit in enumerate(results):
            key = hit.parent_id or hit.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(key, set()).add(hit.source)
            current = best.get(key)
            if current is None or hit.score > current.score:
                best[key] = hit

    merged: list[SearchHit] = []
    for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        hit = best[key]
        hit.score = score
        hit.raw_score = score
        hit.source = "hybrid" if len(sources[key]) > 1 else next(iter(sources[key]))
        merged.append(hit)
    return merged
