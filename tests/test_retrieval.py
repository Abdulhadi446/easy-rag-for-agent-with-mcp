"""Unit tests for rank fusion and parent deduplication."""

from __future__ import annotations

import uuid

from retrieval.base import SearchHit, dedupe_by_parent
from retrieval.hybrid import reciprocal_rank_fusion


def hit(
    name: str,
    score: float = 1.0,
    parent: uuid.UUID | None = None,
    source: str = "vector",
    chunk_id: uuid.UUID | None = None,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=name,
        text=f"text for {name}",
        score=score,
        page_number=1,
        heading_path=None,
        chunk_index=0,
        parent_id=parent,
        window_text=None,
        source=source,
        raw_score=score,
    )


def test_rrf_prefers_documents_found_by_both_lists():
    shared_vector = hit("shared", source="vector")
    shared_keyword = hit("shared", source="keyword", chunk_id=shared_vector.chunk_id)
    vector_only = hit("vector-only", source="vector")
    keyword_only = hit("keyword-only", source="keyword")

    fused = reciprocal_rank_fusion(
        [[shared_vector, vector_only], [keyword_only, shared_keyword]], k=60
    )

    assert [item.filename for item in fused][:1] == ["shared"]
    assert set(item.filename for item in fused) == {"shared", "vector-only", "keyword-only"}
    assert fused[0].source == "hybrid"
    assert fused[0].score > fused[1].score


def test_rrf_scores_follow_rank_positions():
    first, second = hit("first"), hit("second")
    fused = reciprocal_rank_fusion([[first, second]], k=60)
    assert fused[0].filename == "first"
    assert fused[0].score > fused[1].score


def test_dedupe_by_parent_keeps_best_child():
    parent = uuid.uuid4()
    weak = hit("doc", score=0.3, parent=parent)
    strong = hit("doc", score=0.9, parent=parent)
    solo = hit("other", score=0.5)

    deduped = dedupe_by_parent([weak, strong, solo])
    assert len(deduped) == 2
    kept = next(item for item in deduped if item.parent_id == parent)
    assert kept.score == 0.9
