"""Cosine similarity over the HNSW index."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from app.models import Chunk, Document
from embeddings import get_embeddings
from retrieval.base import SearchHit, hit_from_row


def vector_search(
    kb_id: uuid.UUID,
    query: str,
    *,
    limit: int = 40,
    session: Session,
    embedding: list[float] | None = None,
) -> list[SearchHit]:
    if not query.strip():
        return []
    vector = embedding if embedding is not None else get_embeddings().embed_query(query)
    parent = aliased(Chunk)
    distance = Chunk.embedding.cosine_distance(vector)
    rows = session.execute(
        select(Chunk, Document.filename, parent.text.label("parent_text"), distance.label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .outerjoin(parent, parent.id == Chunk.parent_id)
        .where(
            Chunk.knowledge_base_id == kb_id,
            Chunk.embedding.is_not(None),
            Document.status == "ready",
        )
        .order_by(distance)
        .limit(max(1, min(limit, 200)))
    ).all()
    hits: list[SearchHit] = []
    for chunk, filename, parent_text, dist in rows:
        similarity = 1.0 - float(dist)
        hits.append(
            hit_from_row(
                chunk,
                filename,
                parent_text,
                score=similarity,
                raw_score=similarity,
                source="vector",
            )
        )
    return hits
