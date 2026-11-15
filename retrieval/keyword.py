"""Postgres full-text search ranked with ts_rank_cd."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, aliased

from app.models import Chunk, Document
from retrieval.base import SearchHit, hit_from_row

log = logging.getLogger(__name__)


def keyword_search(
    kb_id: uuid.UUID,
    query: str,
    *,
    limit: int = 40,
    session: Session,
) -> list[SearchHit]:
    if not query.strip():
        return []

    parent = aliased(Chunk)
    for builder in (func.websearch_to_tsquery, func.plainto_tsquery):
        tsquery = builder("english", query)
        rank = func.ts_rank_cd(Chunk.content_tsv, tsquery)
        stmt = (
            select(Chunk, Document.filename, parent.text.label("parent_text"), rank.label("rank"))
            .join(Document, Document.id == Chunk.document_id)
            .outerjoin(parent, parent.id == Chunk.parent_id)
            .where(
                Chunk.knowledge_base_id == kb_id,
                Chunk.content_tsv.op("@@")(tsquery),
                Document.status == "ready",
            )
            .order_by(rank.desc())
            .limit(max(1, min(limit, 200)))
        )
        try:
            with session.begin_nested():
                rows = session.execute(stmt).all()
        except SQLAlchemyError as exc:
            log.debug("keyword query failed with %s: %s", builder, exc)
            continue

        if not rows:
            return []
        best = max(float(row.rank) for row in rows) or 1.0
        return [
            hit_from_row(
                chunk,
                filename,
                parent_text,
                score=float(rank_value) / best,
                raw_score=float(rank_value),
                source="keyword",
            )
            for chunk, filename, parent_text, rank_value in rows
        ]
    return []
