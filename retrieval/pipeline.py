"""The one retrieval pipeline used by the web chat, the MCP server and evaluation."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.db import db
from app.models import KnowledgeBase
from embeddings import EmbeddingError, get_embeddings
from retrieval.base import SearchHit, dedupe_by_parent, normalise_scores
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.keyword import keyword_search
from retrieval.rerank import get_reranker
from retrieval.rewrite import rewrite_query
from retrieval.vector import vector_search

log = logging.getLogger(__name__)

VALID_MODES = {"vector", "keyword", "hybrid"}


class RetrievalError(RuntimeError):
    """Safe to show to the user (e.g. embedding model changed)."""


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def retrieve(
    kb: KnowledgeBase,
    query: str,
    *,
    top_k: int | None = None,
    mode: str | None = None,
    rerank: bool | None = None,
    rewrite: bool | None = None,
    session: Session | None = None,
) -> list[SearchHit]:
    """Retrieve passages for one knowledge base (already tenant-scoped by caller)."""
    query = (query or "").strip()
    if not query:
        return []
    session = session or db.session

    chosen_mode = (mode or kb.retrieval_mode or "hybrid").lower()
    if chosen_mode not in VALID_MODES:
        chosen_mode = "hybrid"

    limit = _clamp(top_k or kb.top_k, 1, 20)
    candidates = _clamp(max(kb.candidates, limit), 10, 100)

    provider = get_embeddings()
    if kb.embedding_model and provider.model_id != kb.embedding_model:
        raise RetrievalError(
            f"This knowledge base was built with the embedding model "
            f"{kb.embedding_model!r}, but the app is configured for {provider.model_id!r}. "
            "Restore EMBEDDING_MODEL or re-ingest the documents."
        )
    if provider.dim != kb.embedding_dim:
        raise RetrievalError(
            f"This knowledge base stores {kb.embedding_dim}-dim vectors, but the embedding "
            f"provider returns {provider.dim}. Restore the original model or re-ingest."
        )

    effective_query = query
    should_rewrite = kb.query_rewrite_enabled if rewrite is None else rewrite
    if should_rewrite:
        effective_query = rewrite_query(query)

    try:
        query_vector = provider.embed_query(effective_query)
    except EmbeddingError as exc:
        raise RetrievalError(str(exc)) from exc

    if chosen_mode == "vector":
        hits = vector_search(
            kb.id, effective_query, limit=candidates, session=session, embedding=query_vector
        )
    elif chosen_mode == "keyword":
        hits = keyword_search(kb.id, effective_query, limit=candidates, session=session)
    else:
        vector_hits = vector_search(
            kb.id, effective_query, limit=candidates, session=session, embedding=query_vector
        )
        keyword_hits = keyword_search(kb.id, effective_query, limit=candidates, session=session)
        hits = reciprocal_rank_fusion([vector_hits, keyword_hits])

    hits = dedupe_by_parent(hits)

    should_rerank = kb.rerank_enabled if rerank is None else rerank
    if should_rerank and hits:
        reranker = get_reranker()
        if reranker is not None:
            hits = reranker.rerank(effective_query, hits, top_k=limit)
        else:
            hits = hits[:limit]
    else:
        hits = hits[:limit]

    hits = normalise_scores(hits)
    for hit in hits:
        hit.extra = {**hit.extra, "query_used": effective_query}
    return hits
