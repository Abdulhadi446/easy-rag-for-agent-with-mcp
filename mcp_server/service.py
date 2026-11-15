"""Shared retrieval operations used by the MCP server and the web chat.

Every function takes an already tenant-scoped ``KnowledgeBase``. Nothing here looks
up a knowledge base by an id supplied by a model.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from app.db import db
from app.models import Chunk, Document, KnowledgeBase
from retrieval import SearchHit, retrieve

MAX_TEXT_CHARS = 6000


def _clip(text: str | None, limit: int = MAX_TEXT_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n…[truncated {len(text) - limit} characters]"


def _chunk_payload(chunk: Chunk, document: Document, parent: Chunk | None = None) -> dict[str, Any]:
    text = (parent.text if parent is not None else None) or chunk.window_text or chunk.text
    return {
        "chunk_id": str(chunk.id),
        "document_id": str(document.id),
        "filename": document.filename,
        "page_number": chunk.page_number,
        "heading_path": chunk.heading_path,
        "chunk_index": chunk.chunk_index,
        "parent_id": str(chunk.parent_id) if chunk.parent_id else None,
        "text": _clip(text),
        "citation": _citation(document.filename, chunk.page_number, chunk.heading_path),
    }


def _citation(filename: str, page: int | None, heading: str | None) -> str:
    bits = [filename]
    if page:
        bits.append(f"page {page}")
    if heading:
        bits.append(heading)
    return ", ".join(bits)


def search(kb: KnowledgeBase, query: str, top_k: int = 5, mode: str | None = None) -> dict[str, Any]:
    hits = retrieve(kb, query, top_k=top_k, mode=mode)
    return {
        "query": query,
        "mode_used": (mode or kb.retrieval_mode or "hybrid"),
        "result_count": len(hits),
        "results": [_hit_payload(hit) for hit in hits],
        "note": (
            "Cite filename and page number in your answer. If these passages do not answer the "
            "question, say so and try a different query or drill into a document."
        )
        if hits
        else "No passages matched. Try different wording, or list_documents() to see what exists.",
    }


def _hit_payload(hit: SearchHit) -> dict[str, Any]:
    payload = hit.as_dict()
    payload["text"] = _clip(hit.text)
    payload["parent_id"] = str(hit.parent_id) if hit.parent_id else None
    return payload


def list_documents(kb: KnowledgeBase) -> dict[str, Any]:
    documents = db.session.scalars(
        select(Document)
        .where(Document.knowledge_base_id == kb.id)
        .order_by(Document.created_at.desc())
    ).all()
    return {
        "knowledge_base": kb.name,
        "document_count": len(documents),
        "documents": [
            {
                "document_id": str(doc.id),
                "filename": doc.filename,
                "status": doc.status,
                "page_count": doc.page_count,
                "chunk_count": doc.chunk_count,
                "uploaded_at": doc.created_at.isoformat() if doc.created_at else None,
                "error": doc.error,
            }
            for doc in documents
        ],
        "note": "Only documents with status 'ready' are searchable.",
    }


def get_chunk(kb: KnowledgeBase, chunk_id: str) -> dict[str, Any]:
    chunk, document = _load_chunk(kb, chunk_id)
    parent = db.session.get(Chunk, chunk.parent_id) if chunk.parent_id else None
    payload = _chunk_payload(chunk, document, parent)
    if chunk.window_text:
        payload["window_text"] = _clip(chunk.window_text)
    return payload


def surrounding_context(
    kb: KnowledgeBase, chunk_id: str, before: int = 2, after: int = 2
) -> dict[str, Any]:
    chunk, document = _load_chunk(kb, chunk_id)
    before = max(0, min(int(before), 10))
    after = max(0, min(int(after), 10))
    rows = db.session.scalars(
        select(Chunk)
        .where(
            Chunk.document_id == document.id,
            Chunk.knowledge_base_id == kb.id,
            Chunk.chunk_index >= chunk.chunk_index - before,
            Chunk.chunk_index <= chunk.chunk_index + after,
        )
        .order_by(Chunk.chunk_index)
    ).all()
    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "target_chunk_id": str(chunk.id),
        "range": {"before": before, "after": after, "returned": len(rows)},
        "parts": [
            {
                "chunk_id": str(row.id),
                "chunk_index": row.chunk_index,
                "page_number": row.page_number,
                "heading_path": row.heading_path,
                "is_target": row.id == chunk.id,
                "text": _clip(row.text, 2000),
            }
            for row in rows
        ],
        "text": "\n\n".join(
            f"[{row.chunk_index}] {row.text.strip()}" for row in rows if row.text.strip()
        )[:MAX_TEXT_CHARS],
    }


def document_outline(kb: KnowledgeBase, document_id: str) -> dict[str, Any]:
    document = db.session.scalar(
        select(Document).where(Document.id == _uuid(document_id), Document.knowledge_base_id == kb.id)
    )
    if document is None:
        raise LookupError("Document not found in this knowledge base.")
    outline = document.outline or []
    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "status": document.status,
        "page_count": document.page_count,
        "outline": outline,
        "note": (
            "Headings without page numbers come from Markdown or plain text. Use search_documents() "
            "with heading wording to jump into a section."
        )
        if outline
        else "No headings were detected in this document.",
    }


def _load_chunk(kb: KnowledgeBase, chunk_id: str) -> tuple[Chunk, Document]:
    chunk = db.session.scalar(
        select(Chunk).where(Chunk.id == _uuid(chunk_id), Chunk.knowledge_base_id == kb.id)
    )
    if chunk is None:
        raise LookupError("Chunk not found in this knowledge base.")
    document = db.session.scalar(
        select(Document).where(Document.id == chunk.document_id, Document.knowledge_base_id == kb.id)
    )
    if document is None:
        raise LookupError("Chunk not found in this knowledge base.")
    return chunk, document


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise LookupError("Invalid id: expected a UUID.") from exc
