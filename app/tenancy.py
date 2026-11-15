"""Tenant helpers: the only sanctioned way to reach a knowledge base."""

from __future__ import annotations

import uuid

from flask import g, session
from sqlalchemy import select

from app.db import db
from app.models import KnowledgeBase, User


def load_current_user() -> User | None:
    if "user" in g:
        return g.user
    user_id = session.get("user_id")
    user = None
    if user_id:
        try:
            user = db.session.get(User, uuid.UUID(user_id))
        except (ValueError, TypeError):
            user = None
        if user is not None and not user.is_active:
            user = None
            session.clear()
    g.user = user
    return user


def get_knowledge_base_or_404(kb_id: str | uuid.UUID) -> KnowledgeBase:
    """Return the KB only when it belongs to the signed-in user."""
    from flask import abort

    user = g.get("user")
    if user is None:
        abort(401)
    try:
        kb_uuid = kb_id if isinstance(kb_id, uuid.UUID) else uuid.UUID(str(kb_id))
    except ValueError:
        abort(404)
    kb = db.session.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_uuid,
            KnowledgeBase.user_id == user.id,
        )
    )
    if kb is None:
        # 404, not 403: another tenant's knowledge base must not be enumerable.
        abort(404)
    return kb


def user_knowledge_bases(user: User) -> list[KnowledgeBase]:
    return list(
        db.session.scalars(
            select(KnowledgeBase)
            .where(KnowledgeBase.user_id == user.id)
            .order_by(KnowledgeBase.created_at.desc())
        )
    )


def get_document_or_404(kb: KnowledgeBase, document_id: str | uuid.UUID):
    """A document is only reachable through its own knowledge base."""
    from flask import abort

    from app.models import Document

    try:
        document_uuid = (
            document_id if isinstance(document_id, uuid.UUID) else uuid.UUID(str(document_id))
        )
    except ValueError:
        abort(404)
    document = db.session.scalar(
        select(Document).where(
            Document.id == document_uuid,
            Document.knowledge_base_id == kb.id,
        )
    )
    if document is None:
        abort(404)
    return document
