"""Per-knowledge-base settings: validation, reset, and re-ingestion."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.config import DEFAULT_KB_SETTINGS
from app.db import db
from app.models import KnowledgeBase


@pytest.fixture()
def account(app, client, signup):
    signup()
    with app.app_context():
        kb = db.session.scalar(select(KnowledgeBase))
        kb_id = kb.id
    return client, db.session.get(KnowledgeBase, kb_id)


def test_settings_page_renders_defaults(account):
    client, kb = account
    response = client.get(f"/knowledge-bases/{kb.id}/settings")
    assert response.status_code == 200
    assert b"Advanced" in response.data or b"How documents are split" in response.data
    assert b"Recursive" in response.data


def test_invalid_overlap_is_rejected(account):
    client, kb = account
    response = client.post(
        f"/knowledge-bases/{kb.id}/settings",
        data={
            "chunk_strategy": "recursive",
            "chunk_size": "400",
            "chunk_overlap": "800",
            "retrieval_mode": "hybrid",
            "top_k": "5",
            "candidates": "40",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Overlap must be between 0 and 500" in response.data
    db.session.expire_all()
    assert db.session.get(KnowledgeBase, kb.id).chunk_size == DEFAULT_KB_SETTINGS["chunk_size"]


def test_reset_restores_defaults(account):
    client, kb = account
    client.post(
        f"/knowledge-bases/{kb.id}/settings",
        data={
            "chunk_strategy": "fixed",
            "chunk_size": "300",
            "chunk_overlap": "30",
            "retrieval_mode": "vector",
            "top_k": "3",
            "candidates": "20",
        },
        follow_redirects=True,
    )
    db.session.expire_all()
    assert db.session.get(KnowledgeBase, kb.id).chunk_strategy == "fixed"

    response = client.post(f"/knowledge-bases/{kb.id}/settings/reset", follow_redirects=True)
    assert response.status_code == 200
    db.session.expire_all()
    updated = db.session.get(KnowledgeBase, kb.id)
    assert updated.chunk_strategy == DEFAULT_KB_SETTINGS["chunk_strategy"]
    assert updated.chunk_size == DEFAULT_KB_SETTINGS["chunk_size"]
    assert updated.retrieval_mode == DEFAULT_KB_SETTINGS["retrieval_mode"]
