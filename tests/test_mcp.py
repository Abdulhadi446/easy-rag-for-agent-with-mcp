"""Token lifecycle and MCP bearer enforcement."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from starlette.testclient import TestClient

from app.db import db
from app.models import ApiToken, KnowledgeBase
from app.security import hash_token
from mcp_server import server as mcp_server


@pytest.fixture()
def account(app, client, signup):
    signup()
    with app.app_context():
        kb = db.session.scalar(select(KnowledgeBase))
        kb_id = kb.id
    return client, db.session.get(KnowledgeBase, kb_id)


def _create_token(client, kb, name="agent"):
    page = client.get(f"/knowledge-bases/{kb.id}/tokens")
    assert page.status_code == 200
    response = client.post(
        f"/knowledge-bases/{kb.id}/tokens", data={"name": name}, follow_redirects=True
    )
    assert response.status_code == 200
    return response


def test_token_is_shown_once_and_stored_hashed(app, account):
    client, kb = account
    response = _create_token(client, kb)
    html = response.data.decode()
    assert "New token · copy it now" in html
    plaintext = html.split('id="new-token">')[1].split("<")[0]
    assert plaintext.startswith("rag_")

    with app.app_context():
        token = db.session.scalar(select(ApiToken).where(ApiToken.knowledge_base_id == kb.id))
        assert token is not None
        assert token.prefix.startswith("rag_")
        assert token.token_hash == hash_token(plaintext)
        assert "rag_" not in token.token_hash

    # A second visit never shows the plaintext again.
    again = client.get(f"/knowledge-bases/{kb.id}/tokens")
    assert "New token · copy it now" not in again.data.decode()
    assert plaintext not in again.data.decode()


def test_revoke_and_rotate(app, account):
    client, kb = account
    _create_token(client, kb)
    with app.app_context():
        token = db.session.scalar(select(ApiToken))
        old_hash = token.token_hash
        token_id = token.id

    revoked = client.post(
        f"/knowledge-bases/{kb.id}/tokens/{token_id}/revoke", follow_redirects=True
    )
    assert revoked.status_code == 200
    with app.app_context():
        assert db.session.get(ApiToken, token_id).revoked_at is not None

    # Rotating a revoked token still works and produces a different credential.
    rotated = client.post(
        f"/knowledge-bases/{kb.id}/tokens/{token_id}/rotate", follow_redirects=True
    )
    assert rotated.status_code == 200
    with app.app_context():
        active = db.session.scalar(
            select(ApiToken).where(
                ApiToken.knowledge_base_id == kb.id, ApiToken.revoked_at.is_(None)
            )
        )
        assert active is not None
        assert active.token_hash != old_hash


def test_bearer_middleware_rejects_missing_and_bad_tokens(monkeypatch):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    kb_id = uuid.uuid4()
    monkeypatch.setattr(
        mcp_server, "_resolve_token", lambda token: kb_id if token == "good" else None
    )
    client = TestClient(mcp_server.BearerAuth(app))
    assert client.post("/mcp").status_code == 401
    assert client.post("/mcp", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = client.post("/mcp", headers={"Authorization": "Bearer good"})
    assert ok.status_code == 200
    assert ok.text == "ok"


def test_mcp_tools_are_scoped_to_the_bound_knowledge_base(app, account):
    from app.models import Chunk, Document, DocumentStatus
    from mcp_server import service

    _, kb = account
    other = KnowledgeBase(
        user_id=kb.user_id,
        name="Other",
        embedding_model=kb.embedding_model,
        embedding_dim=kb.embedding_dim,
    )
    db.session.add(other)
    db.session.commit()
    document = Document(
        knowledge_base_id=other.id,
        filename="secret.md",
        content_type="text/markdown",
        size_bytes=10,
        sha256="0" * 64,
        status=DocumentStatus.READY,
        chunk_count=1,
    )
    db.session.add(document)
    db.session.flush()
    chunk = Chunk(
        document_id=document.id,
        knowledge_base_id=other.id,
        text="classified",
        chunk_index=0,
        token_count=2,
    )
    db.session.add(chunk)
    db.session.commit()

    with pytest.raises(LookupError):
        service.get_chunk(kb, str(chunk.id))
    with pytest.raises(LookupError):
        service.document_outline(kb, str(document.id))
    assert service.list_documents(kb)["document_count"] == 0
