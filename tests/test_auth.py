"""Accounts, sessions and the tenant boundary."""

from __future__ import annotations

from sqlalchemy import select

from app.db import db
from app.models import KnowledgeBase, User


def test_signup_creates_user_and_starter_kb(client, signup):
    response = signup("new@example.com")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/knowledge-bases/")

    user = db.session.scalar(select(User).where(User.email == "new@example.com"))
    assert user is not None
    assert user.password_hash != "correct-horse"
    assert user.password_hash.startswith(("scrypt:", "pbkdf2:"))

    kbs = list(db.session.scalars(select(KnowledgeBase).where(KnowledgeBase.user_id == user.id)))
    assert len(kbs) == 1
    assert kbs[0].chunk_strategy == "recursive"
    assert kbs[0].retrieval_mode == "hybrid"
    assert kbs[0].top_k == 5


def test_signup_rejects_bad_email_and_short_password(client):
    bad_email = client.post("/signup", data={"email": "nope", "password": "long-enough"})
    assert bad_email.status_code == 400
    short = client.post("/signup", data={"email": "x@example.com", "password": "short"})
    assert short.status_code == 400


def test_signup_normalises_email_and_rejects_duplicates(client):
    assert client.post(
        "/signup", data={"email": "  Mixed@Example.COM ", "password": "long-enough"}
    ).status_code == 302
    duplicate = client.post(
        "/signup", data={"email": "mixed@example.com", "password": "long-enough"}
    )
    assert duplicate.status_code == 409
    assert db.session.scalar(select(User).where(User.email == "mixed@example.com")) is not None


def test_login_logout_roundtrip(client, signup):
    signup("login@example.com", "correct-horse")
    client.post("/logout")

    anonymous = client.get("/knowledge-bases/")
    assert anonymous.status_code == 302
    assert "/login" in anonymous.headers["Location"]

    wrong = client.post("/login", data={"email": "login@example.com", "password": "wrong-pass"})
    assert wrong.status_code == 401

    ok = client.post("/login", data={"email": "LOGIN@example.com", "password": "correct-horse"})
    assert ok.status_code == 302
    assert client.get("/knowledge-bases/").status_code == 200


def test_login_next_is_not_an_open_redirect(client, signup):
    signup("next@example.com", "correct-horse")
    client.post("/logout")
    evil = client.post(
        "/login",
        data={"email": "next@example.com", "password": "correct-horse", "next": "//evil.test/x"},
    )
    assert evil.status_code == 302
    assert "evil.test" not in evil.headers["Location"]


def test_one_user_cannot_open_another_users_knowledge_base(app, client, signup):
    signup("owner@example.com")
    owner_kb = db.session.scalar(select(KnowledgeBase))
    assert owner_kb is not None

    other = app.test_client()
    other.post("/signup", data={"email": "intruder@example.com", "password": "correct-horse"})
    response = other.get(f"/knowledge-bases/{owner_kb.id}")
    assert response.status_code == 404  # not 403: existence is not leaked


def test_csrf_is_enforced_when_enabled(app, client):
    app.config["CSRF_ENABLED"] = True
    try:
        rejected = client.post(
            "/signup", data={"email": "csrf@example.com", "password": "correct-horse"}
        )
        assert rejected.status_code == 400
    finally:
        app.config["CSRF_ENABLED"] = False


def test_create_knowledge_base_scoped_to_user(client, signup):
    signup("maker@example.com")
    response = client.post(
        "/knowledge-bases/", data={"name": "Manuals", "description": "Product docs"}
    )
    assert response.status_code == 302
    kb = db.session.scalar(select(KnowledgeBase).where(KnowledgeBase.name == "Manuals"))
    assert kb is not None
    assert kb.description == "Product docs"
    assert kb.embedding_dim == 384
