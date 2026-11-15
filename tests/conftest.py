"""Test fixtures: a real Postgres database, a Flask test client, no CSRF friction."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, text

os.environ.setdefault("EMBEDDINGS_PROVIDER", "hash")
os.environ.setdefault("EMBEDDING_DIM", "384")
os.environ.setdefault("RERANK_PROVIDER", "none")
os.environ.setdefault("TESTING", "1")

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5432/rag_test"
)
TEST_UPLOAD_DIR = tempfile.mkdtemp(prefix="rag-test-uploads-")

from app import create_app  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import db  # noqa: E402
from app.models import Base  # noqa: E402


@pytest.fixture(scope="session")
def app():
    settings = replace(
        get_settings(),
        database_url=TEST_DATABASE_URL,
        testing=True,
        embeddings_provider="hash",
        embedding_model="hash-bow-v1",
        rerank_provider="none",
        llm_api_key=None,
        upload_dir=TEST_UPLOAD_DIR,
    )
    application = create_app(
        {
            "TESTING": True,
            "CSRF_ENABLED": False,
            "DATABASE_URL": TEST_DATABASE_URL,
            "SETTINGS": settings,
        }
    )
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()
    yield application
    db.dispose()


@pytest.fixture(autouse=True)
def clean_tables(app):
    yield
    # The test body may have used db.session outside a request context, leaving an
    # open transaction that would block TRUNCATE. Close it first.
    db.remove()
    # Use an explicit engine: the global db may have been rebound to another
    # database by an imported module, and TRUNCATE must only ever touch rag_test.
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE users, knowledge_bases, api_tokens, documents, chunks, jobs "
                "RESTART IDENTITY CASCADE"
            )
        )
    engine.dispose()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def signup(client):
    def _signup(email: str = "a@example.com", password: str = "correct-horse"):
        response = client.post(
            "/signup",
            data={"email": email, "password": password, "display_name": "Test"},
        )
        return response

    return _signup
