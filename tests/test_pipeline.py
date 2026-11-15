"""Upload → ingest → retrieve, including tenant isolation."""

from __future__ import annotations

import io

import pytest
from sqlalchemy import select

from app.db import db
from app.models import Chunk, Document, KnowledgeBase, User
from ingestion.pipeline import drain_queue
from retrieval import retrieve

DOC = """# Handbook

## Refunds

Refunds are processed within five business days. EU customers can withdraw within
fourteen days. Shipping fees are not refunded unless the item was faulty.

## Invoices

Invoices are issued monthly and late payments incur interest.

## Security

Customer data is encrypted at rest with AES-256.
"""


def login(client, email="a@example.com", password="correct-horse"):
    client.post("/login", data={"email": email, "password": password})


def upload(client, kb, name="handbook.md", content=None, content_type="text/markdown"):
    content = DOC.encode() if content is None else content
    return client.post(
        f"/knowledge-bases/{kb.id}/documents",
        data={"files": (io.BytesIO(content), name, content_type)},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


@pytest.fixture()
def account(app, client, signup):
    signup()
    login(client)
    with app.app_context():
        kb = db.session.scalar(select(KnowledgeBase))
        assert kb is not None
        kb_id = kb.id
    kb = db.session.get(KnowledgeBase, kb_id)
    return client, kb


def ready_document(app, kb, filename="handbook.md") -> Document:
    with app.app_context():
        drain_queue()
        db.session.expire_all()
        document = db.session.scalar(
            select(Document).where(
                Document.knowledge_base_id == kb.id, Document.filename == filename
            )
        )
    assert document is not None
    return document


def test_markdown_upload_is_parsed_and_embedded(app, account):
    client, kb = account
    response = upload(client, kb)
    assert response.status_code == 200

    document = ready_document(app, kb)
    assert document.status == "ready"
    assert document.chunk_count >= 3
    assert document.error is None
    assert {entry["title"] for entry in document.outline} >= {"Refunds", "Invoices", "Security"}

    chunks = db.session.scalars(select(Chunk).where(Chunk.document_id == document.id)).all()
    assert len(chunks) == document.chunk_count
    assert all(chunk.embedding is not None for chunk in chunks)
    assert all(chunk.knowledge_base_id == kb.id for chunk in chunks)
    assert any("five business days" in chunk.text for chunk in chunks)


def test_rejects_wrong_extension_and_bad_pdf(account):
    client, kb = account
    bad_type = upload(client, kb, name="notes.exe", content=b"MZ")
    assert bad_type.status_code == 200
    assert b"unsupported type" in bad_type.data

    bad_pdf = upload(client, kb, name="scan.pdf", content=b"not a pdf", content_type="application/pdf")
    assert b"does not look like a PDF" in bad_pdf.data
    assert db.session.scalar(select(Document)) is None


def test_duplicate_upload_is_skipped_but_can_be_reprocessed(app, account):
    client, kb = account
    upload(client, kb)
    ready_document(app, kb)

    again = upload(client, kb)
    assert b"already in this knowledge base" in again.data
    assert len(db.session.scalars(select(Document)).all()) == 1

    document = ready_document(app, kb)
    reprocess = client.post(f"/knowledge-bases/{kb.id}/documents/{document.id}/reprocess")
    assert reprocess.status_code == 302
    ready_document(app, kb)


def test_empty_text_file_fails_with_clear_message(app, account):
    client, kb = account
    upload(client, kb, name="empty.txt", content=b"   \n\n  ", content_type="text/plain")
    document = ready_document(app, kb, "empty.txt")
    assert document.status == "failed"
    assert document.error
    assert "empty" in document.error.lower()


def test_delete_document_removes_chunks_and_file(app, account):
    client, kb = account
    upload(client, kb)
    document = ready_document(app, kb)
    chunk_ids = [chunk.id for chunk in db.session.scalars(select(Chunk))]

    response = client.post(f"/knowledge-bases/{kb.id}/documents/{document.id}/delete")
    assert response.status_code == 302
    assert db.session.scalar(select(Document)) is None
    assert db.session.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all() == []


def test_vector_search_returns_cited_chunks(app, account):
    client, kb = account
    upload(client, kb)
    ready_document(app, kb)

    hits = retrieve(kb, "how long do refunds take?", top_k=3)
    assert hits
    best = hits[0]
    assert best.filename == "handbook.md"
    assert "Refunds" in (best.heading_path or "")
    assert "five business days" in best.text
    assert best.citation.startswith("handbook.md")


def test_one_user_cannot_retrieve_another_users_chunks(app, client, signup):
    signup("a@example.com")
    login(client, "a@example.com")
    with app.app_context():
        kb_a = db.session.scalar(
            select(KnowledgeBase).join(User).where(User.email == "a@example.com")
        )
    upload(client, kb_a)
    doc_a = ready_document(app, kb_a)
    chunk_a = db.session.scalar(select(Chunk).where(Chunk.document_id == doc_a.id))

    client.post("/logout")
    signup("b@example.com")
    login(client, "b@example.com")
    with app.app_context():
        kb_b = db.session.scalar(
            select(KnowledgeBase).join(User).where(User.email == "b@example.com")
        )
    # B's knowledge base contains nothing, so a query returns nothing even though
    # A's chunk has a near-identical embedding.
    with app.app_context():
        assert retrieve(kb_b, "how long do refunds take?", top_k=5) == []
    assert db.session.scalar(select(Chunk).where(Chunk.knowledge_base_id == kb_b.id)) is None

    # Direct navigation to A's data is a 404, not a 403 (no existence leak).
    assert client.get(f"/knowledge-bases/{kb_a.id}").status_code == 404
    assert client.get(f"/knowledge-bases/{kb_a.id}/documents/{doc_a.id}").status_code == 404
    assert client.get(f"/knowledge-bases/{kb_b.id}/documents/{doc_a.id}").status_code == 404
    assert chunk_a.knowledge_base_id == kb_a.id


def test_upload_requires_ownership(app, client, signup):
    signup("a@example.com")
    login(client, "a@example.com")
    with app.app_context():
        kb_a = db.session.scalar(
            select(KnowledgeBase).join(User).where(User.email == "a@example.com")
        )
    client.post("/logout")
    signup("b@example.com")
    login(client, "b@example.com")

    response = client.post(
        f"/knowledge-bases/{kb_a.id}/documents",
        data={"files": (io.BytesIO(DOC.encode()), "stolen.md", "text/markdown")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 404
    assert db.session.scalar(select(Document)) is None


def test_pdf_ingestion_keeps_pages_and_headings(app, account):
    import pymupdf

    client, kb = account
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Refund Policy", fontsize=24)
    page.insert_text((72, 120), "Refunds are processed within five business days.", fontsize=11)
    page = pdf.new_page()
    page.insert_text((72, 72), "Shipping", fontsize=18)
    page.insert_text((72, 120), "Standard shipping takes three to five days.", fontsize=11)
    data = pdf.tobytes()
    pdf.close()

    upload(client, kb, name="policies.pdf", content=data, content_type="application/pdf")
    document = ready_document(app, kb, "policies.pdf")
    assert document.status == "ready"
    assert document.page_count == 2

    chunks = db.session.scalars(select(Chunk).where(Chunk.document_id == document.id)).all()
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert any("Refund Policy" in (chunk.heading_path or "") for chunk in chunks)
