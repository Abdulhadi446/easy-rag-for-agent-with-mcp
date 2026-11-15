"""Ingestion pipeline: parse → chunk → embed → store.

Runs outside the request cycle. Failures are recorded on the document and job so
the UI can explain what happened without leaking file contents.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy import text as sql_text

from app.db import db
from app.models import Chunk, Document, DocumentStatus, Job, JobStatus, KnowledgeBase
from embeddings import EmbeddingError, build_embeddings
from ingestion.chunkers import ChunkingConfig, make_chunker
from ingestion.parsers import EmptyDocumentError, ParseError, parse_upload

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


class PermanentIngestError(RuntimeError):
    """Retrying will not help (bad file, unsupported strategy)."""


def enqueue_ingest(document_id: uuid.UUID, *, kind: str = "ingest", delay_seconds: int = 0) -> Job:
    session = db.session
    document = session.get(Document, document_id)
    if document is None:
        raise ValueError("Document not found")
    job = Job(
        document_id=document.id,
        knowledge_base_id=document.knowledge_base_id,
        kind=kind,
        status=JobStatus.QUEUED,
        run_after=datetime.now(UTC) + timedelta(seconds=delay_seconds),
    )
    document.status = DocumentStatus.QUEUED
    document.error = None
    session.add(job)
    session.commit()
    return job


def enqueue_knowledge_base(kb_id: uuid.UUID, *, kind: str = "reingest") -> int:
    """Queue every ready/failed document after a chunking-settings change."""
    session = db.session
    documents = session.scalars(
        select(Document).where(
            Document.knowledge_base_id == kb_id,
            Document.status.in_(
                [DocumentStatus.READY, DocumentStatus.FAILED, DocumentStatus.QUEUED]
            ),
        )
    ).all()
    now = datetime.now(UTC)
    queued = 0
    for document in documents:
        session.add(
            Job(
                document_id=document.id,
                kb_id=kb_id,
                kind=kind,
                status=JobStatus.QUEUED,
                run_after=now,
            )
        )
        document.status = DocumentStatus.QUEUED
        document.error = None
        queued += 1
    session.commit()
    return queued


def reset_stuck_jobs() -> int:
    """On boot, return jobs from a killed process to the queue."""
    session = db.session
    result = session.execute(
        sql_text(
            "UPDATE jobs SET status='queued', run_after=now() "
            "WHERE status='running'"
        )
    )
    session.commit()
    return result.rowcount or 0


def claim_job() -> uuid.UUID | None:
    """Atomically take one due job. Safe with several worker threads/processes."""
    session = db.session
    try:
        row = session.execute(
            sql_text(
                """
                SELECT id FROM jobs
                WHERE status = 'queued' AND run_after <= now()
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """
            )
        ).first()
        if row is None:
            session.rollback()
            return None
        job_id = row[0]
        session.execute(
            sql_text(
                "UPDATE jobs SET status='running', attempts=attempts+1, started_at=now() "
                "WHERE id=:id"
            ),
            {"id": job_id},
        )
        session.commit()
        return job_id
    except Exception:
        session.rollback()
        raise


def _parse_document(document: Document) -> tuple:
    path = (document.meta or {}).get("storage_path")
    if not path:
        raise PermanentIngestError("The uploaded file is missing from storage.")
    try:
        return parse_upload(Path(path), document.filename), path
    except (ParseError, EmptyDocumentError) as exc:
        raise PermanentIngestError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise PermanentIngestError(
            "The uploaded file was removed before it could be processed. Upload it again."
        ) from exc


def _embedding_texts(drafts) -> tuple[list[int], list[str]]:
    indices: list[int] = []
    texts: list[str] = []
    for index, draft in enumerate(drafts):
        if draft.role == "parent":
            continue
        indices.append(index)
        texts.append(draft.text)
    return indices, texts


def ingest_document(document_id: uuid.UUID) -> None:
    """Process one document. Caller is responsible for job bookkeeping."""
    session = db.session
    document = session.get(Document, document_id)
    if document is None:
        log.warning("ingest skipped: document %s no longer exists", document_id)
        return
    kb = session.get(KnowledgeBase, document.knowledge_base_id)
    if kb is None:
        log.warning("ingest skipped: knowledge base for %s no longer exists", document_id)
        return

    document.status = DocumentStatus.PROCESSING
    document.error = None
    session.commit()

    try:
        from flask import current_app

        parsed, storage = _parse_document(document)
        config = ChunkingConfig.from_kb(kb)
        provider = build_embeddings(current_app.config["SETTINGS"])
        embed_fn = provider.embed_documents if kb.chunk_strategy == "semantic" else None

        chunker = make_chunker(kb.chunk_strategy, config, embed_fn=embed_fn)
        drafts = chunker.split(parsed.blocks)
        if not drafts:
            raise PermanentIngestError(
                "No text could be chunked from this document. It may contain only formatting."
            )

        indices, texts = _embedding_texts(drafts)
        vectors: dict[int, list[float]] = {}
        if texts:
            try:
                embeddings_list = provider.embed_documents(texts)
            except EmbeddingError as exc:
                # Provider problems are often transient (rate limits), so retry.
                raise RuntimeError(str(exc)) from exc
            if len(embeddings_list) != len(texts):
                raise RuntimeError("Embedding provider returned the wrong number of vectors")
            for index, vector in zip(indices, embeddings_list, strict=False):
                vectors[index] = vector

        # Replace any previous chunks atomically.
        session.execute(delete(Chunk).where(Chunk.document_id == document.id))
        session.flush()

        id_by_index: dict[int, uuid.UUID] = {}
        for index, draft in enumerate(drafts):
            chunk = Chunk(
                document_id=document.id,
                knowledge_base_id=kb.id,
                text=draft.text,
                page_number=draft.page_number,
                heading_path=draft.heading_path,
                chunk_index=index,
                token_count=draft.token_count,
                window_text=draft.window_text,
                extra=draft.extra or {},
                embedding=vectors.get(index),
            )
            if draft.parent_index is not None:
                chunk.parent_id = id_by_index.get(draft.parent_index)
            session.add(chunk)
            session.flush()
            id_by_index[index] = chunk.id

        document.status = DocumentStatus.READY
        document.error = None
        document.page_count = parsed.page_count
        document.chunk_count = len(drafts)
        document.outline = parsed.outline
        document.meta = {
            **(document.meta or {}),
            "format": parsed.meta.get("format"),
            "char_count": parsed.char_count,
            "strategy": kb.chunk_strategy,
            "embedding_model": kb.embedding_model,
        }
        document.processed_at = datetime.now(UTC)
        session.commit()
        log.info(
            "ingested document %s (%s chunks, strategy=%s)",
            document.id,
            len(drafts),
            kb.chunk_strategy,
        )
    except PermanentIngestError as exc:
        session.rollback()
        document = session.get(Document, document_id)
        if document is not None:
            document.status = DocumentStatus.FAILED
            document.error = str(exc)[:1000]
            document.processed_at = datetime.now(UTC)
            session.commit()
        raise
    except Exception as exc:
        session.rollback()
        document = session.get(Document, document_id)
        if document is not None:
            document.status = DocumentStatus.FAILED
            document.error = (str(exc) or "Processing failed while embedding or storing chunks.")[
                :500
            ]
            document.processed_at = datetime.now(UTC)
            session.commit()
        raise


def finish_job(job_id: uuid.UUID, *, error: str | None = None) -> None:
    session = db.session
    job = session.get(Job, job_id)
    if job is None:
        return
    if error is None:
        job.status = JobStatus.DONE
        job.error = None
        job.finished_at = datetime.now(UTC)
    else:
        permanent = error.startswith("Permanent:")
        message = error.removeprefix("Permanent:")
        job.error = message[:1000]
        if permanent or job.attempts >= MAX_ATTEMPTS:
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now(UTC)
            document = session.get(Document, job.document_id) if job.document_id else None
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.error = message[:1000]
        else:
            job.status = JobStatus.QUEUED
            job.run_after = datetime.now(UTC) + timedelta(seconds=5 * (2**job.attempts))
            document = session.get(Document, job.document_id) if job.document_id else None
            if document is not None:
                document.status = DocumentStatus.QUEUED
    session.commit()


def process_next_job() -> bool:
    """Claim and run one job. Returns False when the queue is empty."""
    job_id = claim_job()
    if job_id is None:
        return False
    session = db.session
    job = session.get(Job, job_id)
    if job is None or job.document_id is None:
        finish_job(job_id)
        return True
    try:
        ingest_document(job.document_id)
    except PermanentIngestError as exc:
        finish_job(job_id, error=f"Permanent:{exc}")
    except Exception as exc:  # noqa: BLE001
        log.exception("ingest job %s failed", job_id)
        finish_job(job_id, error=str(exc) or "Unknown ingestion error")
    else:
        finish_job(job_id)
    return True


def drain_queue(max_jobs: int = 20) -> int:
    """Synchronous processing helper for tests and the CLI."""
    done = 0
    while done < max_jobs and process_next_job():
        done += 1
    return done


def wait_for_document(document_id: uuid.UUID, timeout: float = 60.0) -> Document | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        db.session.expire_all()
        document = db.session.get(Document, document_id)
        if document is None:
            return None
        if document.status in {DocumentStatus.READY, DocumentStatus.FAILED}:
            return document
        if not process_next_job():
            time.sleep(0.2)
    return db.session.get(Document, document_id)
