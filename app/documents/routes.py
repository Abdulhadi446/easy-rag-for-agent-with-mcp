"""Document upload, status, deletion and reprocessing."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from sqlalchemy import select

from app.auth.decorators import login_required
from app.db import db
from app.documents.storage import remove_file, safe_display_name, storage_path
from app.models import Chunk, Document, DocumentStatus, Job, JobStatus
from app.tenancy import get_document_or_404, get_knowledge_base_or_404
from ingestion.pipeline import enqueue_ingest

log = logging.getLogger(__name__)

bp = Blueprint("documents", __name__, url_prefix="/knowledge-bases/<uuid:kb_id>/documents")

ALLOWED_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}
MAX_FILES_PER_UPLOAD = 20


class UploadProblem(ValueError):
    pass


def _validate_bytes(data: bytes, suffix: str) -> None:
    if not data:
        raise UploadProblem("The file is empty.")
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise UploadProblem("This file does not look like a PDF (missing %PDF header).")
        return
    if b"\x00" in data[:4096]:
        raise UploadProblem("This text file contains binary data.")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            data.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise UploadProblem("Could not decode this file as text.") from exc


def _read_limited(file_storage, limit: int) -> bytes:
    data = file_storage.read(limit + 1)
    if len(data) > limit:
        raise UploadProblem(f"Files must be smaller than {limit // (1024 * 1024)} MB each.")
    return data


@bp.post("")
@login_required
def upload(kb_id):
    kb = get_knowledge_base_or_404(kb_id)
    settings = current_app.config["SETTINGS"]
    files = [item for item in request.files.getlist("files") if item and item.filename]
    if not files:
        flash("Choose at least one file to upload.", "error")
        return redirect(url_for("knowledge_bases.show", kb_id=kb.id))
    if len(files) > MAX_FILES_PER_UPLOAD:
        flash(f"Upload at most {MAX_FILES_PER_UPLOAD} files at a time.", "error")
        return redirect(url_for("knowledge_bases.show", kb_id=kb.id))

    queued = 0
    skipped: list[str] = []
    for file_storage in files:
        display_name = safe_display_name(file_storage.filename or "document")
        suffix = Path(display_name).suffix.lower()
        if suffix not in ALLOWED_TYPES:
            skipped.append(f"{display_name}: unsupported type (PDF, MD, TXT only)")
            continue
        try:
            data = _read_limited(file_storage, settings.max_upload_bytes)
            _validate_bytes(data, suffix)
        except UploadProblem as exc:
            skipped.append(f"{display_name}: {exc}")
            continue

        digest = hashlib.sha256(data).hexdigest()
        existing = db.session.scalar(
            select(Document).where(
                Document.knowledge_base_id == kb.id,
                Document.sha256 == digest,
            )
        )
        if existing is not None:
            active_job = db.session.scalar(
                select(Job.id).where(
                    Job.document_id == existing.id,
                    Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                )
            )
            if existing.status in {DocumentStatus.READY, DocumentStatus.PROCESSING} or active_job:
                skipped.append(f"{display_name}: already in this knowledge base")
                continue

        if existing is not None:
            document = existing
            old_path = (document.meta or {}).get("storage_path")
            document.filename = display_name
            document.content_type = ALLOWED_TYPES[suffix]
            document.size_bytes = len(data)
            document.status = DocumentStatus.QUEUED
            document.error = None
            document.chunk_count = 0
            db.session.execute(Chunk.__table__.delete().where(Chunk.document_id == document.id))
        else:
            document = Document(
                id=uuid.uuid4(),
                knowledge_base_id=kb.id,
                filename=display_name,
                content_type=ALLOWED_TYPES[suffix],
                size_bytes=len(data),
                sha256=digest,
                status=DocumentStatus.QUEUED,
            )
            db.session.add(document)
            old_path = None

        target = storage_path(kb.id, document.id, display_name)
        document.meta = {
            **(document.meta or {}),
            "storage_path": str(target),
            "original_filename": display_name,
        }
        try:
            target.write_bytes(data)
        except OSError:
            log.exception("could not store upload for document %s", document.id)
            db.session.rollback()
            skipped.append(f"{display_name}: could not be saved on the server")
            continue
        if old_path and str(Path(old_path)) != str(target):
            remove_file(old_path)
        db.session.commit()
        enqueue_ingest(document.id)
        queued += 1

    if queued:
        flash(
            f"Queued {queued} document{'s' if queued != 1 else ''} for processing.",
            "success",
        )
    for message in skipped:
        flash(message, "error")
    return redirect(url_for("knowledge_bases.show", kb_id=kb.id))


@bp.get("/fragments/table")
@login_required
def table_fragment(kb_id):
    kb = get_knowledge_base_or_404(kb_id)
    documents = list(
        db.session.scalars(
            select(Document)
            .where(Document.knowledge_base_id == kb.id)
            .order_by(Document.created_at.desc())
        )
    )
    pending = any(
        document.status in {DocumentStatus.QUEUED, DocumentStatus.PROCESSING}
        for document in documents
    )
    return render_template(
        "documents/_table.html", kb=kb, documents=documents, pending=pending
    )


@bp.get("/<uuid:document_id>")
@login_required
def show(kb_id, document_id):
    kb = get_knowledge_base_or_404(kb_id)
    document = get_document_or_404(kb, document_id)
    page = max(1, int(request.args.get("page", 1) or 1))
    per_page = 60
    chunks = list(
        db.session.scalars(
            select(Chunk)
            .where(Chunk.document_id == document.id)
            .order_by(Chunk.chunk_index)
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
    )
    has_next = document.chunk_count > page * per_page
    return render_template(
        "documents/show.html",
        kb=kb,
        document=document,
        chunks=chunks,
        page=page,
        has_next=has_next,
    )


@bp.post("/<uuid:document_id>/delete")
@login_required
def delete(kb_id, document_id):
    kb = get_knowledge_base_or_404(kb_id)
    document = get_document_or_404(kb, document_id)
    path = (document.meta or {}).get("storage_path")
    db.session.delete(document)
    db.session.commit()
    remove_file(path)
    flash(f"Deleted “{document.filename}” and its chunks.", "success")
    return redirect(url_for("knowledge_bases.show", kb_id=kb.id))


@bp.post("/<uuid:document_id>/reprocess")
@login_required
def reprocess(kb_id, document_id):
    kb = get_knowledge_base_or_404(kb_id)
    document = get_document_or_404(kb, document_id)
    if document.status == DocumentStatus.PROCESSING:
        flash("This document is already being processed.", "error")
        return redirect(url_for("documents.show", kb_id=kb.id, document_id=document.id))
    enqueue_ingest(document.id, kind="reprocess")
    flash(f"Re-queued “{document.filename}”.", "success")
    return redirect(url_for("knowledge_bases.show", kb_id=kb.id))


@bp.post("/<uuid:document_id>/chunks/<uuid:chunk_id>/delete")
@login_required
def delete_chunk(kb_id, document_id, chunk_id):
    # Chunk-level deletion is intentionally not exposed; this guards accidental calls.
    kb = get_knowledge_base_or_404(kb_id)
    get_document_or_404(kb, document_id)
    abort(405)
