"""Per-knowledge-base settings with validation and re-ingestion."""

from __future__ import annotations

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from sqlalchemy import select

from app.auth.decorators import login_required
from app.config import DEFAULT_KB_SETTINGS
from app.db import db
from app.models import Document, DocumentStatus
from app.tenancy import get_knowledge_base_or_404
from ingestion.chunkers import STRATEGIES, STRATEGY_LABELS
from ingestion.pipeline import enqueue_ingest

bp = Blueprint("kb_settings", __name__, url_prefix="/knowledge-bases/<uuid:kb_id>/settings")

CHUNKING_FIELDS = {
    "chunk_strategy",
    "chunk_size",
    "chunk_overlap",
    "sentence_window_before",
    "sentence_window_after",
    "parent_chunk_size",
    "child_chunk_size",
    "semantic_threshold",
}

RETRIEVAL_MODES = {
    "hybrid": "Hybrid — vector + keyword (recommended)",
    "vector": "Vector only — meaning-based",
    "keyword": "Keyword only — exact words",
}

RANGES = {
    "chunk_size": (100, 2000, "Chunk size must be between 100 and 2,000 tokens."),
    "chunk_overlap": (0, 500, "Overlap must be between 0 and 500 tokens."),
    "sentence_window_before": (0, 10, "Window before must be between 0 and 10 sentences."),
    "sentence_window_after": (0, 10, "Window after must be between 0 and 10 sentences."),
    "parent_chunk_size": (200, 4000, "Parent size must be between 200 and 4,000 tokens."),
    "child_chunk_size": (50, 1000, "Child size must be between 50 and 1,000 tokens."),
    "top_k": (1, 20, "Top results must be between 1 and 20."),
    "candidates": (10, 100, "Candidate pool must be between 10 and 100."),
}


def _int_field(name: str, default: int, errors: list[str]) -> int:
    raw = (request.form.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{name.replace('_', ' ').capitalize()} must be a whole number.")
        return default
    low, high, message = RANGES[name]
    if not low <= value <= high:
        errors.append(message)
        return default
    return value


def _enqueue_ready_documents(kb) -> int:
    documents = db.session.scalars(
        select(Document).where(
            Document.knowledge_base_id == kb.id,
            Document.status.in_([DocumentStatus.READY, DocumentStatus.FAILED]),
        )
    ).all()
    for document in documents:
        enqueue_ingest(document.id, kind="reingest")
    return len(documents)


@bp.get("")
@login_required
def page(kb_id):
    kb = get_knowledge_base_or_404(kb_id)
    return render_template(
        "settings/page.html",
        kb=kb,
        strategies=STRATEGY_LABELS,
        modes=RETRIEVAL_MODES,
        defaults=DEFAULT_KB_SETTINGS,
        rerank_provider=current_app.config["SETTINGS"].rerank_provider,
    )


@bp.post("")
@login_required
def save(kb_id):
    kb = get_knowledge_base_or_404(kb_id)
    errors: list[str] = []

    strategy = (request.form.get("chunk_strategy") or "").strip()
    if strategy not in STRATEGIES:
        errors.append("Choose a valid chunking strategy.")
        strategy = kb.chunk_strategy

    values = {
        "chunk_strategy": strategy,
        "chunk_size": _int_field("chunk_size", kb.chunk_size, errors),
        "chunk_overlap": _int_field("chunk_overlap", kb.chunk_overlap, errors),
        "sentence_window_before": _int_field(
            "sentence_window_before", kb.sentence_window_before, errors
        ),
        "sentence_window_after": _int_field(
            "sentence_window_after", kb.sentence_window_after, errors
        ),
        "parent_chunk_size": _int_field("parent_chunk_size", kb.parent_chunk_size, errors),
        "child_chunk_size": _int_field("child_chunk_size", kb.child_chunk_size, errors),
        "top_k": _int_field("top_k", kb.top_k, errors),
        "candidates": _int_field("candidates", kb.candidates, errors),
    }
    try:
        threshold = float(request.form.get("semantic_threshold", kb.semantic_threshold))
    except ValueError:
        threshold = float(kb.semantic_threshold)
        errors.append("Semantic threshold must be a number between 0 and 1.")
    if not 0.0 <= threshold <= 1.0:
        errors.append("Semantic threshold must be between 0 and 1.")
        threshold = float(kb.semantic_threshold)

    mode = (request.form.get("retrieval_mode") or "").strip()
    if mode not in RETRIEVAL_MODES:
        errors.append("Choose a valid retrieval mode.")
        mode = kb.retrieval_mode

    if values["chunk_overlap"] >= values["chunk_size"]:
        errors.append("Overlap must be smaller than the chunk size.")
    if values["child_chunk_size"] >= values["parent_chunk_size"]:
        errors.append("Child chunk size must be smaller than the parent chunk size.")
    if values["candidates"] < values["top_k"]:
        errors.append("The candidate pool must be at least as large as top results.")

    if errors:
        for message in errors:
            flash(message, "error")
        return redirect(url_for("kb_settings.page", kb_id=kb.id))

    changed_chunking = any(
        getattr(kb, field) != values[field] for field in CHUNKING_FIELDS if field in values
    ) or float(kb.semantic_threshold) != threshold

    for field, value in values.items():
        setattr(kb, field, value)
    kb.semantic_threshold = threshold
    kb.retrieval_mode = mode
    kb.rerank_enabled = request.form.get("rerank_enabled") == "on"
    kb.query_rewrite_enabled = request.form.get("query_rewrite_enabled") == "on"
    db.session.commit()

    if changed_chunking:
        queued = _enqueue_ready_documents(kb)
        flash(
            "Settings saved. "
            + (
                f"Re-indexing {queued} document{'s' if queued != 1 else ''} with the new chunking."
                if queued
                else "New uploads will use these chunking settings."
            ),
            "success",
        )
    else:
        flash("Settings saved.", "success")
    return redirect(url_for("kb_settings.page", kb_id=kb.id))


@bp.post("/reset")
@login_required
def reset(kb_id):
    kb = get_knowledge_base_or_404(kb_id)
    defaults = DEFAULT_KB_SETTINGS
    chunking_changed = False
    for field, value in defaults.items():
        if field in {"rerank_enabled", "query_rewrite_enabled"}:
            continue
        if hasattr(kb, field) and getattr(kb, field) != value:
            setattr(kb, field, value)
            if field in CHUNKING_FIELDS:
                chunking_changed = True
    # Rerank defaults on, query rewriting defaults off — but only when a reranker exists.
    kb.rerank_enabled = bool(defaults.get("rerank_enabled", True))
    kb.query_rewrite_enabled = bool(defaults.get("query_rewrite_enabled", False))
    db.session.commit()
    if chunking_changed:
        queued = _enqueue_ready_documents(kb)
        flash(
            f"Reset to defaults. Re-indexing {queued} document{'s' if queued != 1 else ''}.",
            "success",
        )
    else:
        flash("Reset to defaults.", "success")
    return redirect(url_for("kb_settings.page", kb_id=kb.id))
