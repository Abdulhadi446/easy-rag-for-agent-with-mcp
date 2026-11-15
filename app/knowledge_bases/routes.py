"""Knowledge base listing and creation."""

from __future__ import annotations

from flask import Blueprint, current_app, flash, g, redirect, render_template, request, url_for
from sqlalchemy import func, select

from app.auth.decorators import login_required
from app.config import DEFAULT_KB_SETTINGS
from app.db import db
from app.models import Document, KnowledgeBase
from app.tenancy import get_knowledge_base_or_404, user_knowledge_bases

bp = Blueprint("knowledge_bases", __name__, url_prefix="/knowledge-bases")


@bp.get("/")
@login_required
def index():
    kbs = user_knowledge_bases(g.user)
    counts = dict(
        db.session.execute(
            select(Document.knowledge_base_id, func.count(Document.id))
            .where(Document.knowledge_base_id.in_([kb.id for kb in kbs] or [None]))
            .group_by(Document.knowledge_base_id)
        ).all()
    )
    return render_template("knowledge_bases/index.html", knowledge_bases=kbs, counts=counts)


@bp.post("/")
@login_required
def create():
    name = (request.form.get("name") or "").strip()
    description = (request.form.get("description") or "").strip() or None
    if not name:
        flash("Give the knowledge base a name.", "error")
        return redirect(url_for("knowledge_bases.index"))
    if len(name) > 120:
        flash("Name must be 120 characters or fewer.", "error")
        return redirect(url_for("knowledge_bases.index"))

    embedding = current_app.config["SETTINGS"]
    kb = KnowledgeBase(
        user_id=g.user.id,
        name=name,
        description=description,
        embedding_model=embedding.embedding_model,
        embedding_dim=embedding.embedding_dim,
        **DEFAULT_KB_SETTINGS,
    )
    db.session.add(kb)
    db.session.commit()
    flash(f"Created “{kb.name}”.", "success")
    return redirect(url_for("knowledge_bases.show", kb_id=kb.id))


@bp.get("/<uuid:kb_id>")
@login_required
def show(kb_id):
    kb = get_knowledge_base_or_404(kb_id)
    documents = list(
        db.session.scalars(
            select(Document)
            .where(Document.knowledge_base_id == kb.id)
            .order_by(Document.created_at.desc())
        )
    )
    return render_template("knowledge_bases/show.html", kb=kb, documents=documents)
