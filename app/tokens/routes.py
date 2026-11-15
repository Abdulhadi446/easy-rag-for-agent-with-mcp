"""Knowledge-base MCP tokens: create, rotate, revoke.

The plaintext token is shown exactly once, through the session, and never logged.
"""

from __future__ import annotations

import uuid

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import func, select

from app.auth.decorators import login_required
from app.db import db
from app.models import ApiToken
from app.security import new_api_token
from app.tenancy import get_knowledge_base_or_404

bp = Blueprint("tokens", __name__, url_prefix="/knowledge-bases/<uuid:kb_id>/tokens")


def _reveal_key(kb_id: uuid.UUID) -> str:
    return f"token_reveal:{kb_id}"


@bp.get("")
@login_required
def page(kb_id: uuid.UUID):
    kb = get_knowledge_base_or_404(kb_id)
    tokens = db.session.scalars(
        select(ApiToken)
        .where(ApiToken.knowledge_base_id == kb.id)
        .order_by(ApiToken.created_at.desc())
    ).all()
    revealed = session.pop(_reveal_key(kb.id), None)
    return render_template(
        "tokens/page.html",
        kb=kb,
        tokens=tokens,
        revealed=revealed,
        mcp_url=_client_mcp_url(),
    )


@bp.post("")
@login_required
def create(kb_id: uuid.UUID):
    kb = get_knowledge_base_or_404(kb_id)
    name = (request.form.get("name") or "agent").strip()[:120] or "agent"
    active = db.session.scalar(
        select(ApiToken).where(
            ApiToken.knowledge_base_id == kb.id, ApiToken.revoked_at.is_(None)
        )
    )
    if active is not None:
        flash("This knowledge base already has an active token. Rotate it instead.", "error")
        return redirect(url_for("tokens.page", kb_id=kb.id))

    plaintext, prefix, token_hash = new_api_token()
    db.session.add(
        ApiToken(knowledge_base_id=kb.id, name=name, prefix=prefix, token_hash=token_hash)
    )
    db.session.commit()
    session[_reveal_key(kb.id)] = plaintext
    flash("Token created. Copy it now: it is shown only once.", "success")
    return redirect(url_for("tokens.page", kb_id=kb.id))


@bp.post("/<uuid:token_id>/revoke")
@login_required
def revoke(kb_id: uuid.UUID, token_id: uuid.UUID):
    kb = get_knowledge_base_or_404(kb_id)
    token = _token_or_404(kb.id, token_id)
    if token.revoked_at is None:
        token.revoked_at = func.now()
        db.session.commit()
        flash("Token revoked. Agents using it will get 401.", "success")
    return redirect(url_for("tokens.page", kb_id=kb.id))


@bp.post("/<uuid:token_id>/rotate")
@login_required
def rotate(kb_id: uuid.UUID, token_id: uuid.UUID):
    kb = get_knowledge_base_or_404(kb_id)
    old = _token_or_404(kb.id, token_id)
    old.revoked_at = func.now()
    plaintext, prefix, token_hash = new_api_token()
    db.session.add(
        ApiToken(
            knowledge_base_id=kb.id,
            name=old.name,
            prefix=prefix,
            token_hash=token_hash,
        )
    )
    db.session.commit()
    session[_reveal_key(kb.id)] = plaintext
    flash("Token rotated. Update your agent with the new value.", "success")
    return redirect(url_for("tokens.page", kb_id=kb.id))


def _token_or_404(kb_id: uuid.UUID, token_id: uuid.UUID) -> ApiToken:
    token = db.session.scalar(
        select(ApiToken).where(ApiToken.id == token_id, ApiToken.knowledge_base_id == kb_id)
    )
    if token is None:
        from flask import abort

        abort(404)
    return token


def _client_mcp_url() -> str:
    """Prefer the public URL; fall back to the local one for dev."""
    from flask import current_app

    settings = current_app.config["SETTINGS"]
    return settings.mcp_public_url or f"http://127.0.0.1:{settings.mcp_port}/mcp"
