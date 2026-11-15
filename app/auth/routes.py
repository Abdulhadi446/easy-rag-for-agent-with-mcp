"""Signup, login, logout."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from flask import (
    Blueprint,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app import ratelimit
from app.config import DEFAULT_KB_SETTINGS
from app.db import db
from app.models import KnowledgeBase, User
from app.security import hash_password, validate_password, verify_password

bp = Blueprint("auth", __name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()


def valid_email(email: str) -> bool:
    return bool(email) and len(email) <= 320 and _EMAIL_RE.match(email) is not None


def _safe_next(target: str | None) -> str | None:
    if not target or not target.startswith("/") or target.startswith("//"):
        return None
    return target


@bp.get("/signup")
def signup_form():
    if g.get("user"):
        return redirect(url_for("knowledge_bases.index"))
    return render_template("auth/signup.html")


@bp.post("/signup")
def signup():
    email = normalize_email(request.form.get("email", ""))
    password = request.form.get("password", "")
    display_name = (request.form.get("display_name") or "").strip() or None

    if not valid_email(email):
        flash("Enter a valid email address.", "error")
        return render_template("auth/signup.html", email=email, display_name=display_name), 400
    problem = validate_password(password)
    if problem:
        flash(problem, "error")
        return render_template("auth/signup.html", email=email, display_name=display_name), 400

    settings = current_app.config["SETTINGS"]
    user = User(
        email=email,
        password_hash=hash_password(password),
        display_name=display_name,
    )
    db.session.add(user)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        flash("That email already has an account. Log in instead.", "error")
        return render_template("auth/signup.html", email=email, display_name=display_name), 409

    # A starter knowledge base so the first upload needs no setup at all.
    kb = KnowledgeBase(
        user_id=user.id,
        name="My documents",
        description="Your first knowledge base. Rename it or add another any time.",
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        **DEFAULT_KB_SETTINGS,
    )
    db.session.add(kb)
    db.session.commit()

    session.clear()
    session["user_id"] = str(user.id)
    session.permanent = True
    flash("Welcome! Your workspace is ready.", "success")
    return redirect(url_for("knowledge_bases.index"))


@bp.get("/login")
def login_form():
    if g.get("user"):
        return redirect(url_for("knowledge_bases.index"))
    return render_template("auth/login.html", next=request.args.get("next", ""))


@bp.post("/login")
def login():
    email = normalize_email(request.form.get("email", ""))
    password = request.form.get("password", "")
    next_url = _safe_next(request.form.get("next"))
    throttle_key = f"{email}|{request.remote_addr}"

    if ratelimit.is_limited(throttle_key):
        flash("Too many failed attempts. Try again in a few minutes.", "error")
        return render_template("auth/login.html", email=email, next=next_url or ""), 429

    user = db.session.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active or not verify_password(user.password_hash, password):
        ratelimit.record_failure(throttle_key)
        # Same message for unknown email and wrong password.
        flash("Email or password is incorrect.", "error")
        return render_template("auth/login.html", email=email, next=next_url or ""), 401

    ratelimit.reset(throttle_key)
    user.last_login_at = datetime.now(UTC)
    db.session.commit()

    session.clear()
    session["user_id"] = str(user.id)
    session.permanent = True
    return redirect(next_url or url_for("knowledge_bases.index"))


@bp.post("/logout")
def logout():
    session.clear()
    flash("Signed out.", "success")
    return redirect(url_for("auth.login"))
