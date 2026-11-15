"""CSRF protection: a session-bound token on every form and every unsafe request."""

from __future__ import annotations

import hmac
import secrets

from flask import Flask, abort, request, session


def csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def _valid(submitted: str | None) -> bool:
    expected = session.get("csrf_token")
    if not submitted or not expected:
        return False
    return hmac.compare_digest(submitted, expected)


def csrf_protect(app: Flask) -> None:
    @app.before_request
    def _check_csrf():
        if not app.config.get("CSRF_ENABLED", True):
            return None
        if request.method in {"GET", "HEAD", "OPTIONS", "TRACE"}:
            return None
        if request.blueprint == "mcp":
            return None
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not _valid(token):
            abort(400, description="CSRF token missing or invalid. Reload the page and retry.")
        return None
