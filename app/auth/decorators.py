"""Require a signed-in user for a view."""

from __future__ import annotations

from functools import wraps

from flask import g, redirect, request, url_for


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.get("user") is None:
            target = request.path
            if request.query_string:
                target = f"{target}?{request.query_string.decode()}"
            return redirect(url_for("auth.login", next=target))
        return view(*args, **kwargs)

    return wrapped
