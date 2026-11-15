"""Password hashing and API-token generation.

Tokens are shown to the user exactly once. The database stores a SHA-256 hash and
an 8-character prefix for display; a leak of the table cannot be replayed.
"""

from __future__ import annotations

import hashlib
import secrets

from werkzeug.security import check_password_hash, generate_password_hash

TOKEN_PREFIX = "rag"
_MIN_PASSWORD = 8


def hash_password(password: str) -> str:
    return generate_password_hash(password, method="scrypt")


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return check_password_hash(password_hash, password)
    except (ValueError, TypeError):
        return False


def validate_password(password: str) -> str | None:
    """Return a human-readable problem, or None when the password is acceptable."""
    if len(password) < _MIN_PASSWORD:
        return f"Password must be at least {_MIN_PASSWORD} characters."
    if password.strip() != password:
        return "Password must not start or end with whitespace."
    if len(password) > 256:
        return "Password is too long."
    return None


def new_api_token() -> tuple[str, str, str]:
    """Return ``(plaintext, prefix, sha256_hash)`` for a fresh knowledge-base token."""
    secret = secrets.token_urlsafe(32)
    plaintext = f"{TOKEN_PREFIX}_{secret}"
    prefix = plaintext[:12]
    return plaintext, prefix, hash_token(plaintext)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
