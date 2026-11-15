"""Where uploaded files live. Outside static/, keyed by KB and document id."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from flask import current_app

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ ()\[\]-]+")


def upload_root() -> Path:
    settings = current_app.config["SETTINGS"]
    root = Path(settings.upload_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_display_name(filename: str) -> str:
    name = Path(filename or "").name.strip()
    name = _SAFE_NAME_RE.sub("_", name).strip(" ._")
    if not name:
        name = "document"
    return name[:200]


def storage_path(kb_id: uuid.UUID, document_id: uuid.UUID, filename: str) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".markdown", ".txt"}:
        suffix = ".bin"
    directory = upload_root() / str(kb_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{document_id}{suffix}"


def remove_file(path: str | None) -> None:
    if not path:
        return
    candidate = Path(path)
    try:
        root = upload_root()
        if root in candidate.resolve().parents and candidate.is_file():
            candidate.unlink()
    except OSError:
        # A missing file must not block deleting the database record.
        pass
