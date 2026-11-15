"""Logging config that never prints document text or credentials.

Alembic and the app both import this so SQL logs stay available in development
without leaking chunk contents into stdout.
"""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    # httpx logs full URLs (which can contain signed paths) at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
