"""Turn a vague chat question into a better search query (optional per KB)."""

from __future__ import annotations

import logging

from app.config import Settings
from llm import LLMClient, LLMError, Message, get_llm

log = logging.getLogger(__name__)


def rewrite_query(
    query: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> str:
    query = (query or "").strip()
    if not query:
        return query
    try:
        client = llm or get_llm(settings)
        response = client.chat(
            [
                Message(
                    role="system",
                    content=(
                        "Rewrite the user's question as one short search query for a document "
                        "index. Keep names, numbers and product terms. Add likely synonyms only "
                        "when they are unambiguous. Reply with the query alone, no quotes, no "
                        "explanation."
                    ),
                ),
                Message(role="user", content=query),
            ],
            temperature=0.0,
            max_tokens=80,
        )
    except LLMError as exc:
        log.info("query rewrite skipped: %s", exc)
        return query
    rewritten = (response.content or "").strip().strip('"').strip()
    if not rewritten or len(rewritten) > 400:
        return query
    return rewritten.splitlines()[0].strip() or query
