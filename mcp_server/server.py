"""Remote MCP server (streamable HTTP) exposing one knowledge base per token.

Run with ``python -m mcp_server`` or alongside Flask with ``python -m app``.
Authentication is a bearer token; the token is hashed and looked up per request, and
the resolved knowledge base rides in a context variable for the duration of the call.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from sqlalchemy import func, select, update
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import get_settings
from app.db import db
from app.models import ApiToken, KnowledgeBase, User
from app.security import hash_token
from mcp_server import service

logger = logging.getLogger(__name__)

current_kb_id: ContextVar[uuid.UUID | None] = ContextVar("current_kb_id", default=None)

INSTRUCTIONS = """\
Search a private document library and answer with citations. Start with
search_documents(); if a hit looks promising but thin, call get_surrounding_context()
to read more around it, or get_document_outline() to navigate a long file. You may
search several times with different wording. Only answer from returned passages; when
they do not contain the answer, say so explicitly instead of guessing. Cite sources as
"filename, page N" (or the heading path) for every claim.
"""


def _flask_app():
    """One Flask app per process, used only for database access and settings."""
    from app import create_app

    app = getattr(_flask_app, "_app", None)
    if app is None:
        app = create_app()
        _flask_app._app = app  # type: ignore[attr-defined]
    return app


def _kb() -> KnowledgeBase:
    kb_id = current_kb_id.get()
    if kb_id is None:
        raise RuntimeError("No knowledge base is bound to this request.")
    kb = db.session.get(KnowledgeBase, kb_id)
    if kb is None:
        raise RuntimeError("Knowledge base no longer exists.")
    return kb


def _run(fn, *args, **kwargs):
    with _flask_app().app_context():
        try:
            return fn(*args, **kwargs)
        finally:
            db.remove()


def build_server() -> MCPServer:
    server = MCPServer(name="rag-mcp", instructions=INSTRUCTIONS, version="0.1.0")

    @server.tool(
        description=(
            "Semantic + keyword search over this knowledge base. Returns the best passages "
            "with filename, page number, heading path, a relevance score and a citation. "
            "Call it multiple times with different wording when the first results are weak. "
            "'mode' can override the library default: 'vector' (meaning-based), 'keyword' "
            "(exact words), or 'hybrid' (both, usually best)."
        )
    )
    def search_documents(query: str, top_k: int = 5, mode: str | None = None) -> dict:
        """Search documents and return cited passages."""
        return _run(service.search, _kb(), query, top_k=top_k, mode=mode)

    @server.tool(
        description=(
            "List every document in this knowledge base with its ingestion status, page "
            "count and chunk count. Use it to discover what is available or to find a "
            "document_id for the outline tool. Only 'ready' documents are searchable."
        )
    )
    def list_documents() -> dict:
        """List documents with status and sizes."""
        return _run(service.list_documents, _kb())

    @server.tool(
        description=(
            "Fetch one chunk by its chunk_id (from search_documents) with full text, "
            "citation, and its parent text when the library uses parent-child chunking. "
            "Use it when a search snippet is truncated or you need the exact wording."
        )
    )
    def get_document_chunk(chunk_id: str) -> dict:
        """Return a chunk's full text and metadata."""
        return _run(service.get_chunk, _kb(), chunk_id)

    @server.tool(
        description=(
            "Return the chunks immediately before and after a hit (default 2 each) so you "
            "can read the surrounding argument or table without guessing. Useful when a "
            "passage references context that was cut off."
        )
    )
    def get_surrounding_context(chunk_id: str, before: int = 2, after: int = 2) -> dict:
        """Expand around a chunk within the same document."""
        return _run(service.surrounding_context, _kb(), chunk_id, before=before, after=after)

    @server.tool(
        description=(
            "Return the heading outline of a document so you can see its structure and pick "
            "where to search next. Takes a document_id from list_documents or search results."
        )
    )
    def get_document_outline(document_id: str) -> dict:
        """Return detected headings for one document."""
        return _run(service.document_outline, _kb(), document_id)

    return server


class BearerAuth:
    """Authenticate the Authorization header before MCP ever sees the request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        token = _bearer(headers.get("authorization", ""))
        kb_id = _resolve_token(token) if token else None
        if kb_id is None:
            await _reject(send)
            return

        reset = current_kb_id.set(kb_id)
        try:
            await self.app(scope, receive, send)
        finally:
            current_kb_id.reset(reset)


def _bearer(header: str) -> str | None:
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def _resolve_token(token: str) -> uuid.UUID | None:
    digest = hash_token(token)
    with _flask_app().app_context():
        try:
            row = db.session.execute(
                select(ApiToken.knowledge_base_id)
                .join(KnowledgeBase, KnowledgeBase.id == ApiToken.knowledge_base_id)
                .join(User, User.id == KnowledgeBase.user_id)
                .where(
                    ApiToken.token_hash == digest,
                    ApiToken.revoked_at.is_(None),
                    User.is_active.is_(True),
                )
            ).first()
            if row is None:
                return None
            kb_id = row[0]
            db.session.execute(
                update(ApiToken)
                .where(ApiToken.token_hash == digest)
                .values(last_used_at=func.now())
            )
            db.session.commit()
            return kb_id
        except Exception:
            db.session.rollback()
            logger.exception("token lookup failed")
            return None
        finally:
            db.remove()


async def _reject(send: Send) -> None:
    body = b'{"error":"unauthorized","message":"Provide a valid knowledge-base token as: Authorization: Bearer <token>"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="rag-mcp"'),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def build_asgi_app() -> ASGIApp:
    settings = get_settings()
    server = build_server()
    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        host=settings.mcp_host,
    )
    return BearerAuth(app)


def run_mcp_server() -> None:
    import uvicorn

    settings = get_settings()
    logger.info("MCP server on http://%s:%s/mcp", settings.mcp_host, settings.mcp_port)
    uvicorn.run(
        build_asgi_app(),
        host=settings.mcp_host,
        port=settings.mcp_port,
        log_level="info",
    )


def main() -> None:
    from app.logging_setup import configure_logging

    configure_logging()
    run_mcp_server()


if __name__ == "__main__":
    main()
