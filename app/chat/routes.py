"""Built-in web chat: simple RAG and agentic tool-calling modes, streamed over SSE."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from typing import Any

from flask import Blueprint, Response, current_app, render_template, request, stream_with_context
from sqlalchemy import func, select

from app.auth.decorators import login_required
from app.db import db
from app.models import Document
from app.tenancy import get_knowledge_base_or_404
from llm import LLMError, Message, get_llm
from mcp_server import service
from retrieval import RetrievalError, retrieve

bp = Blueprint("chat", __name__, url_prefix="/knowledge-bases/<uuid:kb_id>/chat")

MAX_QUESTION_CHARS = 4000
HISTORY_TURNS = 8
CONTEXT_CHARS = 3500

SYSTEM_PROMPT = """\
You are the research assistant for the document library "{name}".
Answer strictly from the passages labelled [1], [2], … in the user message. Cite every
claim with the matching bracket number, and include the filename and page when you can.
If the passages do not contain the answer, say clearly that the documents do not cover it
and suggest what to search for next. Never invent facts, page numbers or filenames.
Keep answers focused: a short paragraph or a few bullets, not an essay."""

AGENT_SYSTEM_PROMPT = """\
You are the research assistant for the document library "{name}". Use the provided tools
to search the library, read around promising hits, and inspect document outlines. You may
search up to {limit} times with different wording. Answer only from what the tools return,
cite sources as "filename, page N" (or the heading path), and say so plainly when the
library does not contain the answer. Never invent facts or citations."""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search this knowledge base. Returns cited passages with filename, page and "
                "heading. Call several times with different wording if results are weak."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    "mode": {
                        "type": "string",
                        "enum": ["vector", "keyword", "hybrid"],
                        "description": "Optional override of the library's retrieval mode",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List documents in the knowledge base with status and page counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_chunk",
            "description": "Fetch one chunk by chunk_id with its full text and citation.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_surrounding_context",
            "description": "Read chunks before and after a hit to see the surrounding context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "before": {"type": "integer", "minimum": 0, "maximum": 10, "default": 2},
                    "after": {"type": "integer", "minimum": 0, "maximum": 10, "default": 2},
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_outline",
            "description": "Return the headings of a document, to navigate long files.",
            "parameters": {
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
            },
        },
    },
]


@bp.get("")
@login_required
def page(kb_id: uuid.UUID):
    kb = get_knowledge_base_or_404(kb_id)
    ready = db.session.scalar(
        select(func.count(Document.id)).where(
            Document.knowledge_base_id == kb.id, Document.status == "ready"
        )
    )
    return render_template(
        "chat/page.html",
        kb=kb,
        ready_documents=ready or 0,
        default_mode="agentic" if get_llm().configured else "simple",
        llm_configured=get_llm().configured,
    )


@bp.post("/stream")
@login_required
def stream(kb_id: uuid.UUID):
    kb = get_knowledge_base_or_404(kb_id)
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("message") or "").strip()[:MAX_QUESTION_CHARS]
    mode = payload.get("mode") if payload.get("mode") in {"simple", "agentic"} else "simple"
    history = _clean_history(payload.get("history"))

    def generate() -> Iterator[str]:
        try:
            if not question:
                yield _event("error", {"message": "Type a question first."})
            elif mode == "agentic" and get_llm().configured:
                yield from _agentic(kb, question, history)
            else:
                yield from _simple(kb, question, history)
        except RetrievalError as exc:
            yield _event("error", {"message": str(exc)})
        except LLMError as exc:
            yield _event("error", {"message": str(exc)})
        except Exception:
            current_app.logger.exception("chat stream failed")
            yield _event("error", {"message": "Something went wrong while answering. Try again."})
        yield _event("done", {})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _simple(kb, question: str, history: list[Message]) -> Iterator[str]:
    hits = retrieve(kb, question)
    sources = [_source(hit, index + 1) for index, hit in enumerate(hits)]
    if not hits:
        yield _event("token", {"text": _no_results_text()})
        yield _event("citations", {"sources": []})
        return

    context = "\n\n".join(
        f"[{index + 1}] {hit.citation}\n{_clip(hit.text, CONTEXT_CHARS)}"
        for index, hit in enumerate(hits)
    )
    messages = [Message("system", SYSTEM_PROMPT.format(name=kb.name)), *history]
    messages.append(
        Message(
            "user",
            f"Question: {question}\n\nCONTEXT PASSAGES:\n{context}\n\n"
            "Answer from these passages and cite [n].",
        )
    )
    llm = get_llm()
    for delta in llm.stream(messages, max_tokens=current_app.config["SETTINGS"].llm_max_tokens):
        yield _event("token", {"text": delta})
    yield _event("citations", {"sources": sources})


def _agentic(kb, question: str, history: list[Message]) -> Iterator[str]:
    settings = current_app.config["SETTINGS"]
    llm = get_llm()
    limit = settings.agentic_max_tool_calls
    messages: list[Message] = [
        Message("system", AGENT_SYSTEM_PROMPT.format(name=kb.name, limit=limit)),
        *history,
        Message("user", question),
    ]
    collected: dict[str, dict[str, Any]] = {}
    calls = 0
    answered = False

    while calls < limit:
        response = llm.chat(messages, tools=TOOLS, temperature=0.1)
        if not response.tool_calls:
            if response.content:
                for block in _chunks(response.content, 24):
                    yield _event("token", {"text": block})
                answered = True
            break

        messages.append(
            Message(
                "assistant",
                response.content or "",
                tool_calls=[call.raw for call in response.tool_calls],
            )
        )
        for call in response.tool_calls:
            calls += 1
            yield _event(
                "tool",
                {"name": call.name, "arguments": call.arguments, "call": calls, "limit": limit},
            )
            result = _execute_tool(kb, call.name, call.arguments)
            _collect_sources(result, collected)
            messages.append(
                Message(
                    "tool",
                    json.dumps(result, default=str)[:12000],
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
            if calls >= limit:
                break

    if not answered:
        messages.append(
            Message(
                "user",
                "Now write the final answer from the tool results above. Cite filename and page. "
                "If the library does not contain the answer, say so plainly.",
            )
        )
        for delta in llm.stream(
            messages, max_tokens=settings.llm_max_tokens, temperature=0.1
        ):
            yield _event("token", {"text": delta})

    sources = sorted(
        collected.values(), key=lambda item: item.get("score") or 0, reverse=True
    )[:6]
    for index, source in enumerate(sources, start=1):
        source["n"] = index
    yield _event("citations", {"sources": sources})


def _execute_tool(kb, name: str, arguments: dict[str, Any]) -> Any:
    try:
        if name == "search_documents":
            query = str(arguments.get("query") or "")[:500]
            if not query.strip():
                return {"error": "query must not be empty"}
            top_k = _int(arguments.get("top_k"), 5, 1, 10)
            mode = arguments.get("mode")
            mode = mode if mode in {"vector", "keyword", "hybrid"} else None
            return service.search(kb, query, top_k=top_k, mode=mode)
        if name == "list_documents":
            return service.list_documents(kb)
        if name == "get_document_chunk":
            return service.get_chunk(kb, str(arguments.get("chunk_id") or ""))
        if name == "get_surrounding_context":
            return service.surrounding_context(
                kb,
                str(arguments.get("chunk_id") or ""),
                before=_int(arguments.get("before"), 2, 0, 10),
                after=_int(arguments.get("after"), 2, 0, 10),
            )
        if name == "get_document_outline":
            return service.document_outline(kb, str(arguments.get("document_id") or ""))
        return {"error": f"unknown tool '{name}'"}
    except LookupError as exc:
        return {"error": str(exc)}
    except RetrievalError as exc:
        return {"error": str(exc)}


def _collect_sources(result: Any, collected: dict[str, dict[str, Any]]) -> None:
    if not isinstance(result, dict):
        return
    for item in result.get("results", []) or []:
        if not isinstance(item, dict) or not item.get("chunk_id"):
            continue
        chunk_id = item["chunk_id"]
        if chunk_id in collected:
            continue
        collected[chunk_id] = {
            "n": len(collected) + 1,
            "chunk_id": chunk_id,
            "document_id": item.get("document_id"),
            "filename": item.get("filename"),
            "page_number": item.get("page_number"),
            "heading_path": item.get("heading_path"),
            "citation": item.get("citation"),
            "score": item.get("score"),
            "text": _clip(item.get("text", ""), 900),
        }


def _source(hit, n: int) -> dict[str, Any]:
    return {
        "n": n,
        "chunk_id": str(hit.chunk_id),
        "document_id": str(hit.document_id),
        "filename": hit.filename,
        "page_number": hit.page_number,
        "heading_path": hit.heading_path,
        "citation": hit.citation,
        "score": round(float(hit.score), 4),
        "text": _clip(hit.text, 900),
    }


def _clean_history(raw: Any) -> list[Message]:
    if not isinstance(raw, list):
        return []
    messages: list[Message] = []
    for item in raw[-HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()[:MAX_QUESTION_CHARS]
        if content:
            messages.append(Message(role, content))
    return messages


def _int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _chunks(text: str, size: int) -> Iterator[str]:
    for index in range(0, len(text), size):
        yield text[index : index + size]


def _no_results_text() -> str:
    return (
        "I could not find anything relevant in this knowledge base. Try different wording, "
        "or open the documents tab to check that your files finished processing."
    )


def _event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, default=str)}\n\n"
