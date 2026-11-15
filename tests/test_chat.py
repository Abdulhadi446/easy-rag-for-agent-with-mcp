"""Chat streaming: simple RAG, agentic tool loop, citations, and configuration errors."""

from __future__ import annotations

import io
import json

import pytest
from sqlalchemy import select

from app.db import db
from app.models import KnowledgeBase
from ingestion.pipeline import drain_queue
from llm import ChatResponse, ToolCall

DOC = """# Handbook

## Refunds

Refunds are processed within five business days. EU customers can withdraw within
fourteen days.

## Invoices

Invoices are issued monthly.
"""


def login(client, email="a@example.com", password="correct-horse"):
    client.post("/login", data={"email": email, "password": password})


@pytest.fixture()
def chat(app, client, signup, monkeypatch):
    signup()
    with app.app_context():
        kb = db.session.scalar(select(KnowledgeBase))
    response = client.post(
        f"/knowledge-bases/{kb.id}/documents",
        data={"files": (io.BytesIO(DOC.encode()), "handbook.md", "text/markdown")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    with app.app_context():
        drain_queue()
    return client, kb


class FakeLLM:
    configured = True

    def __init__(self) -> None:
        self.calls = 0

    def stream(self, messages, **kwargs):
        yield "Refunds take five business days"
        yield " [1]."

    def chat(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raw = {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search_documents", "arguments": json.dumps({"query": "refunds"})},
            }
            return ChatResponse(
                "",
                [
                    ToolCall(
                        id="call_1",
                        name="search_documents",
                        arguments={"query": "refunds"},
                        raw=raw,
                    )
                ],
                "tool_calls",
            )
        return ChatResponse("Refunds are processed within five business days.", [], "stop")


def _events(response) -> list[tuple[str, dict]]:
    events = []
    for block in response.data.decode().split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name:
            events.append((name, data or {}))
    return events


def test_empty_question_returns_friendly_error(chat):
    client, kb = chat
    response = client.post(
        f"/knowledge-bases/{kb.id}/chat/stream", json={"message": "  ", "mode": "simple"}
    )
    assert response.status_code == 200
    events = _events(response)
    assert events[0][0] == "error"
    assert "Type a question" in events[0][1]["message"]
    assert events[-1][0] == "done"


def test_simple_mode_streams_answer_and_citations(chat, monkeypatch):
    client, kb = chat
    monkeypatch.setattr("app.chat.routes.get_llm", lambda: FakeLLM())
    response = client.post(
        f"/knowledge-bases/{kb.id}/chat/stream",
        json={"message": "How long do refunds take?", "mode": "simple", "history": []},
    )
    events = _events(response)
    names = [name for name, _ in events]
    assert "token" in names and "citations" in names
    answer = "".join(data["text"] for name, data in events if name == "token")
    assert "five business days" in answer
    sources = next(data["sources"] for name, data in events if name == "citations")
    assert sources and sources[0]["filename"] == "handbook.md"


def test_agentic_mode_calls_tools_then_answers(chat, monkeypatch):
    client, kb = chat
    fake = FakeLLM()
    monkeypatch.setattr("app.chat.routes.get_llm", lambda: fake)
    response = client.post(
        f"/knowledge-bases/{kb.id}/chat/stream",
        json={"message": "How long do refunds take?", "mode": "agentic", "history": []},
    )
    events = _events(response)
    tools = [data for name, data in events if name == "tool"]
    assert tools and tools[0]["name"] == "search_documents"
    assert fake.calls == 2
    answer = "".join(data["text"] for name, data in events if name == "token")
    assert "five business days" in answer
    sources = next(data["sources"] for name, data in events if name == "citations")
    assert sources and sources[0]["filename"] == "handbook.md"


def test_agentic_falls_back_to_simple_without_a_model(chat, monkeypatch):
    client, kb = chat

    class Unconfigured(FakeLLM):
        configured = False

    monkeypatch.setattr("app.chat.routes.get_llm", lambda: Unconfigured())
    response = client.post(
        f"/knowledge-bases/{kb.id}/chat/stream",
        json={"message": "How long do refunds take?", "mode": "agentic"},
    )
    events = _events(response)
    assert "tool" not in [name for name, _ in events]
    assert "citations" in [name for name, _ in events]
