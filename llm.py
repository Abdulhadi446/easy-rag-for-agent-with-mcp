"""One chat abstraction over OpenAI-compatible APIs (OpenRouter by default).

Used by both the web chat and the evaluation script, so streaming, tool calls and
retry behaviour live here rather than in routes.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import Settings, get_settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Message is safe to show to the user."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw: dict = field(default_factory=dict)

    @property
    def arguments_error(self) -> str | None:
        return self.arguments.get("__error__") if isinstance(self.arguments, dict) else None


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def as_dict(self) -> dict:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    model: str | None = None


def _parse_tool_calls(raw_calls: Sequence[dict] | None) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for raw in raw_calls or []:
        function = raw.get("function") or {}
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be an object")
        except Exception:  # noqa: BLE001
            arguments = {"__error__": f"Could not parse arguments: {raw_arguments[:200]}"}
        calls.append(
            ToolCall(
                id=raw.get("id") or f"call_{len(calls)}",
                name=function.get("name") or "",
                arguments=arguments,
                raw=raw,
            )
        )
    return calls


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = (self.settings.llm_base_url or "").rstrip("/")
        self.api_key = self.settings.llm_api_key
        self.model = self.settings.llm_model
        self.timeout = self.settings.llm_timeout
        self.max_tokens = self.settings.llm_max_tokens
        self.max_retries = 3

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if "openrouter" in self.base_url:
            headers.setdefault("HTTP-Referer", "http://localhost:8000")
            headers.setdefault("X-Title", "RAG MCP Platform")
        return headers

    def _payload(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict] | None,
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.as_dict() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        return payload

    def _check_configured(self) -> None:
        if not self.configured:
            raise LLMError(
                "Chat is not configured yet. Add an API key to .env "
                "(LLM_API_KEY or OPENROUTER_API_KEY) and restart the app."
            )

    def _handle_error_status(self, response: httpx.Response) -> None:
        detail = response.text[:300]
        if response.status_code in {401, 403}:
            raise LLMError(
                "The model provider rejected the API key. Check LLM_API_KEY in .env."
            )
        if response.status_code == 402:
            raise LLMError("The model provider reports no remaining credits for this key.")
        if response.status_code == 404:
            raise LLMError(
                f"The model {self.model!r} was not found at {self.base_url}. "
                "Check LLM_MODEL and LLM_BASE_URL."
            )
        raise LLMError(f"Model provider returned {response.status_code}: {detail}")

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self._check_configured()
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens, stream=False
        )
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    url, json=payload, headers=self._headers(), timeout=self.timeout
                )
            except httpx.RequestError as exc:
                if attempt == self.max_retries:
                    raise LLMError(f"Could not reach the model provider: {exc}") from exc
                delay = 2**attempt
            else:
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt == self.max_retries:
                        self._handle_error_status(response)
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.replace(".", "", 1).isdigit()
                        else 2**attempt
                    )
                    delay += random.uniform(0, 0.5)
                elif response.status_code >= 400:
                    self._handle_error_status(response)
                else:
                    data = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    calls = _parse_tool_calls(message.get("tool_calls"))
                    return ChatResponse(
                        content=content,
                        tool_calls=calls,
                        finish_reason=choice.get("finish_reason"),
                        usage=data.get("usage") or {},
                        model=data.get("model"),
                    )
            time.sleep(min(delay, 30))
        raise LLMError("The model provider kept failing. Try again in a moment.")

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield answer text deltas. Reasoning tokens are intentionally dropped."""
        self._check_configured()
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        attempts = 0
        while True:
            emitted = False
            try:
                with httpx.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                ) as response:
                    if response.status_code >= 400:
                        response.read()
                        if response.status_code in {429, 500, 502, 503, 504} and attempts < 2:
                            attempts += 1
                            time.sleep(2**attempts)
                            continue
                        self._handle_error_status(response)
                    for line in response.iter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if event.get("error"):
                            raise LLMError(
                                f"Model provider error: {str(event['error'])[:200]}"
                            )
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            text = delta.get("content")
                            if text:
                                emitted = True
                                yield text
                        if event.get("usage"):
                            log.debug("chat usage: %s", event["usage"])
                return
            except httpx.RequestError as exc:
                if attempts >= 2:
                    raise LLMError(f"The response stream broke: {exc}") from exc
                attempts += 1
                if emitted:
                    raise LLMError("The response stream broke mid-answer. Try again.") from exc
                time.sleep(2**attempts)


def get_llm(settings: Settings | None = None) -> LLMClient:
    return LLMClient(settings or get_settings())


def estimate_tokens(text: str) -> int:
    from ingestion.tokens import count_tokens

    return count_tokens(text)
