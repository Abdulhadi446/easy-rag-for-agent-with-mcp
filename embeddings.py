"""Embedding providers behind one interface.

Default is a local ONNX model: no API key, no network at query time, ~35 MB
downloaded once. ``EMBEDDINGS_PROVIDER=openai`` switches to any OpenAI-compatible
``/embeddings`` endpoint; ``hash`` is a deterministic stand-in for tests.

Every knowledge base stores the model id and dimension it was built with, so a
provider swap can never silently mix vector spaces.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "easy-rag-mcp" / "models"
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingError(RuntimeError):
    """A provider problem worth showing to the user."""


class Embeddings(ABC):
    model_id: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Vectors for chunks. Order matches the input."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self.query_prefix + text])[0]

    @property
    def query_prefix(self) -> str:
        return ""


# --------------------------------------------------------------------------- local
class LocalOnnxEmbeddings(Embeddings):
    """Hugging Face ONNX sentence encoder (CLS or mean pooling, L2 normalised)."""

    def __init__(
        self,
        model_id: str = "Xenova/bge-small-en-v1.5",
        dim: int = 384,
        cache_dir: Path | None = None,
        pooling: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.dim = dim
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.pooling = pooling or ("cls" if "bge" in model_id.lower() else "mean")
        self.batch_size = batch_size
        self.max_length = max_length
        self._lock = threading.Lock()
        self._tokenizer = None
        self._session = None
        self._input_names: set[str] = set()

    # ------------------------------------------------------------- loading
    def _resolve_onnx(self) -> str:
        candidates = [
            "onnx/model_quantized.onnx",
            "onnx/model_int8.onnx",
            "onnx/model.onnx",
        ]
        local = Path(self.model_id).expanduser()
        if local.is_dir():
            for candidate in candidates:
                path = local / candidate
                if path.is_file():
                    return str(path)
            raise EmbeddingError(f"No ONNX model found under {local}")

        from huggingface_hub import hf_hub_download

        errors: list[str] = []
        for candidate in candidates:
            try:
                return hf_hub_download(self.model_id, candidate, cache_dir=str(self.cache_dir))
            except Exception as exc:  # noqa: BLE001 - try the next variant
                errors.append(f"{candidate}: {type(exc).__name__}")
        raise EmbeddingError(
            f"Could not download {self.model_id} ({'; '.join(errors)}). "
            "Check network access or set EMBEDDING_MODEL to a local ONNX directory."
        )

    def _load(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            local = Path(self.model_id).expanduser()
            tokenizer_path = (
                str(local / "tokenizer.json")
                if local.is_dir()
                else hf_hub_download(
                    self.model_id, "tokenizer.json", cache_dir=str(self.cache_dir)
                )
            )
            tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer.enable_truncation(max_length=self.max_length)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = max(1, min(4, (len(__import__("os").sched_getaffinity(0)))))
            session = ort.InferenceSession(
                self._resolve_onnx(), sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._tokenizer = tokenizer
            self._session = session
            self._input_names = {item.name for item in session.get_inputs()}
            log.info("loaded local embedding model %s", self.model_id)

    # ------------------------------------------------------------- encoding
    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        self._load()
        assert self._tokenizer is not None and self._session is not None
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [text if text.strip() else " " for text in texts[start : start + self.batch_size]]
            encodings = self._tokenizer.encode_batch(batch)
            ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
            mask = np.asarray([encoding.attention_mask for encoding in encodings], dtype=np.int64)
            feed: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            outputs = self._session.run(None, feed)
            hidden = outputs[0]
            if hidden.ndim == 2:
                pooled = hidden
            elif self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                weights = mask[..., None].astype(np.float32)
                pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
            pooled = pooled.astype(np.float32)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, 1e-9, None)
            vectors.extend(pooled.tolist())
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return self._encode(texts)
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Local embeddings failed: {exc}") from exc

    @property
    def query_prefix(self) -> str:
        return _QUERY_INSTRUCTION if "bge" in self.model_id.lower() else ""


# ------------------------------------------------------------------------- openai
class OpenAIEmbeddings(Embeddings):
    def __init__(
        self,
        model_id: str,
        dim: int,
        api_key: str | None,
        base_url: str | None = None,
        batch_size: int = 96,
        timeout: float = 60.0,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise EmbeddingError(
                "EMBEDDINGS_PROVIDER=openai needs EMBEDDING_API_KEY or OPENAI_API_KEY."
            )
        self.model_id = model_id
        self.dim = dim
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            except httpx.RequestError as exc:
                if attempt == self.max_retries:
                    raise EmbeddingError(f"Could not reach {url}: {exc}") from exc
            else:
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt == self.max_retries:
                        raise EmbeddingError(
                            f"Embedding provider returned {response.status_code}: "
                            f"{response.text[:200]}"
                        )
                    retry_after = response.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                elif response.status_code >= 400:
                    raise EmbeddingError(
                        f"Embedding provider returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                else:
                    return response.json()
            time.sleep(min(delay, 30))
        raise EmbeddingError("Embedding provider kept failing")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [text if text.strip() else " " for text in texts[start : start + self.batch_size]]
            payload = self._post({"model": self.model_id, "input": batch})
            data = sorted(payload.get("data", []), key=lambda item: item["index"])
            vectors.extend([item["embedding"] for item in data])
        for vector in vectors:
            if len(vector) != self.dim:
                raise EmbeddingError(
                    f"{self.model_id} returned {len(vector)}-dim vectors, but this database was "
                    f"created for {self.dim}. Set EMBEDDING_DIM and re-embed."
                )
        return vectors


# --------------------------------------------------------------------------- hash
class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing. For tests and offline demos only."""

    def __init__(self, dim: int = 384, model_id: str = "hash-bow-v1") -> None:
        self.dim = dim
        self.model_id = model_id

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


# ------------------------------------------------------------------------ factory
@lru_cache(maxsize=8)
def _cached(
    provider: str,
    model: str,
    dim: int,
    pooling: str,
    api_key: str | None,
    base_url: str | None,
) -> Embeddings:
    if provider == "local":
        return LocalOnnxEmbeddings(model_id=model, dim=dim, pooling=pooling)
    if provider == "openai":
        return OpenAIEmbeddings(model_id=model, dim=dim, api_key=api_key, base_url=base_url)
    if provider == "hash":
        return HashEmbeddings(dim=dim)
    raise EmbeddingError(
        f"Unknown EMBEDDINGS_PROVIDER={provider!r} (expected local, openai or hash)"
    )


def build_embeddings(settings: Settings) -> Embeddings:
    return _cached(
        (settings.embeddings_provider or "local").lower(),
        settings.embedding_model,
        settings.embedding_dim,
        settings.embedding_pooling,
        settings.embedding_api_key,
        settings.embedding_base_url,
    )


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Process-wide provider instance (the ONNX session is expensive to build)."""
    return build_embeddings(settings or get_settings())


def embed_query(text: str) -> list[float]:
    return get_embeddings().embed_query(text)
