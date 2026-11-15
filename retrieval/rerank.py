"""Optional reranking with a local cross-encoder or Cohere's rerank API.

Any failure degrades to the fused order instead of breaking search.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import httpx
import numpy as np

from app.config import Settings, get_settings
from retrieval.base import SearchHit

log = logging.getLogger(__name__)

DEFAULT_LOCAL_RERANK = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker:
    name = "none"

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        return hits[:top_k]


class LocalCrossEncoderReranker(Reranker):
    name = "local"

    def __init__(self, model_id: str, cache_dir=None, batch_size: int = 16) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._tokenizer = None
        self._session = None
        self._input_names: set[str] = set()
        self._failed = False

    def _load(self) -> bool:
        if self._session is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._session is not None:
                return True
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer

                tokenizer_path = hf_hub_download(
                    self.model_id, "tokenizer.json", cache_dir=self.cache_dir
                )
                model_path = None
                for candidate in (
                    "onnx/model_quantized.onnx",
                    "onnx/model_int8.onnx",
                    "onnx/model.onnx",
                ):
                    try:
                        model_path = hf_hub_download(
                            self.model_id, candidate, cache_dir=self.cache_dir
                        )
                        break
                    except Exception:  # noqa: BLE001
                        continue
                if model_path is None:
                    raise RuntimeError("no ONNX weights found")

                tokenizer = Tokenizer.from_file(tokenizer_path)
                tokenizer.enable_truncation(max_length=512)
                tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
                self._tokenizer = tokenizer
                self._session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
                self._input_names = {item.name for item in self._session.get_inputs()}
                log.info("loaded local reranker %s", self.model_id)
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("local reranker unavailable (%s); using fused order", exc)
                self._failed = True
                return False

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        if not self._load():
            return hits[:top_k]
        assert self._tokenizer is not None and self._session is not None
        scores: list[float] = []
        try:
            for start in range(0, len(hits), self.batch_size):
                batch = [
                    (query, hit.text[:4000]) for hit in hits[start : start + self.batch_size]
                ]
                encodings = self._tokenizer.encode_batch(batch)
                ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
                mask = np.asarray([item.attention_mask for item in encodings], dtype=np.int64)
                types = np.asarray([item.type_ids for item in encodings], dtype=np.int64)
                feed = {"input_ids": ids, "attention_mask": mask}
                if "token_type_ids" in self._input_names:
                    feed["token_type_ids"] = types
                logits = self._session.run(None, feed)[0]
                logits = np.asarray(logits)
                if logits.ndim == 2 and logits.shape[1] == 1:
                    values = 1.0 / (1.0 + np.exp(-logits[:, 0]))
                elif logits.ndim == 2 and logits.shape[1] >= 2:
                    shifted = logits - logits.max(axis=1, keepdims=True)
                    exp = np.exp(shifted)
                    values = (exp / exp.sum(axis=1, keepdims=True))[:, -1]
                else:
                    values = logits.reshape(-1)
                scores.extend(float(value) for value in values)
        except Exception as exc:  # noqa: BLE001
            log.warning("rerank failed (%s); using fused order", exc)
            return hits[:top_k]

        reranked: list[SearchHit] = []
        for hit, score in zip(hits, scores, strict=False):
            hit = SearchHit(**{**hit.__dict__})
            hit.score = score
            hit.raw_score = score
            hit.source = "rerank"
            reranked.append(hit)
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]


class CohereReranker(Reranker):
    name = "cohere"

    def __init__(self, api_key: str, model: str = "rerank-v3.5") -> None:
        self.api_key = api_key
        self.model = model

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        try:
            response = httpx.post(
                "https://api.cohere.com/v2/rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [hit.text for hit in hits],
                    "top_n": min(top_k, len(hits)),
                },
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("Cohere rerank failed (%s); using fused order", exc)
            return hits[:top_k]

        reranked: list[SearchHit] = []
        for item in payload.get("results", []):
            index = int(item.get("index", -1))
            if 0 <= index < len(hits):
                hit = SearchHit(**{**hits[index].__dict__})
                hit.score = float(item.get("relevance_score", 0.0))
                hit.raw_score = hit.score
                hit.source = "rerank"
                reranked.append(hit)
        return reranked[:top_k] or hits[:top_k]


@lru_cache(maxsize=4)
def _build(provider: str, model: str, api_key: str | None) -> Reranker | None:
    if provider in {"", "none", "off", "false"}:
        return None
    if provider == "local":
        return LocalCrossEncoderReranker(model or DEFAULT_LOCAL_RERANK)
    if provider == "cohere":
        return CohereReranker(api_key, model or "rerank-v3.5") if api_key else None
    log.warning("unknown RERANK_PROVIDER=%r; reranking disabled", provider)
    return None


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    settings = settings or get_settings()
    return _build(settings.rerank_provider, settings.rerank_model, settings.cohere_api_key)
