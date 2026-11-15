"""Environment-backed configuration with sane zero-config defaults.

Everything the app needs is read here once. Provider calls live behind
``llm.py`` and ``embeddings.py`` so swapping a provider is an env change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_LLM_MODEL = "openrouter/free"
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"

# Zero-config knowledge base defaults. A user who uploads a PDF and starts chatting
# gets these; everything else lives under "Advanced settings".
DEFAULT_KB_SETTINGS: dict[str, object] = {
    "chunk_strategy": "recursive",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "sentence_window_before": 3,
    "sentence_window_after": 3,
    "parent_chunk_size": 1000,
    "child_chunk_size": 200,
    "semantic_threshold": 0.75,
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "candidates": 40,
    "rerank_enabled": True,
    "query_rewrite_enabled": False,
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    secret_key: str
    database_url: str
    upload_dir: Path
    max_upload_mb: int
    max_upload_bytes: int
    public_base_url: str
    mcp_public_url: str

    embeddings_provider: str
    embedding_model: str
    embedding_dim: int
    embedding_base_url: str | None
    embedding_api_key: str | None

    llm_base_url: str
    llm_model: str
    llm_api_key: str | None
    llm_timeout: float

    rerank_provider: str
    rerank_model: str
    cohere_api_key: str | None

    redis_url: str | None
    debug: bool
    testing: bool
    mcp_host: str
    mcp_port: int
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, overrides: dict[str, object] | None = None) -> Settings:
        upload_dir = Path(os.environ.get("UPLOAD_DIR", str(BASE_DIR / "uploads"))).expanduser()
        if not upload_dir.is_absolute():
            upload_dir = (BASE_DIR / upload_dir).resolve()
        max_mb = _env_int("MAX_UPLOAD_MB", 25)
        embeddings_provider = os.environ.get("EMBEDDINGS_PROVIDER", "local").strip().lower()
        embedding_dim = _env_int("EMBEDDING_DIM", DEFAULT_EMBEDDING_DIM)
        settings = cls(
            secret_key=os.environ.get("SECRET_KEY", "dev-only-change-me"),
            database_url=os.environ.get(
                "DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5432/rag"
            ),
            upload_dir=upload_dir,
            max_upload_mb=max_mb,
            max_upload_bytes=max_mb * 1024 * 1024,
            public_base_url=os.environ.get("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/"),
            mcp_public_url=os.environ.get(
                "MCP_PUBLIC_URL", "http://localhost:8001/mcp"
            ).rstrip("/"),
            embeddings_provider=embeddings_provider,
            embedding_model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            embedding_dim=embedding_dim,
            embedding_base_url=os.environ.get("EMBEDDING_BASE_URL") or None,
            embedding_api_key=(
                os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY") or None
            ),
            llm_base_url=os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/"),
            llm_model=os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_api_key=(
                os.environ.get("LLM_API_KEY")
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or None
            ),
            llm_timeout=_env_float("LLM_TIMEOUT", 120.0),
            rerank_provider=os.environ.get("RERANK_PROVIDER", "local").strip().lower(),
            rerank_model=os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
            cohere_api_key=os.environ.get("COHERE_API_KEY") or None,
            redis_url=os.environ.get("REDIS_URL") or None,
            debug=_env_bool("FLASK_DEBUG", False),
            testing=_env_bool("TESTING", False),
            mcp_host=os.environ.get("MCP_HOST", "127.0.0.1"),
            mcp_port=_env_int("MCP_PORT", 8001),
        )
        return settings

    @property
    def vector_dim(self) -> int:
        return self.embedding_dim


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(BASE_DIR / ".env", override=False)
    return Settings.from_env()
