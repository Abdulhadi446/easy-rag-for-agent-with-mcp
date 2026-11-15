"""Chunker factory: add a strategy by writing a class and registering it here."""

from __future__ import annotations

from ingestion.chunkers.base import (
    STRATEGIES,
    STRATEGY_LABELS,
    Chunker,
    ChunkerError,
    ChunkingConfig,
    EmbedFn,
)
from ingestion.chunkers.fixed import FixedChunker
from ingestion.chunkers.parent_child import ParentChildChunker
from ingestion.chunkers.recursive import RecursiveChunker
from ingestion.chunkers.semantic import SemanticChunker
from ingestion.chunkers.sentence import SentenceChunker
from ingestion.chunkers.sentence_window import SentenceWindowChunker

_REGISTRY: dict[str, type[Chunker]] = {
    RecursiveChunker.name: RecursiveChunker,
    FixedChunker.name: FixedChunker,
    SentenceChunker.name: SentenceChunker,
    SentenceWindowChunker.name: SentenceWindowChunker,
    ParentChildChunker.name: ParentChildChunker,
    SemanticChunker.name: SemanticChunker,
}


def make_chunker(
    strategy: str,
    config: ChunkingConfig,
    embed_fn: EmbedFn | None = None,
) -> Chunker:
    if strategy not in _REGISTRY:
        raise ChunkerError(
            f"Unknown chunking strategy {strategy!r}. Choose one of: {', '.join(STRATEGIES)}"
        )
    return _REGISTRY[strategy](config, embed_fn=embed_fn)


__all__ = [
    "STRATEGIES",
    "STRATEGY_LABELS",
    "Chunker",
    "ChunkerError",
    "ChunkingConfig",
    "make_chunker",
]
