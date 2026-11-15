"""Split where consecutive sentences stop being similar to each other."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ingestion.chunkers.base import (
    ChunkDraft,
    Chunker,
    ChunkerError,
    sentence_rows,
)
from ingestion.parsers import Block
from ingestion.tokens import count_tokens


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


class SemanticChunker(Chunker):
    name = "semantic"
    requires_embeddings = True

    def split(self, blocks: Sequence[Block]) -> list[ChunkDraft]:
        if self.embed_fn is None:
            raise ChunkerError(
                "The semantic chunker needs an embedding provider. "
                "Configure embeddings or choose another strategy."
            )
        rows = sentence_rows(blocks)
        if not rows:
            return []

        vectors = self.embed_fn([text for text, _ in rows])
        if len(vectors) != len(rows):
            raise ChunkerError("Embedding provider returned the wrong number of vectors")

        drafts: list[ChunkDraft] = []
        current: list[tuple[str, Block]] = [rows[0]]
        current_tokens = count_tokens(rows[0][0])

        def flush() -> None:
            nonlocal current, current_tokens
            first_block = current[0][1]
            drafts.append(
                ChunkDraft(
                    text=" ".join(text for text, _ in current).strip(),
                    page_number=first_block.page_number,
                    heading_path=first_block.heading_path,
                    token_count=current_tokens,
                )
            )
            current = []
            current_tokens = 0

        for index in range(1, len(rows)):
            text, block = rows[index]
            similarity = _cosine(vectors[index - 1], vectors[index])
            tokens = count_tokens(text)
            if similarity < self.config.semantic_threshold or (
                current_tokens + tokens > self.config.chunk_size
            ):
                flush()
                current = [(text, block)]
                current_tokens = tokens
            else:
                current.append((text, block))
                current_tokens += tokens

        if current:
            flush()
        return [draft for draft in drafts if draft.text]
