"""One sentence per chunk."""

from __future__ import annotations

from collections.abc import Sequence

from ingestion.chunkers.base import ChunkDraft, Chunker, sentence_rows
from ingestion.parsers import Block
from ingestion.tokens import count_tokens, truncate_tokens


class SentenceChunker(Chunker):
    name = "sentence"

    def split(self, blocks: Sequence[Block]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for sentence, block in sentence_rows(blocks):
            # A single monstrous sentence is truncated for embedding sanity but kept
            # in the payload only up to the configured size.
            text = truncate_tokens(sentence, self.config.chunk_size) if count_tokens(
                sentence
            ) > self.config.chunk_size else sentence
            drafts.append(
                ChunkDraft(
                    text=text,
                    page_number=block.page_number,
                    heading_path=block.heading_path,
                )
            )
        return drafts
