"""Headings first, then paragraphs, then sentences — the default strategy."""

from __future__ import annotations

from collections.abc import Sequence

from ingestion.chunkers.base import (
    ChunkDraft,
    Chunker,
    ChunkingConfig,
    block_pieces,
    build_draft,
    content_blocks,
    group_by_section,
    pack_pieces,
)
from ingestion.parsers import Block


class RecursiveChunker(Chunker):
    name = "recursive"

    def split(self, blocks: Sequence[Block]) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        body = content_blocks(blocks)
        if not body:
            return drafts

        for section in group_by_section(body):
            pieces = []
            for block in section:
                pieces.extend(block_pieces(block, self.config.chunk_size))
            for group in pack_pieces(pieces, self.config.chunk_size, self.config.chunk_overlap):
                draft = build_draft(group)
                if draft.text:
                    drafts.append(draft)
        return drafts


def recursive_with(size: int, overlap: int) -> Chunker:
    """Used by parent-child chunks to subdivide a parent without changing strategy."""
    return RecursiveChunker(ChunkingConfig(chunk_size=size, chunk_overlap=overlap))
