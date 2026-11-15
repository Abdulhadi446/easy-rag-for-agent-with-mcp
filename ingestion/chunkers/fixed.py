"""Fixed token windows with configurable overlap, word-aligned."""

from __future__ import annotations

from collections.abc import Sequence

from ingestion.chunkers.base import (
    ChunkDraft,
    Chunker,
    Piece,
    build_draft,
    content_blocks,
    pack_pieces,
)
from ingestion.parsers import Block
from ingestion.tokens import count_tokens


class FixedChunker(Chunker):
    name = "fixed"

    def split(self, blocks: Sequence[Block]) -> list[ChunkDraft]:
        pieces: list[Piece] = []
        for block in content_blocks(blocks):
            for word_group in self._word_groups(block.text, self.config.chunk_size):
                pieces.append(
                    Piece(
                        text=word_group,
                        tokens=count_tokens(word_group),
                        page_number=block.page_number,
                        heading_path=block.heading_path,
                    )
                )
        drafts: list[ChunkDraft] = []
        for group in pack_pieces(pieces, self.config.chunk_size, self.config.chunk_overlap):
            draft = build_draft(group, separator=" ")
            if draft.text:
                drafts.append(draft)
        return drafts

    def _word_groups(self, text: str, max_tokens: int) -> list[str]:
        words = text.split()
        groups: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word]) if current else word
            if current and count_tokens(candidate) > max_tokens:
                groups.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            groups.append(" ".join(current))
        return groups
