"""Embed single sentences, return the surrounding sentences to the caller."""

from __future__ import annotations

from collections.abc import Sequence

from ingestion.chunkers.base import ChunkDraft, Chunker, sentence_rows
from ingestion.parsers import Block


class SentenceWindowChunker(Chunker):
    name = "sentence_window"

    def split(self, blocks: Sequence[Block]) -> list[ChunkDraft]:
        rows = sentence_rows(blocks)
        drafts: list[ChunkDraft] = []
        for index, (sentence, block) in enumerate(rows):
            start = max(0, index - self.config.sentence_window_before)
            end = min(len(rows), index + self.config.sentence_window_after + 1)
            window = " ".join(text for text, _ in rows[start:end]).strip()
            drafts.append(
                ChunkDraft(
                    text=sentence,
                    window_text=window,
                    page_number=block.page_number,
                    heading_path=block.heading_path,
                )
            )
        return drafts
