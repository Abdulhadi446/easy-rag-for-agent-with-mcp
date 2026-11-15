"""Large parents for context, small embedded children for precise matching."""

from __future__ import annotations

from collections.abc import Sequence

from ingestion.chunkers.base import ChunkDraft, Chunker, ChunkingConfig, content_blocks
from ingestion.chunkers.recursive import recursive_with
from ingestion.parsers import Block


class ParentChildChunker(Chunker):
    name = "parent_child"

    def split(self, blocks: Sequence[Block]) -> list[ChunkDraft]:
        body = content_blocks(blocks)
        if not body:
            return []

        parent_splitter = recursive_with(
            self.config.parent_chunk_size, min(50, self.config.parent_chunk_size // 10)
        )
        parents = parent_splitter.split(body)

        child_config = ChunkingConfig(
            chunk_size=self.config.child_chunk_size,
            chunk_overlap=0,
        ).normalised()
        child_splitter = recursive_with(child_config.chunk_size, 0)

        drafts: list[ChunkDraft] = []
        for parent in parents:
            parent_index = len(drafts)
            drafts.append(
                ChunkDraft(
                    text=parent.text,
                    page_number=parent.page_number,
                    heading_path=parent.heading_path,
                    token_count=parent.token_count,
                    role="parent",
                    extra={"role": "parent"},
                )
            )
            pseudo_block = Block(
                text=parent.text,
                page_number=parent.page_number,
                heading_path=parent.heading_path,
            )
            children = child_splitter.split([pseudo_block])
            if not children:
                # A very small parent still needs one searchable child.
                drafts.append(
                    ChunkDraft(
                        text=parent.text,
                        page_number=parent.page_number,
                        heading_path=parent.heading_path,
                        parent_index=parent_index,
                        role="child",
                    )
                )
                continue
            for child in children:
                drafts.append(
                    ChunkDraft(
                        text=child.text,
                        page_number=child.page_number or parent.page_number,
                        heading_path=child.heading_path or parent.heading_path,
                        token_count=child.token_count,
                        parent_index=parent_index,
                        role="child",
                    )
                )
        return drafts
