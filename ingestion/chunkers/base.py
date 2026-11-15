"""Common chunker interface and the token-aware helpers they share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ingestion.parsers import Block
from ingestion.tokens import count_tokens, split_sentences, tail_tokens

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

STRATEGIES = (
    "recursive",
    "fixed",
    "sentence",
    "sentence_window",
    "parent_child",
    "semantic",
)

STRATEGY_LABELS = {
    "recursive": "Recursive (recommended)",
    "fixed": "Fixed size",
    "sentence": "One sentence per chunk",
    "sentence_window": "Sentence window",
    "parent_child": "Parent–child",
    "semantic": "Semantic",
}


class ChunkerError(RuntimeError):
    pass


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    sentence_window_before: int = 3
    sentence_window_after: int = 3
    parent_chunk_size: int = 1000
    child_chunk_size: int = 200
    semantic_threshold: float = 0.75

    @classmethod
    def from_kb(cls, kb: Any) -> ChunkingConfig:
        return cls(
            chunk_size=max(32, int(kb.chunk_size)),
            chunk_overlap=max(0, int(kb.chunk_overlap)),
            sentence_window_before=max(0, int(kb.sentence_window_before)),
            sentence_window_after=max(0, int(kb.sentence_window_after)),
            parent_chunk_size=max(64, int(kb.parent_chunk_size)),
            child_chunk_size=max(32, int(kb.child_chunk_size)),
            semantic_threshold=float(kb.semantic_threshold),
        )

    def normalised(self) -> ChunkingConfig:
        """Keep overlap strictly below size so packing always makes progress."""
        overlap = min(max(0, self.chunk_overlap), max(0, self.chunk_size - 1))
        return ChunkingConfig(
            chunk_size=max(8, self.chunk_size),
            chunk_overlap=overlap,
            sentence_window_before=max(0, self.sentence_window_before),
            sentence_window_after=max(0, self.sentence_window_after),
            parent_chunk_size=max(16, self.parent_chunk_size),
            child_chunk_size=max(8, self.child_chunk_size),
            semantic_threshold=min(0.999, max(-1.0, self.semantic_threshold)),
        )


@dataclass
class ChunkDraft:
    text: str
    page_number: int | None = None
    heading_path: str | None = None
    token_count: int = 0
    parent_index: int | None = None
    window_text: str | None = None
    role: str = "chunk"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = count_tokens(self.text)


@dataclass
class Piece:
    text: str
    tokens: int
    page_number: int | None
    heading_path: str | None


class Chunker(ABC):
    """Every strategy implements this; the pipeline only knows this interface."""

    name: str = "base"
    requires_embeddings = False

    def __init__(self, config: ChunkingConfig, embed_fn: EmbedFn | None = None) -> None:
        self.config = config.normalised()
        self.embed_fn = embed_fn

    @abstractmethod
    def split(self, blocks: Sequence[Block]) -> list[ChunkDraft]: ...

    def describe(self) -> dict:
        return {"strategy": self.name, **self.config.__dict__}


# --------------------------------------------------------------------------- helpers
def content_blocks(blocks: Iterable[Block]) -> list[Block]:
    """Drop standalone heading blocks: their text already lives in heading_path."""
    result = []
    for block in blocks:
        if block.kind == "heading":
            continue
        if block.text.strip():
            result.append(block)
    return result


def _hard_split_words(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word]) if current else word
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def block_pieces(block: Block, max_tokens: int) -> list[Piece]:
    pieces: list[Piece] = []
    paragraphs = [part.strip() for part in block.text.split("\n\n") if part.strip()]
    for paragraph in paragraphs or [block.text.strip()]:
        if not paragraph:
            continue
        if count_tokens(paragraph) <= max_tokens:
            pieces.append(
                Piece(
                    text=paragraph,
                    tokens=count_tokens(paragraph),
                    page_number=block.page_number,
                    heading_path=block.heading_path,
                )
            )
            continue
        for sentence in split_sentences(paragraph):
            if count_tokens(sentence) <= max_tokens:
                pieces.append(
                    Piece(
                        text=sentence,
                        tokens=count_tokens(sentence),
                        page_number=block.page_number,
                        heading_path=block.heading_path,
                    )
                )
            else:
                for fragment in _hard_split_words(sentence, max_tokens):
                    pieces.append(
                        Piece(
                            text=fragment,
                            tokens=count_tokens(fragment),
                            page_number=block.page_number,
                            heading_path=block.heading_path,
                        )
                    )
    return pieces


def pack_pieces(pieces: Sequence[Piece], size: int, overlap: int) -> list[list[Piece]]:
    """Greedy packing with a trailing overlap carried into the next window."""
    groups: list[list[Piece]] = []
    current: list[Piece] = []
    tokens = 0
    for piece in pieces:
        if current and tokens + piece.tokens > size:
            groups.append(current)
            carry: list[Piece] = []
            carried = 0
            for previous in reversed(current):
                if carried + previous.tokens > overlap:
                    break
                if carried + previous.tokens + piece.tokens > size:
                    break
                carry.insert(0, previous)
                carried += previous.tokens
            current = carry
            tokens = carried
        current.append(piece)
        tokens += piece.tokens
    if current:
        groups.append(current)
    return groups


def group_by_section(blocks: Sequence[Block]) -> list[list[Block]]:
    sections: list[list[Block]] = []
    for block in blocks:
        if sections and sections[-1][0].heading_path == block.heading_path:
            sections[-1].append(block)
        else:
            sections.append([block])
    return sections


def build_draft(group: Sequence[Piece], separator: str = "\n\n") -> ChunkDraft:
    first = group[0]
    return ChunkDraft(
        text=separator.join(piece.text for piece in group).strip(),
        page_number=first.page_number,
        heading_path=first.heading_path,
        token_count=sum(piece.tokens for piece in group),
    )


def sentence_rows(blocks: Sequence[Block]) -> list[tuple[str, Block]]:
    rows: list[tuple[str, Block]] = []
    for block in content_blocks(blocks):
        for sentence in split_sentences(block.text):
            if sentence.strip():
                rows.append((sentence.strip(), block))
    return rows


def last_sentences(text: str, limit: int) -> str:
    """Utility used by window chunks and tests."""
    return tail_tokens(text, limit)
