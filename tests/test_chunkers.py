"""Chunker behaviour: sizes, overlap, parent links and sentence windows."""

from __future__ import annotations

import pytest

from ingestion.chunkers import ChunkingConfig, make_chunker
from ingestion.parsers import Block
from ingestion.tokens import count_tokens, split_sentences

LONG_TEXT = (
    "Retrieval augmented generation connects a language model to a document store. "
    "The retriever finds passages that are likely to answer the question. "
    "A reranker then reorders those passages with a cross encoder. "
    "Finally the generator writes an answer and cites the passages it used. "
) * 12

BLOCKS = [
    Block(text="Billing", heading_path="Billing", heading_level=1, kind="heading"),
    Block(
        text=(
            "Invoices are generated on the first of each month. "
            "You can download them from the billing page. "
            "Refunds are processed within five business days."
        ),
        page_number=4,
        heading_path="Billing",
    ),
    Block(text="Support", heading_path="Support", heading_level=1, kind="heading"),
    Block(
        text=(
            "Support is available around the clock. "
            "Use the chat widget for the fastest response. "
            "Email works too, but replies take longer."
        ),
        page_number=5,
        heading_path="Support",
    ),
]


def config(**overrides) -> ChunkingConfig:
    base = dict(chunk_size=80, chunk_overlap=20, parent_chunk_size=200, child_chunk_size=60)
    base.update(overrides)
    return ChunkingConfig(**base)


def test_recursive_chunks_respect_size_and_overlap():
    chunker = make_chunker("recursive", config())
    chunks = chunker.split([Block(text=LONG_TEXT, heading_path="Guide", page_number=3)])
    assert len(chunks) > 2
    assert all(chunk.token_count <= 80 for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert all(chunk.heading_path == "Guide" for chunk in chunks)
    # Overlap: something from the end of one chunk appears in the next.
    overlaps = 0
    for previous, following in zip(chunks, chunks[1:], strict=False):
        tail = previous.text.split()[-3:]
        if any(word in following.text for word in tail):
            overlaps += 1
    assert overlaps >= 1


def test_recursive_never_crosses_headings():
    chunker = make_chunker("recursive", config(chunk_size=40, chunk_overlap=5))
    chunks = chunker.split(BLOCKS)
    assert chunks
    for chunk in chunks:
        assert "Invoices" not in chunk.text or "Support" not in chunk.text


def test_fixed_size_is_word_aligned_and_overlaps():
    chunker = make_chunker("fixed", config(chunk_size=60, chunk_overlap=15))
    chunks = chunker.split([Block(text=LONG_TEXT)])
    assert len(chunks) > 3
    assert all(count_tokens(chunk.text) <= 60 for chunk in chunks)
    assert chunks[0].text.split()[0] == "Retrieval"
    # Windows were cut at whitespace, so no chunk begins mid-word.
    assert all(not chunk.text.startswith(" ") for chunk in chunks)


def test_sentence_chunker_emits_one_sentence_per_chunk():
    chunker = make_chunker("sentence", config())
    chunks = chunker.split(BLOCKS)
    expected = sum(len(split_sentences(block.text)) for block in BLOCKS if block.kind != "heading")
    assert len(chunks) == expected
    for chunk in chunks:
        assert chunk.text.endswith((".", "!", "?")) or chunk.text.endswith("days")


def test_sentence_window_stores_neighbours_without_embedding_them():
    chunker = make_chunker(
        "sentence_window", config(sentence_window_before=1, sentence_window_after=1)
    )
    chunks = chunker.split([BLOCKS[1]])
    assert len(chunks) == 3
    assert chunks[0].window_text is not None
    assert "Invoices are generated" in chunks[0].text
    assert "billing page" in chunks[0].window_text
    assert len(chunks[1].window_text.split(".")) >= 3
    # The embedded text stays a single sentence.
    assert chunks[1].text.count(".") == 1


def test_parent_child_links_every_child_to_a_parent():
    chunker = make_chunker("parent_child", config(parent_chunk_size=100, child_chunk_size=40))
    drafts = chunker.split([Block(text=LONG_TEXT, page_number=2, heading_path="Guide")])
    parents = [index for index, draft in enumerate(drafts) if draft.role == "parent"]
    children = [draft for draft in drafts if draft.role == "child"]
    assert parents
    assert len(children) > len(parents)
    assert all(draft.parent_index in parents for draft in children)
    assert all(draft.token_count <= 40 for draft in children)
    assert all(draft.text.strip() for draft in children)


def test_semantic_chunker_splits_on_topic_change():
    blocks = [
        Block(
            text="Alpha systems use blue headers. Alpha systems retry on failure.",
            heading_path="Alpha",
        ),
        Block(
            text="Beta systems use green headers. Beta systems never retry.",
            heading_path="Beta",
        ),
    ]
    vectors = {
        "Alpha systems use blue headers.": [1.0, 0.0],
        "Alpha systems retry on failure.": [0.99, 0.01],
        "Beta systems use green headers.": [0.0, 1.0],
        "Beta systems never retry.": [0.02, 0.98],
    }

    def fake_embed(texts):
        return [vectors[text] for text in texts]

    chunker = make_chunker("semantic", config(semantic_threshold=0.5), embed_fn=fake_embed)
    chunks = chunker.split(blocks)
    assert len(chunks) == 2
    assert "Alpha" in chunks[0].text
    assert "Beta" in chunks[1].text


def test_semantic_without_embeddings_raises():
    from ingestion.chunkers import ChunkerError

    with pytest.raises(ChunkerError):
        make_chunker("semantic", config()).split([Block(text="Hello there.")])


def test_unknown_strategy_is_rejected():
    from ingestion.chunkers import ChunkerError

    with pytest.raises(ChunkerError):
        make_chunker("magic", config())


def test_chunk_indices_are_assigned_by_the_pipeline_not_the_chunker():
    # Chunkers return drafts without ids; the pipeline owns ordering/indices.
    chunker = make_chunker("recursive", config())
    drafts = chunker.split(BLOCKS)
    assert all(not hasattr(draft, "chunk_index") for draft in drafts)
