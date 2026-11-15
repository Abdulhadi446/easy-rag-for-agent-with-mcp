"""Retrieval evaluation: hit@k for every chunking × retrieval combination.

    python -m evaluation --docs samples --dataset evaluation/dataset.json

Creates a throwaway user and one knowledge base per strategy, ingests the sample
documents directly (no HTTP or worker), asks every question, and prints a table.
Nothing is left behind: deleting the user cascades to knowledge bases, documents and
chunks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import delete

from app.config import get_settings
from app.db import db
from app.models import Chunk, Document, DocumentStatus, KnowledgeBase, User
from embeddings import build_embeddings
from ingestion.chunkers import ChunkingConfig, make_chunker
from ingestion.parsers import parse_upload
from retrieval import retrieve

STRATEGIES = ["recursive", "fixed", "sentence", "sentence_window", "parent_child", "semantic"]
MODES = ["vector", "keyword", "hybrid"]


def normalise(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def setup_user(embedding_model: str, embedding_dim: int) -> User:
    user = User(
        email=f"eval-{uuid.uuid4().hex}@example.invalid",
        password_hash="!",
        display_name="Evaluation",
    )
    db.session.add(user)
    db.session.flush()
    assert user.id is not None
    return user


def ingest_documents(kb: KnowledgeBase, paths: list[Path], provider, strategy: str) -> int:
    embed_fn = provider.embed_documents if strategy == "semantic" else None
    chunker = make_chunker(strategy, ChunkingConfig.from_kb(kb), embed_fn=embed_fn)
    total = 0
    for path in paths:
        parsed = parse_upload(path, path.name)
        drafts = chunker.split(parsed.blocks)
        document = Document(
            knowledge_base_id=kb.id,
            filename=path.name,
            content_type="text/markdown" if path.suffix != ".txt" else "text/plain",
            size_bytes=path.stat().st_size,
            sha256=uuid.uuid4().hex + uuid.uuid4().hex,
            status=DocumentStatus.READY,
            page_count=parsed.page_count,
            chunk_count=len([draft for draft in drafts if draft.role != "parent"]),
            outline=parsed.outline,
        )
        db.session.add(document)
        db.session.flush()

        indices = [index for index, draft in enumerate(drafts) if draft.role != "parent"]
        vectors = provider.embed_documents([drafts[index].text for index in indices]) if indices else []
        vector_by_index = dict(zip(indices, vectors, strict=True))

        id_by_index: dict[int, uuid.UUID] = {}
        for index, draft in enumerate(drafts):
            chunk = Chunk(
                document_id=document.id,
                knowledge_base_id=kb.id,
                text=draft.text,
                embedding=vector_by_index.get(index),
                page_number=draft.page_number,
                heading_path=draft.heading_path,
                chunk_index=index,
                token_count=draft.token_count,
                window_text=draft.window_text,
                parent_id=id_by_index.get(draft.parent_index) if draft.parent_index is not None else None,
                extra=draft.extra or {},
            )
            db.session.add(chunk)
            db.session.flush()
            id_by_index[index] = chunk.id
            total += 1
    db.session.commit()
    return total


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    provider = build_embeddings(settings)
    dataset = json.loads(Path(args.dataset).read_text())
    questions = dataset["questions"]
    docs = sorted(Path(args.docs).glob("*"))
    docs = [path for path in docs if path.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}]
    if not docs:
        raise SystemExit(f"No documents found in {args.docs}")

    user = setup_user(provider.model_id, provider.dim)
    results: dict[str, dict[str, float]] = {}
    per_question: list[dict[str, Any]] = []

    for strategy in args.strategies:
        kb = KnowledgeBase(
            user_id=user.id,
            name=f"Eval {strategy}",
            embedding_model=provider.model_id,
            embedding_dim=provider.dim,
            chunk_strategy=strategy,
            retrieval_mode="hybrid",
            rerank_enabled=args.rerank,
        )
        db.session.add(kb)
        db.session.flush()
        chunk_count = ingest_documents(kb, docs, provider, strategy)
        print(f"… {strategy}: {chunk_count} chunks", file=sys.stderr)

        results[strategy] = {}
        for mode in args.modes:
            hits = 0
            reciprocal_rank = 0.0
            for item in questions:
                found = retrieve(
                    kb, item["question"], top_k=args.top_k, mode=mode, rerank=args.rerank
                )
                expected = normalise(item["expected"])
                rank = next(
                    (
                        position
                        for position, hit in enumerate(found, start=1)
                        if expected in normalise(hit.text)
                    ),
                    None,
                )
                if rank is not None:
                    hits += 1
                    reciprocal_rank += 1.0 / rank
                per_question.append(
                    {
                        "strategy": strategy,
                        "mode": mode,
                        "question": item["question"],
                        "hit": rank is not None,
                        "rank": rank,
                        "top": [hit.citation for hit in found[:3]],
                    }
                )
            results[strategy][mode] = round(hits / len(questions), 3)
            results[strategy][f"{mode}_mrr"] = round(reciprocal_rank / len(questions), 3)

    db.session.execute(delete(User).where(User.id == user.id))
    db.session.commit()
    return {
        "dataset": dataset.get("name", Path(args.dataset).stem),
        "questions": len(questions),
        "top_k": args.top_k,
        "rerank": args.rerank,
        "hit_rate": results,
        "per_question": per_question,
    }


def print_table(report: dict[str, Any], modes: list[str]) -> None:
    print(f"\nHit@{report['top_k']} — {report['dataset']} ({report['questions']} questions)\n")
    header = f"{'strategy':<16}" + "".join(f"{mode:>10}" for mode in modes)
    print(header)
    print("-" * len(header))
    for strategy, scores in report["hit_rate"].items():
        row = f"{strategy:<16}" + "".join(f"{scores[mode]:>10.0%}" for mode in modes)
        print(row)
    print("\nMRR (mean reciprocal rank)\n")
    print(header)
    print("-" * len(header))
    for strategy, scores in report["hit_rate"].items():
        row = f"{strategy:<16}" + "".join(f"{scores[mode + '_mrr']:>10.2f}" for mode in modes)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="samples")
    parser.add_argument("--dataset", default="evaluation/dataset.json")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    args.strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    args.modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    for strategy in args.strategies:
        if strategy not in STRATEGIES:
            raise SystemExit(f"Unknown strategy {strategy}; choose from {', '.join(STRATEGIES)}")
    for mode in args.modes:
        if mode not in MODES:
            raise SystemExit(f"Unknown mode {mode}; choose from {', '.join(MODES)}")

    from app import create_app
    from app.logging_setup import configure_logging

    configure_logging()
    app = create_app()
    with app.app_context():
        report = evaluate(args)
    print_table(report, args.modes)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
