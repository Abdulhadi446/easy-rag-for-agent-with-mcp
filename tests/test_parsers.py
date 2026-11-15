"""Parser structure: headings, pages, outline and friendly failures."""

from __future__ import annotations

import pymupdf

from ingestion.parsers import (
    EmptyDocumentError,
    parse_markdown,
    parse_pdf,
    parse_text,
)

MARKDOWN = """---
title: Billing guide
---

# Billing

Invoices are generated monthly.

## Refunds

Refunds take five business days.

### Partial refunds

Partial refunds are prorated.

## Support

Email support for edge cases.
"""


def test_markdown_keeps_heading_paths_and_outline():
    document = parse_markdown(MARKDOWN)
    assert document.meta["format"] == "markdown"
    paragraphs = [block for block in document.blocks if block.kind == "paragraph"]
    assert len(paragraphs) == 4
    assert paragraphs[1].heading_path == "Billing > Refunds"
    assert paragraphs[2].heading_path == "Billing > Refunds > Partial refunds"
    assert paragraphs[3].heading_path == "Billing > Support"
    assert [item["level"] for item in document.outline] == [1, 2, 3, 2]
    assert document.outline[0]["title"] == "Billing"


def test_markdown_ignores_hash_inside_code_fence():
    document = parse_markdown("# Real\n\n```\n# not a heading\n```\n\nBody text.")
    titles = [item["title"] for item in document.outline]
    assert titles == ["Real"]


def test_plain_text_splits_paragraphs():
    document = parse_text("First paragraph.\n\nSecond paragraph.\n\nThird.")
    assert [block.text for block in document.blocks] == [
        "First paragraph.",
        "Second paragraph.",
        "Third.",
    ]
    assert document.page_count is None


def test_empty_text_is_a_friendly_error():
    try:
        parse_text("   \n\n  ")
    except EmptyDocumentError as exc:
        assert "empty" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected EmptyDocumentError")


def test_pdf_extracts_pages_and_heading_sizes(tmp_path):
    path = tmp_path / "manual.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 80), "Returns", fontsize=26)
    page.insert_text((72, 120), "Refunds are processed within five business days.", fontsize=11)
    page.insert_text((72, 160), "Shipping is free above fifty dollars.", fontsize=11)
    page2 = document.new_page()
    page2.insert_text((72, 80), "Support", fontsize=22)
    page2.insert_text((72, 120), "Email support for invoices.", fontsize=11)
    document.save(path)
    document.close()

    parsed = parse_pdf(path)
    assert parsed.page_count == 2
    assert parsed.meta["format"] == "pdf"
    paragraphs = [block for block in parsed.blocks if block.kind == "paragraph"]
    assert len(paragraphs) == 3
    assert paragraphs[0].page_number == 1
    assert paragraphs[0].heading_path == "Returns"
    assert paragraphs[-1].page_number == 2
    assert [item["title"] for item in parsed.outline] == ["Returns", "Support"]


def test_scanned_pdf_reports_ocr_not_supported(tmp_path):
    path = tmp_path / "scan.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    try:
        parse_pdf(path)
    except EmptyDocumentError as exc:
        assert "scanned" in str(exc).lower()
        assert "ocr" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected EmptyDocumentError")
