"""Document parsing that keeps structure: pages, headings, paragraphs.

PDFs use PyMuPDF with a font-size heuristic for headings, so retrieval results can
say "page 7 · Returns > Refunds" instead of dumping anonymous text.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
MAX_HEADING_WORDS = 24


class ParseError(RuntimeError):
    """A file could not be parsed; the message is shown to the user."""


class EmptyDocumentError(ParseError):
    """The file parsed successfully but contains no usable text."""


@dataclass
class Block:
    text: str
    page_number: int | None = None
    heading_path: str | None = None
    heading_level: int | None = None
    kind: str = "paragraph"


@dataclass
class ParsedDocument:
    blocks: list[Block]
    outline: list[dict] = field(default_factory=list)
    page_count: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)


_WS_RE = re.compile(r"[ \t\u00a0]+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DEHYPHEN_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_BLANK_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = _CONTROL_RE.sub("", text)
    text = _DEHYPHEN_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw[:4096]:
        raise ParseError("This file looks binary. Upload a PDF, Markdown or plain text file.")
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ParseError("Could not decode this file as UTF-8 or Latin-1 text.")


# ------------------------------------------------------------------------- markdown
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_RE = re.compile(r"^\s*(=+|-{2,})\s*$")


def parse_markdown(text: str, *, filename: str = "") -> ParsedDocument:
    text = text.lstrip("\ufeff")
    # Drop YAML frontmatter.
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            text = text[end + 4 :].lstrip("\n")

    lines = text.split("\n")
    blocks: list[Block] = []
    outline: list[dict] = []
    heading_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        if not buffer:
            return
        paragraph = clean_text("\n".join(buffer))
        buffer.clear()
        if paragraph:
            path = " > ".join(title for _, title in heading_stack) or None
            blocks.append(Block(text=paragraph, heading_path=path))

    def push_heading(level: int, title: str) -> None:
        title = clean_text(title)
        if not title:
            return
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        outline.append(
            {
                "level": level,
                "title": title,
                "page": None,
                "heading_path": " > ".join(item[1] for item in heading_stack),
            }
        )

    index = 0
    while index < len(lines):
        line = lines[index]
        fence = _FENCE_RE.match(line)
        if fence:
            if not in_fence:
                flush()
                in_fence = True
                fence_marker = fence.group(1)
                buffer.append(line)
            elif fence.group(1) == fence_marker:
                buffer.append(line)
                flush()
                in_fence = False
            else:
                buffer.append(line)
            index += 1
            continue

        if in_fence:
            buffer.append(line)
            index += 1
            continue

        heading = _ATX_RE.match(line)
        if heading:
            flush()
            push_heading(len(heading.group(1)), heading.group(2))
            index += 1
            continue

        # Setext heading: a single line underlined with === or ---
        if (
            line.strip()
            and index + 1 < len(lines)
            and _SETEXT_RE.match(lines[index + 1])
            and not line.lstrip().startswith(("-", "*", ">", "|"))
        ):
            flush()
            push_heading(1 if lines[index + 1].strip().startswith("=") else 2, line)
            index += 2
            continue

        if not line.strip():
            flush()
            index += 1
            continue

        buffer.append(line)
        index += 1

    flush()
    if not blocks:
        raise EmptyDocumentError("This Markdown file has no readable text.")
    return ParsedDocument(
        blocks=blocks,
        outline=outline,
        page_count=None,
        meta={"format": "markdown", "filename": filename},
    )


# ------------------------------------------------------------------------------ txt
def parse_text(text: str, *, filename: str = "") -> ParsedDocument:
    text = clean_text(text)
    if not text.strip():
        raise EmptyDocumentError("This text file is empty.")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) == 1 and "\n" in paragraphs[0]:
        paragraphs = [line.strip() for line in paragraphs[0].split("\n") if line.strip()]
    blocks = [Block(text=paragraph) for paragraph in paragraphs]
    return ParsedDocument(
        blocks=blocks, outline=[], page_count=None, meta={"format": "text", "filename": filename}
    )


# ------------------------------------------------------------------------------ pdf
def _heading_level(size: float, body_size: float) -> int | None:
    if body_size <= 0:
        return None
    ratio = size / body_size
    if ratio >= 1.55:
        return 1
    if ratio >= 1.3:
        return 2
    if ratio >= 1.15:
        return 3
    return None


def parse_pdf(path: Path, *, filename: str = "") -> ParsedDocument:
    import pymupdf

    try:
        document = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"This PDF could not be opened: {exc}") from exc

    try:
        if document.needs_pass:
            raise ParseError("This PDF is password protected. Remove the password and re-upload.")
        page_count = document.page_count
        if page_count == 0:
            raise EmptyDocumentError("This PDF has no pages.")

        # Pass 1: body font size, weighted by how much text uses it.
        sizes: list[float] = []
        pages_data = []
        for page in document:
            data = page.get_text("dict")
            pages_data.append(data)
            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        content = (span.get("text") or "").strip()
                        if content:
                            sizes.extend([float(span.get("size", 0))] * max(1, len(content)))
        body_size = statistics.median(sizes) if sizes else 0.0

        blocks: list[Block] = []
        outline: list[dict] = []
        heading_stack: list[tuple[int, str]] = []
        for page_index, data in enumerate(pages_data, start=1):
            for raw_block in data.get("blocks", []):
                if raw_block.get("type", 0) != 0:
                    continue
                lines = raw_block.get("lines", [])
                text_parts: list[str] = []
                max_size = 0.0
                for line in lines:
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    line_text = line_text.strip()
                    if line_text:
                        text_parts.append(line_text)
                    for span in line.get("spans", []):
                        max_size = max(max_size, float(span.get("size", 0)))
                content = clean_text(" ".join(text_parts))
                if not content:
                    continue

                level = _heading_level(max_size, body_size)
                word_count = len(content.split())
                is_heading = (
                    level is not None
                    and word_count <= MAX_HEADING_WORDS
                    and not content.endswith((".", ";", ":"))
                )
                if is_heading:
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, content))
                    heading_path = " > ".join(title for _, title in heading_stack)
                    outline.append(
                        {
                            "level": level,
                            "title": content,
                            "page": page_index,
                            "heading_path": heading_path,
                        }
                    )
                    blocks.append(
                        Block(
                            text=content,
                            page_number=page_index,
                            heading_path=heading_path,
                            heading_level=level,
                            kind="heading",
                        )
                    )
                    continue

                heading_path = " > ".join(title for _, title in heading_stack) or None
                blocks.append(
                    Block(
                        text=content,
                        page_number=page_index,
                        heading_path=heading_path,
                        kind="paragraph",
                    )
                )

        if not any(block.kind == "paragraph" for block in blocks):
            raise EmptyDocumentError(
                "This PDF contains no extractable text — it looks like scanned images. "
                "OCR is not enabled; upload a text-based PDF or run it through OCR first."
            )
        return ParsedDocument(
            blocks=blocks,
            outline=outline,
            page_count=page_count,
            meta={
                "format": "pdf",
                "filename": filename,
                "body_font_size": round(body_size, 1),
            },
        )
    finally:
        document.close()


def parse_upload(path: Path, filename: str) -> ParsedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, filename=filename)
    if suffix in {".md", ".markdown"}:
        return parse_markdown(read_text_file(path), filename=filename)
    if suffix == ".txt":
        return parse_text(read_text_file(path), filename=filename)
    raise ParseError(f"Unsupported file type {suffix!r}. Upload PDF, Markdown or TXT.")
