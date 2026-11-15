"""Token counting used by every chunker.

tiktoken needs to fetch its BPE file the first time; if that fails (offline
install) we fall back to a sane approximation instead of breaking ingestion.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+|[^\w\s]")

_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "eg",
    "ie",
    "fig",
    "figs",
    "no",
    "nos",
    "inc",
    "ltd",
    "co",
    "corp",
    "st",
    "approx",
    "dept",
    "est",
    "al",
    "cf",
    "pp",
    "vol",
    "sec",
    "art",
    "u.s",
    "u.k",
}
_INITIALS_RE = re.compile(r"(?:\b[A-Z]\.)+$")


def _is_abbrev(candidate: str) -> bool:
    """True when the period before the split belongs to an abbreviation."""
    stripped = candidate.rstrip().rstrip("\"')]").rstrip()
    if _INITIALS_RE.search(stripped):
        return True
    match = re.search(r"([A-Za-z][A-Za-z.]*)$", stripped)
    if not match:
        return False
    return match.group(1).lower() in _ABBREVIATIONS


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001
        log.warning("tiktoken unavailable (%s); using approximate token counts", exc)
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoder = _encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:  # noqa: BLE001
            pass
    # ~4 characters per token, at least one per word-ish unit.
    return max(1, len(text) // 4)


def truncate_tokens(text: str, limit: int) -> str:
    """Keep a prefix of at most ``limit`` tokens."""
    if limit <= 0:
        return ""
    encoder = _encoder()
    if encoder is not None:
        try:
            tokens = encoder.encode(text, disallowed_special=())
            if len(tokens) <= limit:
                return text
            return encoder.decode(tokens[:limit])
        except Exception:  # noqa: BLE001
            pass
    return text[: limit * 4]


def tail_tokens(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    encoder = _encoder()
    if encoder is not None:
        try:
            tokens = encoder.encode(text, disallowed_special=())
            if len(tokens) <= limit:
                return text
            return encoder.decode(tokens[-limit:])
        except Exception:  # noqa: BLE001
            pass
    return text[-limit * 4 :]


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter: abbreviations stay put, code lines survive."""
    if not text.strip():
        return []
    parts: list[str] = []
    buffer: list[str] = []
    for line in text.split("\n"):
        stripped = line.rstrip()
        if not stripped.strip():
            if buffer:
                parts.append("\n".join(buffer).strip())
                buffer = []
            continue
        buffer.append(stripped)
    if buffer:
        parts.append("\n".join(buffer).strip())

    sentences: list[str] = []
    for block in parts:
        # Split after ., !, ? or a closing quote followed by whitespace + capital,
        # but not after common abbreviations or list numbers.
        block = re.sub(r"\s+", " ", block) if "\n" not in block else block
        start = 0
        for match in re.finditer(r"(?<=[.!?])[\"')\]]?\s+(?=[A-Z0-9(\"'\[])|\n{2,}", block):
            candidate = block[start : match.start()].strip()
            if candidate and not _is_abbrev(candidate):
                sentences.append(candidate)
                start = match.end()
        tail = block[start:].strip()
        if tail:
            sentences.append(tail)
    return [sentence for sentence in sentences if sentence.strip()]
