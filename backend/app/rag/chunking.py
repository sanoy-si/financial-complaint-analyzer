"""Text chunking for ingestion.

A small, dependency-free recursive splitter: it tries to break on the largest
natural boundary that keeps chunks under ``chunk_size`` (paragraphs, then lines,
then sentences, then words), with a configurable overlap so context isn't lost
across chunk edges.
"""

from __future__ import annotations

from dataclasses import dataclass

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size or not seps:
        return [text] if text else []
    sep, *rest = seps
    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = part if not buf else buf + sep + part
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(part) > size:
            out.extend(_split_recursive(part, size, rest))
            buf = ""
        else:
            buf = part
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, *, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Split ``text`` into overlapping chunks, preserving natural boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    raw = _split_recursive(text, chunk_size, _SEPARATORS)

    chunks: list[Chunk] = []
    carry = ""
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        body = (carry + " " + piece).strip() if carry else piece
        chunks.append(Chunk(index=len(chunks), content=body))
        carry = piece[-overlap:] if overlap and len(piece) > overlap else ""
    return chunks
