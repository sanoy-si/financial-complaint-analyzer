"""PDF text extraction."""

from __future__ import annotations

import io


def extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF byte stream (pypdf, lazily imported)."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts).strip()
