"""Document indexing: extract -> chunk -> embed -> store.

Decoupled from transport (HTTP/file) and from the concrete embedder so it is easy
to test: callers pass the already-extracted text and an embedder.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Chunk, Document
from app.rag.chunking import chunk_text
from app.rag.providers import Embedder


def index_text(db: Session, document: Document, text: str, embedder: Embedder) -> int:
    """Chunk + embed ``text`` and persist chunks for ``document``. Returns the count."""
    settings = get_settings()
    chunks = chunk_text(
        text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    if not chunks:
        document.status = "failed"
        document.error = "No extractable text"
        db.commit()
        return 0

    embeddings = embedder.embed([c.content for c in chunks])
    for chunk, vector in zip(chunks, embeddings):
        db.add(
            Chunk(
                document_id=document.id,
                project_id=document.project_id,
                chunk_index=chunk.index,
                content=chunk.content,
                embedding=vector,
            )
        )
    document.status = "ready"
    document.error = None
    db.commit()
    return len(chunks)
