"""Database-backed retriever, scoped to a single project (tenant isolation).

Embeddings are stored in the pgvector column. On PostgreSQL this can be pushed
into an ANN index with the ``<=>`` operator for scale; here we rank with cosine
similarity in the application so the exact same code path is correct on both
PostgreSQL and SQLite (used by the test suite).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk
from app.rag.pipeline import RetrievedChunk, _cosine


class DocumentRetriever:
    def __init__(self, db: Session, project_id: str) -> None:
        self._db = db
        self._project_id = project_id

    def search(self, query_embedding, top_k: int) -> list[RetrievedChunk]:
        rows = self._db.scalars(
            select(Chunk).where(Chunk.project_id == self._project_id)
        ).all()
        scored = [
            RetrievedChunk(
                content=row.content,
                score=_cosine(query_embedding, row.embedding),
                document_id=row.document_id,
                metadata={"chunk_index": row.chunk_index},
            )
            for row in rows
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]
