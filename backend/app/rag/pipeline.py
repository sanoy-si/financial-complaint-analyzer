"""The grounded RAG pipeline: retrieve -> assemble prompt -> generate.

The retriever is an interface so Phase 2 can swap the in-memory cosine search for
pgvector without touching the pipeline. Answers are grounded strictly in the
retrieved chunks, and the retrieved sources are returned alongside the answer so
the UI can cite them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from app.rag.providers import Embedder, LLM

_SYSTEM = (
    "You are a helpful assistant that answers questions using only the provided "
    "context. If the answer is not in the context, say you don't know. Be concise."
)


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    score: float
    document_id: str | None = None
    metadata: dict = field(default_factory=dict)


class Retriever(Protocol):
    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        ...


@dataclass
class InMemoryRetriever:
    """Cosine-similarity search over a fixed set of pre-embedded chunks.

    Used for tests and single-process demos. Production uses a pgvector-backed
    retriever scoped to a project.
    """

    chunks: list[RetrievedChunk]
    embeddings: list[list[float]]

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(
                content=c.content,
                score=_cosine(query_embedding, emb),
                document_id=c.document_id,
                metadata=c.metadata,
            )
            for c, emb in zip(self.chunks, self.embeddings)
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievedChunk]


class RagPipeline:
    def __init__(self, embedder: Embedder, llm: LLM, *, top_k: int = 5) -> None:
        self._embedder = embedder
        self._llm = llm
        self._top_k = top_k

    def answer(self, question: str, retriever: Retriever) -> Answer:
        q_emb = self._embedder.embed_one(question)
        sources = retriever.search(q_emb, self._top_k)
        prompt = self._build_prompt(question, sources)
        text = self._llm.generate(prompt, system=_SYSTEM)
        return Answer(text=text, sources=sources)

    @staticmethod
    def _build_prompt(question: str, sources: Sequence[RetrievedChunk]) -> str:
        blocks = []
        for i, s in enumerate(sources, start=1):
            blocks.append(f"[{i}] {s.content}")
        context = "\n\n".join(blocks)
        return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
