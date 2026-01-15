"""Chat request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class Source(BaseModel):
    content: str
    score: float
    document_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str
