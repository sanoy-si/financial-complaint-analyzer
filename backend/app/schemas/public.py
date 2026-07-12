"""Schemas for the public, key-authenticated widget API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PublicChatRequest(BaseModel):
    public_key: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class PublicDemoResponse(BaseModel):
    """A seeded, no-login demo dataset exposed by its non-secret public key."""

    name: str
    public_key: str
    sample_questions: list[str] = Field(default_factory=list)
