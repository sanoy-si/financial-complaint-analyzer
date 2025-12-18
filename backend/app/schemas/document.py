"""Document request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UrlIngestRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class DocumentResponse(BaseModel):
    id: str
    source_type: str
    source_ref: str
    status: str
    error: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
