"""Project request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ProjectResponse(BaseModel):
    id: str
    name: str
    public_key: str
    settings: dict
    created_at: datetime

    model_config = {"from_attributes": True}
