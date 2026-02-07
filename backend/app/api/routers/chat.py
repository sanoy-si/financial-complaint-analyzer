"""Authenticated chat endpoint for the project owner's playground."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import Project, User
from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import answer_for_project

router = APIRouter(prefix="/projects/{project_id}/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    project_id: str,
    payload: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    project = db.get(Project, project_id)
    if project is None or project.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    session_id = payload.session_id or str(uuid.uuid4())
    return answer_for_project(db, project, payload.question, session_id)
