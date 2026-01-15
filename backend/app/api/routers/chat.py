"""Authenticated chat endpoint for the project owner's playground."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.models import Conversation, Message, Project, User
from app.db.session import get_db
from app.rag.pipeline import RagPipeline
from app.rag.providers import get_embedder, get_llm
from app.rag.retriever_db import DocumentRetriever
from app.schemas.chat import ChatRequest, ChatResponse, Source

router = APIRouter(prefix="/projects/{project_id}/chat", tags=["chat"])


def answer_for_project(
    db: Session, project: Project, question: str, session_id: str
) -> ChatResponse:
    settings = get_settings()
    pipeline = RagPipeline(get_embedder(), get_llm(), top_k=settings.retrieval_top_k)
    result = pipeline.answer(question, DocumentRetriever(db, project.id))

    conversation = Conversation(project_id=project.id, session_id=session_id)
    db.add(conversation)
    db.flush()
    db.add(Message(conversation_id=conversation.id, role="user", content=question))
    db.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result.text,
            citations=[s.document_id for s in result.sources],
        )
    )
    db.commit()

    return ChatResponse(
        answer=result.text,
        sources=[
            Source(content=s.content, score=s.score, document_id=s.document_id)
            for s in result.sources
        ],
        session_id=session_id,
    )


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
