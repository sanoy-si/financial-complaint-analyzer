"""Shared chat service used by both the authenticated playground and the public
widget endpoint, so grounding/persistence behaves identically in both."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Conversation, Message, Project
from app.rag.pipeline import RagPipeline
from app.rag.providers import get_embedder, get_llm
from app.rag.retriever_db import DocumentRetriever
from app.schemas.chat import ChatResponse, Source


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
