"""Document ingestion endpoints: upload a PDF or submit a URL."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.models import Document, Project, User
from app.db.session import get_db
from app.errors import IngestionError
from app.ingestion.pdf import extract_pdf_text
from app.ingestion.url import fetch_url_text
from app.rag.providers import get_embedder
from app.schemas.document import DocumentResponse, UrlIngestRequest
from app.services.indexing import index_text

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])


def _owned_project(project_id: str, user: User, db: Session) -> Project:
    project = db.get(Project, project_id)
    if project is None or project.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Project not found")
    return project


@router.post("/pdf", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf(
    project_id: str,
    file: UploadFile,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Document:
    project = _owned_project(project_id, user, db)
    settings = get_settings()
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")

    document = Document(
        project_id=project.id, source_type="pdf", source_ref=file.filename or "upload.pdf"
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        text = extract_pdf_text(data)
        index_text(db, document, text, get_embedder())
    except IngestionError as exc:
        document.status, document.error = "failed", str(exc)
        db.commit()
    return document


@router.post("/url", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def ingest_url(
    project_id: str,
    payload: UrlIngestRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Document:
    project = _owned_project(project_id, user, db)
    document = Document(project_id=project.id, source_type="url", source_ref=payload.url)
    db.add(document)
    db.commit()
    db.refresh(document)

    try:
        text = fetch_url_text(payload.url)
        index_text(db, document, text, get_embedder())
    except IngestionError as exc:
        document.status, document.error = "failed", str(exc)
        db.commit()
    return document


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    project_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Document]:
    project = _owned_project(project_id, user, db)
    return list(db.scalars(select(Document).where(Document.project_id == project.id)))
