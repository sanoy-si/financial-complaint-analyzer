"""Public widget API.

Authenticated by a project's non-secret public key (``pk_…``) rather than a user
JWT, so it can be called directly from a third-party page by the embed script.
Protected by an optional per-project domain allowlist and a per-key rate limit.
"""

from __future__ import annotations

import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Project, User
from app.db.session import get_db
from app.schemas.chat import ChatResponse
from app.schemas.public import PublicChatRequest, PublicDemoResponse
from app.services.chat import answer_for_project
from app.services.rate_limit import InMemoryRateLimiter

router = APIRouter(prefix="/public", tags=["public"])

_settings = get_settings()
_rate_limiter = InMemoryRateLimiter(_settings.widget_rate_limit_per_minute, 60.0)


def get_rate_limiter() -> InMemoryRateLimiter:
    return _rate_limiter


@router.get("/demos", response_model=list[PublicDemoResponse])
def list_demos(db: Session = Depends(get_db)) -> list[PublicDemoResponse]:
    """List the seeded demo datasets so a logged-out visitor can try the product.

    Only the demo account's projects are exposed, and only their non-secret
    public keys — the same keys the embed widget already ships to the browser.
    """
    user = db.scalar(select(User).where(User.email == _settings.demo_user_email))
    if user is None:
        return []
    projects = db.scalars(
        select(Project).where(Project.user_id == user.id).order_by(Project.created_at)
    ).all()
    return [
        PublicDemoResponse(
            name=p.name,
            public_key=p.public_key,
            sample_questions=list((p.settings or {}).get("sample_questions", [])),
        )
        for p in projects
    ]


def _domain_allowed(project: Project, origin: str | None) -> bool:
    allowed = project.settings.get("allowed_domains") or []
    if not allowed:
        return True  # not restricted
    if not origin:
        return False
    host = urlparse(origin).hostname or origin
    return host in allowed


@router.post("/chat", response_model=ChatResponse)
def public_chat(
    payload: PublicChatRequest,
    db: Session = Depends(get_db),
    rate_limiter: InMemoryRateLimiter = Depends(get_rate_limiter),
    origin: str | None = Header(default=None),
) -> ChatResponse:
    project = db.scalar(select(Project).where(Project.public_key == payload.public_key))
    if project is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid public key")
    if not _domain_allowed(project, origin):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Origin not allowed")
    if not rate_limiter.allow(payload.public_key):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded")

    session_id = payload.session_id or str(uuid.uuid4())
    return answer_for_project(db, project, payload.question, session_id)
