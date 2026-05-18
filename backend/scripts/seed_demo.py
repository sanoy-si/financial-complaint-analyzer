"""Seed a demo account with sample datasets for the one-click demos.

Creates a demo user and three projects (a finance/compliance flagship plus two
generic examples) and indexes a small sample corpus into each. Safe to re-run:
it upserts the demo user and skips projects that already exist.

Usage:
    cd backend && DATABASE_URL=sqlite:///demo.db python -m scripts.seed_demo
"""

from __future__ import annotations

from sqlalchemy import select

from app.db.base import Base
from app.db.models import Document, Project, User
from app.db.session import get_engine, _session_factory
from app.core.security import generate_public_key, hash_password
from app.rag.providers import get_embedder
from app.services.indexing import index_text

DEMO_EMAIL = "demo@grounded.app"

SAMPLES: dict[str, list[str]] = {
    "Finance complaints (CFPB)": [
        "Consumer reports an unauthorized charge on a credit card. The issuer "
        "reversed the charge after 10 business days and opened a fraud claim.",
        "Customer disputes a late fee applied during a payment-processing outage. "
        "Policy waives late fees when the outage is on the institution's side.",
        "Mortgage escrow shortfall led to an unexpected payment increase; the "
        "servicer offered a 12-month spread to repay the shortfall.",
    ],
    "Policy handbook": [
        "Refunds are issued to the original payment method within five business days.",
        "Employees may carry over up to ten unused vacation days into the next year.",
        "Expense reports must be submitted within 30 days of the expense date.",
    ],
    "Product docs": [
        "To rotate an API key, open Settings → API and click Rotate. The old key "
        "stops working immediately.",
        "Rate limits are 60 requests per minute on the free tier and 1000 on Pro.",
        "Webhooks retry with exponential backoff for up to 24 hours on failure.",
    ],
}


def seed() -> None:
    Base.metadata.create_all(get_engine())
    db = _session_factory()()
    try:
        user = db.scalar(select(User).where(User.email == DEMO_EMAIL))
        if user is None:
            user = User(email=DEMO_EMAIL, password_hash=hash_password("demo-password"))
            db.add(user)
            db.commit()
            db.refresh(user)

        embedder = get_embedder()
        for name, docs in SAMPLES.items():
            existing = db.scalar(
                select(Project).where(Project.user_id == user.id, Project.name == name)
            )
            if existing is not None:
                continue
            project = Project(
                user_id=user.id, name=name, public_key=generate_public_key(), settings={}
            )
            db.add(project)
            db.flush()
            document = Document(project_id=project.id, source_type="url", source_ref="sample")
            db.add(document)
            db.flush()
            index_text(db, document, "\n\n".join(docs), embedder)
            print(f"seeded project {name!r} ({project.public_key})")
        print(f"\nDemo user: {DEMO_EMAIL} / demo-password")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
