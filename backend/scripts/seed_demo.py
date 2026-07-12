"""Seed a demo account with sample datasets for the one-click demos.

Creates a demo user and three projects (a finance/compliance flagship plus two
generic examples) and indexes a small sample corpus into each. Safe to re-run:
it upserts the demo user and skips projects that already exist.

Usage:
    cd backend && DATABASE_URL=sqlite:///demo.db python -m scripts.seed_demo
"""

from __future__ import annotations

from typing import NamedTuple

from sqlalchemy import select

from app.db.base import Base
from app.db.models import Document, Project, User
from app.db.session import get_engine, _session_factory
from app.core.security import generate_public_key, hash_password
from app.rag.providers import get_embedder
from app.services.indexing import index_text

DEMO_EMAIL = "demo@grounded.app"


class Sample(NamedTuple):
    docs: list[str]
    questions: list[str]


# Each demo project pairs a small corpus with a few starter questions whose
# answers live verbatim in that corpus, so the no-login playground always has a
# grounded, citation-backed reply to show first-time visitors.
SAMPLES: dict[str, Sample] = {
    "Finance complaints (CFPB)": Sample(
        docs=[
            "Consumer reports an unauthorized charge on a credit card. The issuer "
            "reversed the charge after 10 business days and opened a fraud claim.",
            "Customer disputes a late fee applied during a payment-processing outage. "
            "Policy waives late fees when the outage is on the institution's side.",
            "Mortgage escrow shortfall led to an unexpected payment increase; the "
            "servicer offered a 12-month spread to repay the shortfall.",
            "A debt collector contacted the consumer after the validation window had "
            "closed. The bureau requires collectors to cease contact until the debt "
            "is verified in writing.",
            "Customer's ACH transfer was held for review for three business days. The "
            "bank releases held transfers once identity re-verification is complete.",
            "Overdraft fees were charged on a account that had opted out of overdraft "
            "coverage. The institution refunded the fees and re-applied the opt-out.",
        ],
        questions=[
            "How long did the issuer take to reverse the unauthorized charge?",
            "When are late fees waived during an outage?",
            "What did the servicer offer for the escrow shortfall?",
        ],
    ),
    "Policy handbook": Sample(
        docs=[
            "Refunds are issued to the original payment method within five business days.",
            "Employees may carry over up to ten unused vacation days into the next year.",
            "Expense reports must be submitted within 30 days of the expense date.",
            "Remote employees are reimbursed up to $500 per year for home-office equipment.",
            "Parental leave is 16 weeks of fully paid time off for all new parents.",
            "Probationary period for new hires is 90 days, after which benefits vest.",
        ],
        questions=[
            "How long do refunds take?",
            "How many vacation days can I carry over?",
            "What is the parental leave policy?",
        ],
    ),
    "Product docs": Sample(
        docs=[
            "To rotate an API key, open Settings → API and click Rotate. The old key "
            "stops working immediately.",
            "Rate limits are 60 requests per minute on the free tier and 1000 on Pro.",
            "Webhooks retry with exponential backoff for up to 24 hours on failure.",
            "SDKs are available for Python, JavaScript, and Go; all share the same "
            "authentication flow.",
            "Uploaded files are scanned and indexed within a few seconds of upload.",
            "Deleting a project permanently removes its documents and vector index.",
        ],
        questions=[
            "How do I rotate an API key?",
            "What are the rate limits on the free tier?",
            "How long do webhooks retry on failure?",
        ],
    ),
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
        for name, sample in SAMPLES.items():
            existing = db.scalar(
                select(Project).where(Project.user_id == user.id, Project.name == name)
            )
            if existing is not None:
                # Backfill starter questions on a project seeded before this field existed.
                if not (existing.settings or {}).get("sample_questions"):
                    existing.settings = {
                        **(existing.settings or {}),
                        "sample_questions": sample.questions,
                    }
                    db.commit()
                continue
            project = Project(
                user_id=user.id,
                name=name,
                public_key=generate_public_key(),
                settings={"sample_questions": sample.questions},
            )
            db.add(project)
            db.flush()
            document = Document(project_id=project.id, source_type="url", source_ref="sample")
            db.add(document)
            db.flush()
            index_text(db, document, "\n\n".join(sample.docs), embedder)
            print(f"seeded project {name!r} ({project.public_key})")
        print(f"\nDemo user: {DEMO_EMAIL} / demo-password")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
