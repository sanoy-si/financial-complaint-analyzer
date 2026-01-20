from sqlalchemy import select

from app.db.models import Chunk, Document, Project, User
from app.core.security import hash_password, generate_public_key
from app.rag.providers import HashEmbedder
from app.services.indexing import index_text


def _project(db):
    user = User(email="u@example.com", password_hash=hash_password("password123"))
    db.add(user)
    db.flush()
    project = Project(user_id=user.id, name="P", public_key=generate_public_key(), settings={})
    db.add(project)
    db.flush()
    return project


def test_index_text_creates_chunks(db_session):
    project = _project(db_session)
    doc = Document(project_id=project.id, source_type="url", source_ref="x")
    db_session.add(doc)
    db_session.flush()

    text = "\n\n".join(f"Sentence {i} about refunds and shipping." for i in range(15))
    n = index_text(db_session, doc, text, HashEmbedder(dim=64))

    assert n > 0
    assert doc.status == "ready"
    stored = db_session.scalars(select(Chunk).where(Chunk.project_id == project.id)).all()
    assert len(stored) == n
    assert len(stored[0].embedding) == 64  # embedding round-trips through the column


def test_index_empty_text_marks_failed(db_session):
    project = _project(db_session)
    doc = Document(project_id=project.id, source_type="url", source_ref="x")
    db_session.add(doc)
    db_session.flush()

    n = index_text(db_session, doc, "   ", HashEmbedder(dim=32))
    assert n == 0
    assert doc.status == "failed"
