from app.db.models import Conversation, Document, Project
from app.rag.providers import get_embedder
from app.services.indexing import index_text


def _seed_project_with_docs(auth_client, db_session) -> str:
    pid = auth_client.post("/api/v1/projects", json={"name": "KB"}).json()["id"]
    project = db_session.get(Project, pid)
    doc = Document(project_id=project.id, source_type="url", source_ref="seed")
    db_session.add(doc)
    db_session.flush()
    text = (
        "Refunds are processed within five business days.\n\n"
        "Our support team is available Monday to Friday.\n\n"
        "Shipping is free on orders over fifty dollars."
    )
    index_text(db_session, doc, text, get_embedder())
    return pid


def test_chat_answers_from_indexed_docs(auth_client, db_session):
    pid = _seed_project_with_docs(auth_client, db_session)
    resp = auth_client.post(
        f"/api/v1/projects/{pid}/chat",
        json={"question": "refunds processed business days"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "Refunds are processed within five business days" in body["answer"]
    assert body["sources"]
    assert body["session_id"]


def test_chat_persists_conversation(auth_client, db_session):
    pid = _seed_project_with_docs(auth_client, db_session)
    auth_client.post(f"/api/v1/projects/{pid}/chat", json={"question": "shipping free"})
    convs = db_session.query(Conversation).filter_by(project_id=pid).all()
    assert len(convs) == 1
    assert len(convs[0].messages) == 2  # user + assistant


def test_chat_empty_project_says_no_content(auth_client, db_session):
    pid = auth_client.post("/api/v1/projects", json={"name": "Empty"}).json()["id"]
    resp = auth_client.post(f"/api/v1/projects/{pid}/chat", json={"question": "anything?"})
    assert resp.status_code == 200
    assert "don't have any indexed content" in resp.json()["answer"]


def test_chat_requires_ownership(auth_client):
    resp = auth_client.post("/api/v1/projects/nope/chat", json={"question": "hi"})
    assert resp.status_code == 404
