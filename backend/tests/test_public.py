from app.api.routers.public import get_rate_limiter
from app.core.config import get_settings
from app.core.security import generate_public_key, hash_password
from app.db.models import Document, Project, User
from app.rag.providers import get_embedder
from app.services.indexing import index_text
from app.services.rate_limit import InMemoryRateLimiter


def _seed_demo_project(db_session, name="Policy handbook", questions=("How long do refunds take?",)):
    user = User(
        email=get_settings().demo_user_email,
        password_hash=hash_password("demo-password"),
    )
    db_session.add(user)
    db_session.flush()
    project = Project(
        user_id=user.id,
        name=name,
        public_key=generate_public_key(),
        settings={"sample_questions": list(questions)},
    )
    db_session.add(project)
    db_session.flush()
    doc = Document(project_id=project.id, source_type="url", source_ref="seed")
    db_session.add(doc)
    db_session.flush()
    index_text(db_session, doc, "Refunds are processed within five business days.", get_embedder())
    db_session.commit()
    return project


def _seed(auth_client, db_session):
    created = auth_client.post("/api/v1/projects", json={"name": "Widget KB"}).json()
    project = db_session.get(Project, created["id"])
    doc = Document(project_id=project.id, source_type="url", source_ref="seed")
    db_session.add(doc)
    db_session.flush()
    index_text(
        db_session,
        doc,
        "Refunds are processed within five business days.",
        get_embedder(),
    )
    return created["public_key"]


def test_list_demos_empty_without_seed(client):
    resp = client.get("/api/v1/public/demos")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_demos_returns_seeded_projects(client, db_session):
    project = _seed_demo_project(db_session)
    resp = client.get("/api/v1/public/demos")
    assert resp.status_code == 200
    demos = resp.json()
    assert len(demos) == 1
    assert demos[0]["name"] == "Policy handbook"
    assert demos[0]["public_key"] == project.public_key
    assert demos[0]["sample_questions"] == ["How long do refunds take?"]


def test_list_demos_public_key_answers_chat(client, db_session):
    """The public key returned by /demos works against /public/chat, no login."""
    _seed_demo_project(db_session)
    pk = client.get("/api/v1/public/demos").json()[0]["public_key"]
    resp = client.post(
        "/api/v1/public/chat",
        json={"public_key": pk, "question": "How long do refunds take?"},
    )
    assert resp.status_code == 200
    assert "five business days" in resp.json()["answer"]


def test_public_chat_with_valid_key(auth_client, db_session):
    pk = _seed(auth_client, db_session)
    resp = auth_client.post(
        "/api/v1/public/chat",
        json={"public_key": pk, "question": "refunds processed business days"},
    )
    assert resp.status_code == 200
    assert "Refunds are processed within five business days" in resp.json()["answer"]


def test_public_chat_invalid_key(auth_client):
    resp = auth_client.post(
        "/api/v1/public/chat", json={"public_key": "pk_nope", "question": "hi"}
    )
    assert resp.status_code == 401


def test_public_chat_rate_limited(auth_client, db_session):
    pk = _seed(auth_client, db_session)
    # Swap in a single shared limiter that allows just one request.
    limiter = InMemoryRateLimiter(limit=1, window_seconds=60, clock=lambda: 0.0)
    auth_client.app.dependency_overrides[get_rate_limiter] = lambda: limiter
    body = {"public_key": pk, "question": "refunds"}
    assert auth_client.post("/api/v1/public/chat", json=body).status_code == 200
    assert auth_client.post("/api/v1/public/chat", json=body).status_code == 429


def test_public_chat_domain_allowlist(auth_client, db_session):
    pk = _seed(auth_client, db_session)
    project = db_session.scalars(
        __import__("sqlalchemy").select(Project).where(Project.public_key == pk)
    ).one()
    project.settings = {"allowed_domains": ["trusted.com"]}
    db_session.commit()

    blocked = auth_client.post(
        "/api/v1/public/chat",
        json={"public_key": pk, "question": "refunds"},
        headers={"origin": "https://evil.com"},
    )
    assert blocked.status_code == 403

    ok = auth_client.post(
        "/api/v1/public/chat",
        json={"public_key": pk, "question": "refunds"},
        headers={"origin": "https://trusted.com"},
    )
    assert ok.status_code == 200
