"""Shared pytest fixtures.

Each test gets an isolated in-memory SQLite database so the suite runs without a
real Postgres (the portable Embedding column makes the models SQLite-compatible).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_current_user  # noqa: F401  (imported for symmetry)
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(db_session) -> TestClient:
    app = create_app()

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


@pytest.fixture
def auth_client(client) -> TestClient:
    """A client already signed up and carrying a bearer token."""
    resp = client.post(
        "/api/v1/auth/signup",
        json={"email": "owner@example.com", "password": "supersecret123"},
    )
    token = resp.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client
