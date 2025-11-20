def test_signup_returns_token(client):
    resp = client.post(
        "/api/v1/auth/signup",
        json={"email": "a@example.com", "password": "password123"},
    )
    assert resp.status_code == 201
    assert resp.json()["access_token"]


def test_signup_duplicate_email_conflicts(client):
    body = {"email": "dup@example.com", "password": "password123"}
    assert client.post("/api/v1/auth/signup", json=body).status_code == 201
    assert client.post("/api/v1/auth/signup", json=body).status_code == 409


def test_signup_rejects_short_password(client):
    resp = client.post(
        "/api/v1/auth/signup", json={"email": "b@example.com", "password": "short"}
    )
    assert resp.status_code == 422


def test_login_success_and_failure(client):
    client.post(
        "/api/v1/auth/signup",
        json={"email": "c@example.com", "password": "password123"},
    )
    ok = client.post(
        "/api/v1/auth/login",
        json={"email": "c@example.com", "password": "password123"},
    )
    assert ok.status_code == 200 and ok.json()["access_token"]

    bad = client.post(
        "/api/v1/auth/login",
        json={"email": "c@example.com", "password": "wrong"},
    )
    assert bad.status_code == 401


def test_me_requires_auth(client):
    assert client.get("/api/v1/auth/me").status_code == 401


def test_me_returns_current_user(auth_client):
    resp = auth_client.get("/api/v1/auth/me")
    assert resp.status_code == 200
    assert resp.json()["email"] == "owner@example.com"
