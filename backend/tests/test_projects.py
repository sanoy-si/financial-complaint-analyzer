def test_create_project_issues_public_key(auth_client):
    resp = auth_client.post("/api/v1/projects", json={"name": "Docs Bot"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Docs Bot"
    assert body["public_key"].startswith("pk_")


def test_list_projects_scoped_to_user(auth_client):
    auth_client.post("/api/v1/projects", json={"name": "A"})
    auth_client.post("/api/v1/projects", json={"name": "B"})
    resp = auth_client.get("/api/v1/projects")
    assert resp.status_code == 200
    assert {p["name"] for p in resp.json()} == {"A", "B"}


def test_get_project_by_id(auth_client):
    created = auth_client.post("/api/v1/projects", json={"name": "C"}).json()
    resp = auth_client.get(f"/api/v1/projects/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == created["id"]


def test_get_unknown_project_404(auth_client):
    assert auth_client.get("/api/v1/projects/does-not-exist").status_code == 404


def test_projects_require_auth(client):
    assert client.get("/api/v1/projects").status_code == 401
