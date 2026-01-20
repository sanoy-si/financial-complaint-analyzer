def _project_id(auth_client) -> str:
    return auth_client.post("/api/v1/projects", json={"name": "Docs"}).json()["id"]


def test_url_ingest_records_failure_for_unsafe_url(auth_client):
    pid = _project_id(auth_client)
    # A loopback URL is rejected by the SSRF guard before any network call,
    # so the document is recorded as failed deterministically.
    resp = auth_client.post(
        f"/api/v1/projects/{pid}/documents/url",
        json={"url": "http://127.0.0.1/secrets"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error"]


def test_list_documents(auth_client):
    pid = _project_id(auth_client)
    auth_client.post(
        f"/api/v1/projects/{pid}/documents/url", json={"url": "http://10.0.0.1/"}
    )
    resp = auth_client.get(f"/api/v1/projects/{pid}/documents")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_documents_require_project_ownership(auth_client):
    resp = auth_client.post(
        "/api/v1/projects/not-mine/documents/url", json={"url": "http://10.0.0.1/"}
    )
    assert resp.status_code == 404
