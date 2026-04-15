def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["product"]


def test_widget_js_served(client):
    resp = client.get("/widget.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    assert "data-project-key" in resp.text
