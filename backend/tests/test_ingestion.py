import pytest

from app.errors import IngestionError
from app.ingestion.url import assert_public_url, extract_html_text


def test_extract_html_strips_scripts_and_chrome():
    html = """
    <html><head><style>.x{}</style></head>
    <body><nav>menu</nav><script>evil()</script>
    <p>Refunds take five days.</p><footer>©</footer></body></html>
    """
    text = extract_html_text(html)
    assert "Refunds take five days." in text
    assert "evil()" not in text
    assert "menu" not in text


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "ftp://example.com/file",
    ],
)
def test_assert_public_url_blocks_unsafe(url):
    with pytest.raises(IngestionError):
        assert_public_url(url)


def test_assert_public_url_allows_public_host(monkeypatch):
    # Resolve to a public address without hitting real DNS.
    monkeypatch.setattr(
        "app.ingestion.url.socket.getaddrinfo",
        lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    assert_public_url("https://example.com/page")  # does not raise
