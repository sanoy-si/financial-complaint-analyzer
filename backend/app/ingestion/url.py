"""URL fetching and HTML-to-text extraction, with SSRF protection.

Because users supply the URL, we must not let the server be tricked into fetching
internal/metadata endpoints. Before fetching we resolve the host and reject any
address that is private, loopback, link-local, or otherwise non-global.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from app.errors import IngestionError


def assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IngestionError("Only http(s) URLs are supported")
    host = parsed.hostname
    if not host:
        raise IngestionError("URL has no host")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:  # pragma: no cover - network dependent
        raise IngestionError(f"Could not resolve host: {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_loopback or ip.is_private or ip.is_link_local:
            raise IngestionError("Refusing to fetch a non-public address")


def extract_html_text(html: str) -> str:
    """Strip scripts/styles and collapse an HTML document to readable text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def fetch_url_text(url: str, *, timeout: float = 15.0, max_bytes: int = 5_000_000) -> str:
    """Validate, fetch, and extract readable text from ``url``."""
    import httpx

    assert_public_url(url)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, headers={"User-Agent": "GroundedBot/1.0"})
        resp.raise_for_status()
        content = resp.text[: max_bytes]
    return extract_html_text(content)
