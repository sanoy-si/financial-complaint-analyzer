import math

import pytest

from app.rag.providers import HashEmbedder, MockLLM, get_embedder, get_llm
from app.core.config import Settings


def test_hash_embedder_is_deterministic_and_unit_norm():
    emb = HashEmbedder(dim=64)
    a = emb.embed_one("the quick brown fox")
    b = emb.embed_one("the quick brown fox")
    assert a == b
    assert len(a) == 64
    assert math.isclose(sum(x * x for x in a) ** 0.5, 1.0, rel_tol=1e-9)


def test_hash_embedder_empty_text():
    emb = HashEmbedder(dim=16)
    assert emb.embed_one("") == [0.0] * 16


def test_hash_embedder_batch():
    emb = HashEmbedder(dim=32)
    out = emb.embed(["alpha", "beta"])
    assert len(out) == 2 and len(out[0]) == 32


def test_mock_llm_grounds_in_context():
    llm = MockLLM()
    prompt = "Context:\n[1] Refunds take 5 days.\n\nQuestion: How long do refunds take?\n\nAnswer:"
    out = llm.generate(prompt)
    assert "Refunds take 5 days" in out


def test_mock_llm_no_context():
    llm = MockLLM()
    out = llm.generate("Context:\n\n\nQuestion: anything?\n\nAnswer:")
    assert "don't have any indexed content" in out


def test_get_embedder_hash_default():
    emb = get_embedder(Settings(embedding_provider="hash", embedding_dim=128))
    assert emb.dim == 128


def test_get_llm_mock_default():
    assert isinstance(get_llm(Settings(llm_provider="mock")), MockLLM)


def test_get_llm_openai_requires_key():
    with pytest.raises(ValueError):
        get_llm(Settings(llm_provider="openai", llm_api_key=""))


def test_get_embedder_unknown_raises():
    with pytest.raises(ValueError):
        get_embedder(Settings(embedding_provider="nope"))


def test_database_url_normalized_to_psycopg():
    assert Settings(database_url="postgresql://u:p@h/db").database_url.startswith("postgresql+psycopg://")
    assert Settings(database_url="postgres://u:p@h/db").database_url.startswith("postgresql+psycopg://")
    # already-correct and sqlite URLs are left alone
    assert Settings(database_url="postgresql+psycopg://x").database_url == "postgresql+psycopg://x"
    assert Settings(database_url="sqlite://").database_url == "sqlite://"
