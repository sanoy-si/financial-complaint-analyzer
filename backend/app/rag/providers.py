"""Pluggable embedding and generation providers.

The whole RAG stack talks to two small interfaces, :class:`Embedder` and
:class:`LLM`, resolved by ``get_embedder()`` / ``get_llm()`` from configuration.
This is what lets the same code run the always-free public demo (local embeddings
+ a deterministic mock generator) and a client's private deployment (their own
OpenAI/Groq/Gemini/Anthropic key) by changing environment variables only.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence

from app.core.config import Settings, get_settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        ...


class LLM(Protocol):
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        ...


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class HashEmbedder:
    """Deterministic, dependency-free embedder.

    Produces a stable unit vector from token hashes. It is not semantically
    strong, but it is fast, reproducible, and needs no model download — ideal for
    tests, CI, and local development without a GPU. Production uses
    :class:`SentenceTransformersEmbedder`.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class SentenceTransformersEmbedder:
    """Local sentence-transformers embedder (lazy-loaded).

    Imported lazily so the rest of the app (and the test suite) never needs torch
    installed unless this provider is actually selected.
    """

    def __init__(self, model_name: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# --------------------------------------------------------------------------- #
# LLMs
# --------------------------------------------------------------------------- #
class MockLLM:
    """Deterministic generator for the demo, tests, and offline development.

    It answers strictly from the retrieved context it is given, so behaviour is
    reproducible and never incurs an API cost or leaks a key.
    """

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        context = _extract_context(prompt)
        question = _extract_question(prompt)
        if not context.strip():
            return (
                "I don't have any indexed content that answers that yet. "
                "Add a document or URL to this project and try again."
            )
        snippet = context.strip().split("\n\n")[0][:400]
        return (
            f"Based on the provided sources, here is what is relevant to "
            f'"{question}":\n\n{snippet}'
        )


class OpenAICompatibleLLM:
    """Generation via any OpenAI-compatible Chat Completions endpoint.

    Works for OpenAI, Groq, and other compatible gateways by varying base_url and
    model. Imported lazily and only used when a key is configured.
    """

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        from openai import OpenAI  # lazy

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.0
        )
        return resp.choices[0].message.content or ""


_OPENAI_BASE_URLS = {
    "openai": None,
    "groq": "https://api.groq.com/openai/v1",
}


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()
    if provider == "hash":
        return HashEmbedder(dim=settings.embedding_dim)
    if provider == "sentence_transformers":
        return SentenceTransformersEmbedder(
            model_name=settings.embedding_model, dim=settings.embedding_dim
        )
    raise ValueError(f"unknown embedding provider: {settings.embedding_provider}")


def get_llm(settings: Settings | None = None) -> LLM:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()
    if provider == "mock":
        return MockLLM()
    if provider in _OPENAI_BASE_URLS:
        if not settings.llm_api_key:
            raise ValueError(f"{provider} provider requires LLM_API_KEY")
        return OpenAICompatibleLLM(
            api_key=settings.llm_api_key,
            model=settings.llm_model or _default_model(provider),
            base_url=_OPENAI_BASE_URLS[provider],
        )
    raise ValueError(f"unsupported llm provider: {settings.llm_provider}")


def _default_model(provider: str) -> str:
    return {"openai": "gpt-4o-mini", "groq": "llama-3.1-8b-instant"}.get(provider, "")


def _extract_context(prompt: str) -> str:
    if "Context:" in prompt and "Question:" in prompt:
        return prompt.split("Context:", 1)[1].split("Question:", 1)[0]
    return prompt


def _extract_question(prompt: str) -> str:
    if "Question:" in prompt:
        return prompt.split("Question:", 1)[1].strip().splitlines()[0]
    return prompt.strip().splitlines()[-1] if prompt.strip() else ""
