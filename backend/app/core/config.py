"""Application settings, loaded from the environment.

All configuration is environment-driven so the same image runs locally, in the
public demo, and in a client's private deployment. No secrets are hard-coded; see
``.env.example`` for the full list of variables.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- product ---
    product_name: str = Field(default="Grounded")
    environment: str = Field(default="development")

    # --- api ---
    api_v1_prefix: str = "/api/v1"
    # The embeddable widget is served from arbitrary third-party domains, so the
    # default is permissive; per-project domain allowlisting provides the real
    # restriction. Lock this down to known origins in a private deployment.
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- database ---
    database_url: str = Field(
        default="postgresql+psycopg://grounded:grounded@localhost:5432/grounded"
    )

    @field_validator("database_url")
    @classmethod
    def _use_psycopg_driver(cls, value: str) -> str:
        """Managed hosts (Render/Heroku) hand out ``postgres(ql)://`` URLs that
        SQLAlchemy maps to psycopg2; we ship psycopg v3, so normalise the scheme."""
        if value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value[len("postgresql://"):]
        if value.startswith("postgres://"):
            return "postgresql+psycopg://" + value[len("postgres://"):]
        return value

    # --- auth ---
    jwt_secret: str = Field(default="dev-only-change-me")
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 60
    refresh_token_ttl_days: int = 14

    # --- rag providers ---
    # Embeddings default to a local model so documents never leave the server.
    embedding_provider: str = Field(default="hash")  # hash | sentence_transformers
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)
    # Generation is pluggable; the public demo defaults to a deterministic mock
    # so it costs nothing and never exposes a paid key.
    llm_provider: str = Field(default="mock")  # mock | openai | groq | gemini | anthropic | ollama
    llm_model: str = Field(default="")
    llm_api_key: str = Field(default="")

    # --- ingestion ---
    max_upload_mb: int = 20
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5

    # --- public widget ---
    widget_rate_limit_per_minute: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
