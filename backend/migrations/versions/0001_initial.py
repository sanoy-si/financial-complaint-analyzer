"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2025-10-04

Baseline migration: enables pgvector and creates the full schema from the ORM
metadata. Subsequent changes should be generated with ``alembic revision
--autogenerate``.
"""

from __future__ import annotations

from alembic import op

from app.db.base import Base
from app.db import models  # noqa: F401 - register models on the metadata

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
