"""A portable embedding column type.

Uses pgvector's native ``vector`` type on PostgreSQL (so similarity search runs in
the database), and transparently falls back to JSON-encoded text on other engines
(e.g. SQLite under test) so the same models work everywhere.
"""

from __future__ import annotations

import json

from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator


class Embedding(TypeDecorator):
    impl = Text
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        return json.dumps(list(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return json.loads(value)
