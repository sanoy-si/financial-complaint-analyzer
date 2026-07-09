#!/bin/sh
set -e

echo "=== Running alembic upgrade ==="
alembic upgrade head || echo "=== alembic upgrade failed (will attempt create_all fallback) ==="

echo "=== Ensuring tables exist ==="
python - <<'EOF'
import sys
from sqlalchemy import create_engine, inspect, text
from app.core.config import get_settings
from app.db.base import Base
from app.db import models  # noqa - register all ORM models

settings = get_settings()
engine = create_engine(settings.database_url)

with engine.connect() as conn:
    if engine.dialect.name == "postgresql":
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    if "users" not in inspect(conn).get_table_names():
        print("Tables missing — running create_all", file=sys.stderr)
        Base.metadata.create_all(engine)
        print("create_all done", file=sys.stderr)
    else:
        print("Tables already present", file=sys.stderr)

engine.dispose()
EOF

echo "=== Starting uvicorn ==="
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
