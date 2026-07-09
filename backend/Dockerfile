FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; the default providers (local hash embeddings + mock
# generation) need no heavy ML libraries, so the image stays slim.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

# start.sh: runs alembic upgrade, falls back to create_all if tables are
# missing (handles SA 2.0 + psycopg3 transaction edge cases), then starts
# uvicorn with the PaaS-injected $PORT.
CMD ["sh", "start.sh"]
