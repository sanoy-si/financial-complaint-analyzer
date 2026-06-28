FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; the default providers (local hash embeddings + mock
# generation) need no heavy ML libraries, so the image stays slim.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

# Shell form so $PORT (injected by Render and other PaaS) is honoured; falls back
# to 8000 for local runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
