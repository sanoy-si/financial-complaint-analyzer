# Grounded — chat with your documents

Grounded turns any PDF or website into a **grounded** chatbot: it ingests your
content, answers questions using only your sources (with citations), and gives you
an embeddable `<script>` widget to drop on your own site.

The engine is fully generic ("chat with my docs"), and its flagship demo targets
**finance & compliance** — making sense of customer complaints, policies, and
filings — building on this project's original CFPB-complaint analysis work
(see `notebooks/`).

> Evolved from a single-purpose RAG notebook into a small multi-tenant product.
> Earlier history (the CFPB EDA + prototype) is preserved under `notebooks/` and
> `src/`.

## Highlights

- **Pluggable, no lock-in.** A tiny `get_embedder()` / `get_llm()` factory selects
  providers by env var. Embeddings default to a **local** model (documents never
  leave your server); generation defaults to a deterministic mock for a free demo,
  and swaps to OpenAI/Groq/etc. with one variable.
- **Multi-tenant.** Users → projects (chatbots) → documents → pgvector-backed
  chunks, isolated per project.
- **Embeddable widget.** One `<script>` tag renders a Shadow-DOM chat bubble that
  calls a public, key-scoped API with per-project domain allowlisting and rate
  limiting.
- **Standard practice.** JWT auth, Pydantic validation, SSRF-guarded URL
  ingestion, Alembic migrations, Docker Compose, GitHub Actions CI, and a real
  test suite.

## Architecture

```
Next.js (Vercel)  ─┐
embed widget       ─┼──▶  FastAPI (Render)  ──▶  Postgres + pgvector
                   │            ├─ auth (JWT) / projects
                   │            ├─ ingestion (PDF/URL → chunk → embed)
                   │            ├─ rag (pluggable providers, retrieval, citations)
                   │            └─ public widget API (key + rate limit)
```

| Area | Stack |
|------|-------|
| Frontend | Next.js (App Router), TypeScript |
| Backend | FastAPI, SQLAlchemy, Pydantic |
| Vector store | Postgres + pgvector (JSON fallback for tests) |
| RAG | retrieval pipeline, local/OpenAI/Groq providers |
| Infra | Docker Compose, Render (API+DB), Vercel (web), GitHub Actions |

## Repo layout

```
backend/        FastAPI app, RAG engine, migrations, tests
apps/web/       Next.js dashboard + landing
apps/widget/    embeddable widget demo + docs (served from the backend)
infra/          Render blueprint
notebooks/, src/  original CFPB analysis (provenance)
docker-compose.yml
```

## Quickstart (local, one command)

```bash
cp backend/.env.example backend/.env   # defaults work offline
docker compose up --build
```

- API → http://localhost:8000 (docs at `/docs`, health at `/health`)
- Web → http://localhost:3000

Defaults run **fully offline**: local hash embeddings + a deterministic mock
generator, so there's no API key or cost. Point `LLM_PROVIDER` / `LLM_API_KEY` at a
real model for production-quality answers.

## Backend dev / tests

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
DATABASE_URL=sqlite:// pytest -q
```

## Deploy

- **Backend + DB:** Render blueprint at `infra/render.yaml` (Docker + managed
  Postgres with pgvector; runs `alembic upgrade head` on deploy).
- **Frontend:** deploy `apps/web` to Vercel; set `NEXT_PUBLIC_API_BASE` to the API URL.

## Security notes

- Secrets only via env (`.env` is git-ignored); `.env.example` documents them.
- URL ingestion blocks private/loopback/metadata addresses (SSRF).
- The widget public key is non-secret; restrict by domain allowlist + rate limit.
