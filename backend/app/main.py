"""FastAPI application factory for the Grounded backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routers import auth, chat, documents, health, projects, public
from app.core.config import get_settings

_WIDGET_PATH = Path(__file__).parent / "static" / "widget.js"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=f"{settings.product_name} API", version="0.1.0")

    # Bearer tokens (not cookies) carry auth, so credentials mode isn't needed;
    # that also lets us use a wildcard origin for the cross-site widget.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials="*" not in settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(auth.router, prefix=settings.api_v1_prefix)
    app.include_router(projects.router, prefix=settings.api_v1_prefix)
    app.include_router(documents.router, prefix=settings.api_v1_prefix)
    app.include_router(chat.router, prefix=settings.api_v1_prefix)
    app.include_router(public.router, prefix=settings.api_v1_prefix)

    @app.get("/widget.js", include_in_schema=False)
    def widget_js() -> FileResponse:
        return FileResponse(_WIDGET_PATH, media_type="application/javascript")

    return app


app = create_app()
