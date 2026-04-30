"""
FastAPI entrypoint per NeuralForge.

Avvio (da backend/):
    python -m uvicorn main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from db import init_db
from logging_config import setup_logging

__version__ = "0.1.0"

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Startup/shutdown hooks."""
    logger.info("NeuralForge v%s — startup", __version__)
    settings.ensure_directories()
    init_db()

    # Inizializza NVML una volta sola per tutta la vita dell'app.
    from core.memory import init_nvml, shutdown_nvml
    init_nvml()

    logger.info("Server pronto su http://%s:%d", settings.host, settings.port)
    yield

    # Shutdown ordinato
    shutdown_nvml()
    logger.info("NeuralForge — shutdown")


app = FastAPI(
    title="NeuralForge API",
    description="Local LLM fine-tuning platform",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    """Healthcheck base."""
    return {
        "status": "ok",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(
        {
            "name": "NeuralForge",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/health",
        }
    )


# Routers
from api import system

app.include_router(system.router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_config=None,
    )