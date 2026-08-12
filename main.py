"""
NeuroForge — ML Architecture Design AI Agent
Production FastAPI application.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from config import settings
from api.routes import router

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("neuroforge")


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("  NeuroForge APEX — ML Architecture Design AI Agent")
    logger.info("=" * 60)
    logger.info(f"  Provider : {settings.LLM_PROVIDER.value.upper()}")
    logger.info(f"  Model    : {settings.primary_model}")
    logger.info(f"  Code     : {settings.code_model}")
    logger.info(f"  Host     : {settings.APP_HOST}:{settings.APP_PORT}")
    logger.info("=" * 60)

    try:
        settings.validate_provider()
        logger.info("  API key  : ✓ valid")
    except ValueError as e:
        logger.error(f"  API key  : ✗ {e}")
        sys.exit(1)

    yield  # Application runs here

    logger.info("NeuroForge shutting down.")


# ─── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuroForge APEX",
    description=(
        "Multi-agent AI system for designing ML, Deep Learning, and Reinforcement Learning architectures. "
        "Powered by the APEX Protocol (Adaptive Progressive EXpert). "
        "Supports NVIDIA NIM and Mistral free-tier APIs."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ─── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Global error handler ────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "NeuroForge APEX",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "design": "POST /api/v1/design",
        "analyze": "POST /api/v1/analyze-dataset",
        "compare": "POST /api/v1/compare",
        "websocket": "ws://host/api/v1/design/stream",
    }


# ─── CLI entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1,
    )
