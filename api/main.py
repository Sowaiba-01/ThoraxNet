"""
ChestAI FastAPI application entry point.

Features:
  - Async lifespan: model loads once at startup, GradCAM hooks are cleaned up on shutdown
  - CORS: configured for local dev + production Vercel frontend
  - Rate limiting: 10 predictions/minute per IP (slowapi)
  - Auto-generated OpenAPI docs at /docs
  - Request ID middleware for distributed tracing
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.inference import pipeline
from api.routes import health, predict


# ------------------------------------------------------------------
# Rate limiter
# ------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


# ------------------------------------------------------------------
# Lifespan: startup / shutdown
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[ChestAI] Loading model...")
    await pipeline.load()
    print("[ChestAI] Ready.")
    yield
    # Cleanup GradCAM hooks on shutdown.
    if pipeline.gradcam:
        pipeline.gradcam.remove_hooks()
    print("[ChestAI] Shutdown complete.")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
app = FastAPI(
    title="ChestAI",
    description=(
        "Uncertainty-aware multi-label chest X-ray diagnostic API. "
        "Analyzes 14 pathology classes with MC Dropout uncertainty estimation, "
        "GradCAM explainability, and auto-generated radiology reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",          # your Vercel deployments
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Request ID middleware (for tracing in logs)
# ------------------------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ------------------------------------------------------------------
# Global exception handler
# ------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "type": type(exc).__name__},
    )


# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------
app.include_router(health.router)
app.include_router(predict.router)
