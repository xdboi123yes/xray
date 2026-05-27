"""FastAPI main application module for chest X-ray tiered classification.

Registers modular routers for inference, thresholds, history database queries,
ablation matrices, and health checks, and configures lifespan hooks.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from web.backend.deps import get_system_state
from web.backend.middleware import (
    setup_cors,
    setup_error_handlers,
    setup_logging,
    setup_rate_limiting,
)
from web.backend.routes import ablation, health, history, inference, models, threshold
from web.backend.ws import inference_ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event manager starting backend assets."""
    state = get_system_state()
    # Safely load weights and database connections at bootstrap
    state.initialize()
    yield


app = FastAPI(
    title="Chest X-Ray Tiered Classification API",
    description=(
        "Production-grade statistical out-of-distribution tiered routing "
        "pneumothorax diagnostic backend server."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Initialize structured logging, rate limit, CORS, and standard exception handlers
setup_logging(app)
setup_rate_limiting(app)
setup_cors(app)
setup_error_handlers(app)


# Register modular routers
app.include_router(inference.router)
app.include_router(threshold.router)
app.include_router(ablation.router)
app.include_router(history.router)
app.include_router(health.router)
app.include_router(models.router)
app.include_router(inference_ws.router)
