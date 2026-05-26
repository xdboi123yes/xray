"""System diagnostic health endpoints."""

from __future__ import annotations

import time
from typing import Any

import torch
from fastapi import APIRouter, Depends

from application.dto.prediction_dto import HealthDTO
from web.backend.deps import SystemState, get_system_state

router = APIRouter(prefix="/api/v1", tags=["health"])

# System boot monotonic time reference
START_UPTIME = time.monotonic()


@router.get("/health", response_model=HealthDTO)
async def health_check(state: SystemState = Depends(get_system_state)) -> Any:
    """Returns uptime statistics, GPU registrations, and active models status."""
    loaded_models = []
    if state.tier1_model is not None:
        loaded_models.append("tier1_mobilenet")
    if state.tier2_model is not None:
        loaded_models.append("tier2_efficientnet")

    gpu_active = torch.backends.mps.is_available() or torch.cuda.is_available()

    return {
        "status": "ok" if len(loaded_models) == 2 else "degraded",
        "gpu": gpu_active,
        "models_loaded": loaded_models,
        "version": "2.0.0",
        "uptime_s": time.monotonic() - START_UPTIME,
    }
