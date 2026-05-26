"""Threshold Configuration REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from application.dto.prediction_dto import ThresholdDTO
from web.backend.deps import SystemState, get_system_state

router = APIRouter(prefix="/api/v1", tags=["threshold"])


@router.get("/threshold", response_model=ThresholdDTO)
async def get_threshold_config(
    state: SystemState = Depends(get_system_state),
) -> Any:
    """Returns the active routing confidence bound and system calibration mode."""
    return {"value": state.threshold_value, "mode": state.threshold_mode}


@router.put("/threshold", response_model=ThresholdDTO)
async def update_threshold_config(
    payload: ThresholdDTO, state: SystemState = Depends(get_system_state)
) -> Any:
    """Dynamically updates the active routing threshold and calibration modes."""
    state.update_threshold(payload.value, payload.mode)
    return {"value": state.threshold_value, "mode": state.threshold_mode}
