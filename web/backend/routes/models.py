"""Loaded model analytics and loading endpoints."""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from application.dto.prediction_dto import ModelLoadDTO
from web.backend.deps import SystemState, get_system_state

router = APIRouter(prefix="/api/v1", tags=["models"])


def calculate_model_checksum(model: Any) -> str:
    """Compute a sha256 checksum hash of the model parameters."""
    if model is None:
        return "N/A"
    try:
        hasher = hashlib.sha256()
        for param in model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()[:8]
    except Exception:
        return "unknown"


@router.get("/models")
async def get_loaded_models_details(
    state: SystemState = Depends(get_system_state),
) -> Any:
    """Returns metadata description of currently loaded model backbones."""
    t1_name = state.tier1_model.__class__.__name__ if state.tier1_model else "None"
    t2_name = state.tier2_model.__class__.__name__ if state.tier2_model else "None"

    return {
        "tier1": t1_name,
        "tier1_checksum": calculate_model_checksum(state.tier1_model),
        "tier2": t2_name,
        "tier2_checksum": calculate_model_checksum(state.tier2_model),
        "device": str(state.device),
        "conformal_calibration_status": "loaded",
    }


@router.post("/models/load")
async def load_model_from_registry(
    payload: ModelLoadDTO,
    state: SystemState = Depends(get_system_state),
) -> Any:
    """Loads a specific model configuration and weight file dynamically."""
    weights_filename = payload.name
    if not weights_filename.endswith(".pth"):
        # Map common names to actual model file basenames
        if "mobilenet" in weights_filename.lower():
            weights_filename = "best_tier1_mobilenet.pth"
        elif "efficientnet" in weights_filename.lower():
            weights_filename = "best_tier2_efficientnet.pth"
        else:
            weights_filename = f"best_{weights_filename}.pth"

    weights_path = f"outputs/models/{weights_filename}"

    # Determine backbone registry key and tier
    backbone_name = payload.name.lower()
    if "mobilenet" in backbone_name:
        backbone_key = "mobilenet_v2"
        tier = 1
    elif "efficientnet" in backbone_name:
        backbone_key = "efficientnet_b4"
        tier = 2
    elif "ark" in backbone_name:
        backbone_key = "ark_plus"
        tier = 2
    else:
        backbone_key = "mobilenet_v2" if "tier1" in backbone_name else "efficientnet_b4"
        tier = 1 if "tier1" in backbone_name else 2

    try:
        state.reload_tier(tier=tier, backbone_name=backbone_key, weights_path=weights_path)
        return {
            "status": "loaded",
            "model_name": payload.name,
            "backbone": backbone_key,
            "tier": tier,
            "version": payload.version,
            "weights_path": weights_path,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model weights dynamically: {exc!s}",
        ) from exc
