"""Ablation evaluation results REST endpoints."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException, status, Response

router = APIRouter(prefix="/api/v1", tags=["ablation"])

ABLATION_FILE = "outputs/results/ablation.json"


@router.get("/ablation")
async def get_ablation_records(response: Response) -> Any:
    """Returns thesis-ready A1-A15 model evaluation ablation matrices."""
    if not os.path.exists(ABLATION_FILE):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ablation evaluation results not yet generated on this deployment.",
        )

    try:
        with open(ABLATION_FILE, encoding="utf-8") as f:
            data = json.load(f)
            # Add dynamic provenance headers to warn API clients
            if any(row.get("provenance") == "preliminary_placeholder" for row in data):
                response.headers["X-Data-Provenance"] = "preliminary"
            else:
                response.headers["X-Data-Provenance"] = "mlflow_run"
            return data
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load ablation results: {exc!s}",
        ) from exc

