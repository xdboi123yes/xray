"""Inference REST endpoints for chest radiograph tiered predictions."""



from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status, Request

from application.dto.prediction_dto import PredictionDTO
from application.services.inference_service import InferenceService
from core.interfaces.base_model import BaseClassifier
from core.interfaces.base_router import BaseRouter
from web.backend.database import HistoryDatabaseManager
from web.backend.middleware import limiter
from web.backend.deps import (
    SystemState,
    get_db,
    get_router,
    get_system_state,
    get_tier1_model,
    get_tier2_model,
)

router = APIRouter(prefix="/api/v1", tags=["inference"])
inference_service = InferenceService()


@router.post("/predict", response_model=PredictionDTO)
@limiter.limit("20/minute")
async def predict_single(
    request: Request,
    file: UploadFile,
    return_gradcam: bool = True,
    db: HistoryDatabaseManager = Depends(get_db),
    tier1: BaseClassifier = Depends(get_tier1_model),
    tier2: BaseClassifier = Depends(get_tier2_model),
    router_algo: BaseRouter = Depends(get_router),
    state: SystemState = Depends(get_system_state),
) -> Any:
    """Uploads a chest radiograph and performs dynamic tiered routing.

    Saves diagnostic history logs to SQLite.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise ValueError("File uploaded must be an image format.")

    file_bytes = await file.read()
    # Execute tiered pipeline via decoupled application service
    result_dto = inference_service.predict(
        image_bytes=file_bytes,
        tier1=tier1,
        tier2=tier2,
        router=router_algo,
        device=state.device,
        return_gradcam=return_gradcam,
    )

    # Save details to database persistence
    db.save_prediction(
        request_id=result_dto.request_id,
        prediction=result_dto.prediction,
        confidence=result_dto.confidence,
        tier_used=result_dto.tier_used,
        mc_variance=result_dto.mc_variance,
        flagged_for_review=result_dto.flagged_for_review,
        timestamp=result_dto.timestamp,
    )

    return result_dto


@router.post("/predict/batch", response_model=list[PredictionDTO])
@limiter.limit("5/minute")
async def predict_batch(
    request: Request,
    files: list[UploadFile],
    db: HistoryDatabaseManager = Depends(get_db),
    tier1: BaseClassifier = Depends(get_tier1_model),
    tier2: BaseClassifier = Depends(get_tier2_model),
    router_algo: BaseRouter = Depends(get_router),
    state: SystemState = Depends(get_system_state),
) -> Any:
    """Performs dynamic tiered diagnostics for multiple uploaded radiographs (max 50)."""
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch limit exceeded. Maximum 50 images allowed.",
        )

    predictions = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            continue

        file_bytes = await file.read()
        result_dto = inference_service.predict(
            image_bytes=file_bytes,
            tier1=tier1,
            tier2=tier2,
            router=router_algo,
            device=state.device,
            return_gradcam=True,
        )

        db.save_prediction(
            request_id=result_dto.request_id,
            prediction=result_dto.prediction,
            confidence=result_dto.confidence,
            tier_used=result_dto.tier_used,
            mc_variance=result_dto.mc_variance,
            flagged_for_review=result_dto.flagged_for_review,
            timestamp=result_dto.timestamp,
        )

        predictions.append(result_dto)

    return predictions
