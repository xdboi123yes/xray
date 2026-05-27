"""History Diagnostics Persistence REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from application.dto.prediction_dto import HistoryRecordDTO
from infrastructure.persistence.prediction_log import HistoryDatabaseManager
from web.backend.deps import get_db

router = APIRouter(prefix="/api/v1", tags=["history"])


@router.get("/history", response_model=list[HistoryRecordDTO])
async def get_prediction_history(
    limit: int = 50,
    offset: int = 0,
    db: HistoryDatabaseManager = Depends(get_db),
) -> Any:
    """Retrieves paginated past diagnostic prediction log entries."""
    return db.get_history(limit=limit, offset=offset)


@router.delete("/history/{request_id}")
async def delete_history_record(
    request_id: str, db: HistoryDatabaseManager = Depends(get_db)
) -> Any:
    """Deletes a past prediction record by its request_id from SQLite history."""
    deleted = db.delete_record(request_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Record not found in system history.",
        )
    return {"status": "deleted"}
