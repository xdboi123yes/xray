"""Application Data Transfer Objects (DTOs) for Chest X-Ray classification.

Defines schemas for prediction responses, thresholds, registry operations,
history, and system health statistics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionDTO(BaseModel):
    """Data transfer object representing a chest X-ray diagnosis prediction."""

    request_id: str = Field(..., description="Unique UUID generated for the diagnostic request.")
    prediction: str = Field(..., description="Final diagnostic decision: 'Pneumothorax' or 'No Finding'.")
    confidence: float = Field(..., description="Prediction probability confidence score between 0.0 and 1.0.")
    tier_used: int = Field(..., description="Classifier tier utilized: 1 (MobileNet) or 2 (Ark+).")
    mc_variance: float | None = Field(None, description="Monte Carlo Dropout variance for uncertainty rating.")
    mc_passes: int | None = Field(None, description="Number of Monte Carlo dropout forward passes run.")
    tta_passes: int | None = Field(None, description="Number of Test-Time Augmentation passes run.")
    conformal_set: list[str] | None = Field(None, description="Set prediction boundaries calculated via conformal calibration.")
    conformal_coverage: float | None = Field(None, description="Nominal coverage guarantees (e.g. 0.95).")
    flagged_for_review: bool = Field(..., description="Whether the sample was flagged due to extreme high uncertainty.")
    inference_time_ms: float = Field(..., description="Total wall-clock duration in milliseconds for preprocessing and inference.")
    gradcam_tier1_b64: str | None = Field(None, description="Base64-encoded Grad-CAM diagnostic heatmap from Tier 1.")
    gradcam_tier2_b64: str | None = Field(None, description="Base64-encoded Grad-CAM diagnostic heatmap from Tier 2.")
    model_version: str = Field(..., description="Active versions of the deployed diagnostic models.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of the completed diagnostic request.")


class ThresholdDTO(BaseModel):
    """Data transfer object for checking or setting classifier routing thresholds."""

    value: float = Field(..., ge=0.0, le=1.0, description="Routing threshold confidence bound.")
    mode: str = Field(..., description="Threshold calibration mode: 'static' or 'dynamic'.")


class ModelLoadDTO(BaseModel):
    """Data transfer object to instruct dynamically loading a model from the registry."""

    name: str = Field(..., description="Name of the model backbone to load (e.g. 'tier2_arkplus').")
    version: str = Field(..., description="Registry model version index to retrieve.")


class HealthDTO(BaseModel):
    """Data transfer object representing system diagnostic health status."""

    status: str = Field(..., description="System status indicator ('ok' or 'degraded').")
    gpu: bool = Field(..., description="Whether a GPU/MPS accelerator is active and registered by PyTorch.")
    models_loaded: list[str] = Field(..., description="Identifiers of currently instantiated model layers in memory.")
    version: str = Field(..., description="Deployed semantic API software version.")
    uptime_s: float = Field(..., description="Total system uptime count in seconds.")


class HistoryRecordDTO(BaseModel):
    """Simplified data transfer object representing past prediction records stored in SQLite database."""

    id: int = Field(..., description="Internal database unique sequence identifier.")
    request_id: str = Field(..., description="UUID of the original prediction request.")
    prediction: str = Field(..., description="Diagnostic outcome: 'Pneumothorax' or 'No Finding'.")
    confidence: float = Field(..., description="Prediction probability confidence level.")
    tier_used: int = Field(..., description="Model tier sequence used: 1 or 2.")
    mc_variance: float | None = Field(None, description="Uncertainty MC variance if recorded.")
    flagged_for_review: bool = Field(..., description="Whether flagged for visual clinician oversight.")
    timestamp: str = Field(..., description="ISO 8601 timestamp of record creation.")
