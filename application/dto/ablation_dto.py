"""Ablation studies and model diagnostics Data Transfer Objects (DTOs).

Provides data schemas to represent experiment configs, parameter overrides,
and clinical metrics outcomes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AblationConfigDTO(BaseModel):
    """Configuration definition representing a clinical ablation experiment run."""

    ablation_id: str = Field(..., description="Unique alphanumeric identifier (e.g. 'A11', 'A13').")
    name: str = Field(..., description="Short alphanumeric clinical name representing the experiment.")
    description: str = Field(..., description="Long descriptive paragraph detailing the ablated elements.")
    script: str = Field(..., description="Python execution path target.")
    args: list[str] = Field(default_factory=list, description="Target CLI parameters to supply.")
    overrides: dict[str, Any] = Field(
        default_factory=dict, description="YAML config attributes to dynamically override."
    )


class AblationResultDTO(BaseModel):
    """Detailed metadata and outcomes returned from a completed ablation experiment."""

    ablation_id: str = Field(..., description="Experiment sequence identifier.")
    name: str = Field(..., description="Clinical experiment name.")
    description: str = Field(..., description="Experiment description details.")
    started_at: str = Field(..., description="ISO 8601 UTC starting timestamp.")
    completed_at: str | None = Field(None, description="ISO 8601 UTC completion timestamp.")
    status: str = Field(..., description="Final running status: 'SUCCESS' or 'FAILED'.")
    metrics: dict[str, float] = Field(
        default_factory=dict, description="Calculated test metrics obtained during the run."
    )
    error: str | None = Field(None, description="Detailed traceback logs if execution crashed.")
