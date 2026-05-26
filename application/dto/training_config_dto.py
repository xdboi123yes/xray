"""Model training loop parameterization Data Transfer Objects (DTOs).

Defines strict type schemas to pass hyperparameters and constraints to training runs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingConfigDTO(BaseModel):
    """Hyperparameters and configuration mapping for deep learning training sessions."""

    backbone: str = Field(
        ..., description="Backbone architecture name (e.g. 'mobilenet_v2', 'efficientnet_b4', 'ark_plus')."
    )
    run_name: str = Field(..., description="Unique name allocated to the MLflow logging run.")
    batch_size: int = Field(32, ge=1, description="Size of training data loader batches.")
    epochs: int = Field(50, ge=1, description="Total running epoch iteration boundaries.")
    lr_backbone: float = Field(1e-4, gt=0.0, description="Optimizer learning rate for base features.")
    lr_head: float = Field(1e-3, gt=0.0, description="Optimizer learning rate for classifier heads.")
    early_stopping_patience: int = Field(7, ge=1, description="Patience epochs before early termination.")
    seed: int = Field(42, description="Randomization seed state.")
    use_synthetic: bool = Field(True, description="True to inject synthetic image sets into Pneumothorax classes.")
