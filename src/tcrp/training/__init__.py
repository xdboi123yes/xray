"""Training loop, lifecycle observers, ablation runner, and ONNX export."""

from __future__ import annotations

from tcrp.training.ablation import AblationRunner
from tcrp.training.export import ONNXModelExporter
from tcrp.training.observers import (
    CarbonTrackerObserver,
    CheckpointObserver,
    EarlyStoppingObserver,
    LRLoggerObserver,
    MLflowObserver,
    TrainingObserver,
)
from tcrp.training.trainer import (
    CheckpointManager,
    Trainer,
    TrainingConfig,
    get_class_weighted_loss,
    get_scheduler,
    train_model,
)

__all__ = [
    "AblationRunner",
    "CarbonTrackerObserver",
    "CheckpointManager",
    "CheckpointObserver",
    "EarlyStoppingObserver",
    "LRLoggerObserver",
    "MLflowObserver",
    "ONNXModelExporter",
    "Trainer",
    "TrainingConfig",
    "TrainingObserver",
    "get_class_weighted_loss",
    "get_scheduler",
    "train_model",
]
