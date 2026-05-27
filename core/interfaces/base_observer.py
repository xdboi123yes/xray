"""Base training observer interface for the Chest X-Ray classification system.

This module defines the abstract base class for observers that listen to
various training lifecycle events.
"""

from abc import ABC
from typing import Any


class TrainingObserver(ABC):
    """Abstract base class for all training lifecycle observers.

    Implements the Observer pattern to decouple concerns such as model checkpointing,
    MLflow tracking, learning rate logging, early stopping, and carbon footprint tracking
    from the main Trainer implementation.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called when training starts.

        Args:
            trainer: The active Trainer instance.
        """
        pass

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the beginning of each epoch.

        Args:
            epoch: The 1-indexed epoch number.
            trainer: The active Trainer instance.
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        """Called at the end of each epoch after validation.

        Args:
            epoch: The 1-indexed epoch number.
            metrics: Dictionary of metric names and their calculated values.
            trainer: The active Trainer instance.
        """
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called when the entire training process is complete.

        Args:
            trainer: The active Trainer instance.
        """
        pass

    def on_validation_end(self, epoch: int, val_metrics: dict[str, float], trainer: Any) -> None:
        """Called at the end of the validation cycle in an epoch.

        Args:
            epoch: The 1-indexed epoch number.
            val_metrics: Dictionary of validation metrics.
            trainer: The active Trainer instance.
        """
        pass
