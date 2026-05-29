"""Checkpoint observer for model state saving.

Saves training snapshots and weights to ensure progress retention.
"""

from typing import Any

import structlog

from core.interfaces.base_observer import TrainingObserver
from infrastructure.persistence.checkpoint import CheckpointManager

log = structlog.get_logger(__name__)


class CheckpointObserver(TrainingObserver):
    """Observer that persists model checkpoints.

    Monitors validation metrics (e.g. val_auc) at epoch end and utilizes
    CheckpointManager to save complete snapshots when performance improves.
    """

    def __init__(
        self,
        checkpoint_path: str,
        monitor: str = "val_auc",
        mode: str = "max",
    ) -> None:
        """Initialize the checkpoint observer.

        Args:
            checkpoint_path: Path where the checkpoint will be saved.
            monitor: The metric name to observe (e.g. 'val_auc', 'val_loss').
            mode: Direction of optimization ('max' for AUC, 'min' for loss).
        """
        self._checkpoint_path = checkpoint_path
        self._monitor = monitor
        self._mode = mode

        # Track internal best based on direction
        self._best_metric = -float("inf") if mode == "max" else float("inf")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        """Evaluate validation metrics and save checkpoint if performance improves.

        Args:
            epoch: The completed epoch number.
            metrics: Metrics computed for the epoch.
            trainer: The active Trainer instance.
        """
        val_val = metrics.get(self._monitor)
        if val_val is None:
            return

        improved = False
        if self._mode == "max":
            if val_val > self._best_metric:
                self._best_metric = val_val
                improved = True
        else:  # mode == 'min'
            if val_val < self._best_metric:
                self._best_metric = val_val
                improved = True

        if improved:
            trainer.best_metric = self._best_metric
            CheckpointManager.save_state(self._checkpoint_path, trainer)
            log.info(
                f"--> [CheckpointObserver] Improved! Saved best model weights "
                f"(Epoch: {epoch}, {self._monitor}: {val_val:.4f}) to {self._checkpoint_path}"
            )
            _print = print
            _print(
                f"💾 [Checkpoint] Metric improved! Saved best model weights (AUC: {val_val:.4f}) to {self._checkpoint_path}",
                flush=True,
            )
