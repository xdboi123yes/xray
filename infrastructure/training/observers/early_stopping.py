"""Early stopping observer to halt training when validation metric stalls.

Avoids overfitting and saves compute time by stopping epoch progression when
no improvements are recorded over a set patience window.
"""

from typing import Any

import structlog

from core.interfaces.base_observer import TrainingObserver

log = structlog.get_logger(__name__)


class EarlyStoppingObserver(TrainingObserver):
    """Observer that enforces early stopping of training loops.

    Monitors a designated validation metric (like val_auc). Stops the trainer's
    epoch iterations if the metric fails to improve after `patience` epochs.
    """

    def __init__(
        self,
        patience: int = 7,
        monitor: str = "val_auc",
        mode: str = "max",
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement before stopping.
            monitor: The metric name to observe (e.g. 'val_auc', 'val_loss').
            mode: Direction of optimization ('max' for AUC, 'min' for loss).
        """
        self._patience = patience
        self._monitor = monitor
        self._mode = mode

        self._best_metric = -float("inf") if mode == "max" else float("inf")
        self._epochs_no_improve = 0

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], trainer: Any
    ) -> None:
        """Evaluate validation metrics and set trainer.stop_training if patience is exceeded.

        Args:
            epoch: The completed epoch number.
            metrics: Metrics computed for the epoch.
            trainer: The active Trainer instance.
        """
        val_val = metrics.get(self._monitor, None)
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
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
            log.info(
                f"--> [EarlyStoppingObserver] No improvement for "
                f"{self._epochs_no_improve}/{self._patience} epochs."
            )
            print(
                f"⚠️ [EarlyStopping] No improvement on '{self._monitor}' for {self._epochs_no_improve}/{self._patience} epochs.",
                flush=True
            )

        if self._epochs_no_improve >= self._patience:
            trainer.stop_training = True
            log.info(
                f"[EarlyStoppingObserver] Early stopping triggered at epoch {epoch}! "
                f"No improvement on '{self._monitor}' for {self._patience} consecutive epochs."
            )
            print(
                f"\n🛑 [EarlyStopping] Triggered at epoch {epoch}! Halting training since '{self._monitor}' did not improve for {self._patience} consecutive epochs.",
                flush=True
            )
