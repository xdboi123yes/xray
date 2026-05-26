"""Learning rate logging observer for training visibility.

Monitors and prints the current learning rate of all parameter groups
at the beginning of each epoch.
"""

from typing import Any

from core.interfaces.base_observer import TrainingObserver

import structlog

log = structlog.get_logger(__name__)


class LRLoggerObserver(TrainingObserver):
    """Observer that logs learning rates at epoch start.

    Prints learning rate transitions, assisting with debugging scheduler decays.
    """

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Log active learning rates for the starting epoch.

        Args:
            epoch: The starting epoch number.
            trainer: The active Trainer instance.
        """
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            lr = param_group["lr"]
            group_suffix = f" (Group {i})" if len(trainer.optimizer.param_groups) > 1 else ""
            log.info(f"--> [LRLogger] Epoch {epoch}: Current Learning Rate{group_suffix} is {lr:.6g}")
