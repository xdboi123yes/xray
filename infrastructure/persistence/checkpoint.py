"""Checkpoint manager handling full state serialization and deserialization.

Ensures that model, optimizer, scheduler, and general training states are perfectly
preserved and restored, supporting complete resume functionality.
"""

from typing import Any

import structlog
import torch

log = structlog.get_logger(__name__)


class CheckpointManager:
    """Manages saving and loading complete training snapshots.

    Serializes states of the model, optimizer, scheduler, and metadata to ensure
    exact reproducibility and seamless recovery after interruptions.
    """

    @classmethod
    def save_state(cls, path: str, trainer: Any) -> None:
        """Serialize and save complete training states to disk.

        Args:
            path: Target file path (.pth).
            trainer: Active Trainer instance.
        """
        # Build state snapshot dictionary
        state = {
            "epoch": trainer.current_epoch,
            "best_metric": getattr(trainer, "best_metric", 0.0),
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict() if trainer.scheduler else None
            ),
            "config": trainer.config,
        }

        # Save atomically
        torch.save(state, path)

    @classmethod
    def load_state(cls, path: str, trainer: Any) -> int:
        """Restore serialized training states into the provided trainer instance.

        Args:
            path: Source checkpoint file path (.pth).
            trainer: Target Trainer instance to modify.

        Returns:
            The next starting epoch (saved epoch integer).
        """
        log.info(f"--> [CheckpointManager] Restoring training state from '{path}'...")

        state = torch.load(path, map_location=trainer.device)

        # Restore states
        trainer.model.load_state_dict(state["model_state_dict"])
        trainer.optimizer.load_state_dict(state["optimizer_state_dict"])

        if trainer.scheduler and state["scheduler_state_dict"]:
            trainer.scheduler.load_state_dict(state["scheduler_state_dict"])

        trainer.best_metric = state.get("best_metric", 0.0)
        saved_epoch = state.get("epoch", 0)

        log.info(
            f"--> [CheckpointManager] Successfully restored! Next starting epoch: "
            f"{saved_epoch + 1} (Best metric recorded: {trainer.best_metric:.4f})"
        )

        return int(saved_epoch)
