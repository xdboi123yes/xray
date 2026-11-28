"""Lifecycle observers attached to :class:`tcrp.training.trainer.Trainer`."""

from __future__ import annotations

import json
import os
import time
from abc import ABC
from typing import Any

import mlflow
import structlog
import torch

log = structlog.get_logger(__name__)


class TrainingObserver(ABC):
    """Lifecycle hook interface for training observers."""

    def on_train_start(self, trainer: Any) -> None:
        """Called once before the epoch loop starts."""

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        """Called at the start of each 1-indexed epoch."""

    def on_validation_end(self, epoch: int, val_metrics: dict[str, float], trainer: Any) -> None:
        """Called immediately after each validation pass."""

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        """Called at the end of each epoch with combined train/val metrics."""

    def on_train_end(self, trainer: Any) -> None:
        """Called once after the training loop finishes."""


class CheckpointObserver(TrainingObserver):
    """Persists full training snapshots whenever the monitored metric improves."""

    def __init__(
        self,
        checkpoint_path: str,
        monitor: str = "val_auc",
        mode: str = "max",
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._monitor = monitor
        self._mode = mode
        self._best_metric = -float("inf") if mode == "max" else float("inf")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        val_val = metrics.get(self._monitor)
        if val_val is None:
            return

        improved = (val_val > self._best_metric) if self._mode == "max" else (val_val < self._best_metric)
        if improved:
            from tcrp.training.trainer import CheckpointManager

            self._best_metric = val_val
            trainer.best_metric = self._best_metric
            CheckpointManager.save_state(self._checkpoint_path, trainer)
            log.info(
                "Saved best model checkpoint",
                epoch=epoch,
                monitor=self._monitor,
                value=round(val_val, 4),
                path=self._checkpoint_path,
            )


class EarlyStoppingObserver(TrainingObserver):
    """Halts training when ``monitor`` fails to improve for ``patience`` epochs."""

    def __init__(
        self,
        patience: int = 7,
        monitor: str = "val_auc",
        mode: str = "max",
    ) -> None:
        self._patience = patience
        self._monitor = monitor
        self._mode = mode
        self._best_metric = -float("inf") if mode == "max" else float("inf")
        self._epochs_no_improve = 0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        val_val = metrics.get(self._monitor)
        if val_val is None:
            return

        improved = (val_val > self._best_metric) if self._mode == "max" else (val_val < self._best_metric)
        if improved:
            self._best_metric = val_val
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
            log.info(
                "No validation improvement",
                monitor=self._monitor,
                stalled_epochs=self._epochs_no_improve,
                patience=self._patience,
            )

        if self._epochs_no_improve >= self._patience:
            trainer.stop_training = True
            log.info(
                "Early stopping triggered",
                epoch=epoch,
                monitor=self._monitor,
                patience=self._patience,
            )


class LRLoggerObserver(TrainingObserver):
    """Logs active learning rates for every optimizer parameter group at epoch start."""

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        for i, param_group in enumerate(trainer.optimizer.param_groups):
            log.info(
                "Active learning rate",
                epoch=epoch,
                group=i,
                lr=param_group["lr"],
            )


class MLflowObserver(TrainingObserver):
    """Logs training hyperparameters, metrics, and learning rates to MLflow."""

    def __init__(
        self,
        run_name: str | None = None,
        experiment_name: str = "chest_xray_tiered",
    ) -> None:
        self._run_name = run_name
        self._experiment_name = experiment_name
        self.run_id: str | None = None
        self._owns_run = False
        self._disabled = False

    def on_train_start(self, trainer: Any) -> None:
        try:
            mlflow.set_experiment(self._experiment_name)
            existing_run = mlflow.active_run()
            self._owns_run = existing_run is None
            run = mlflow.start_run(run_name=self._run_name) if self._owns_run else existing_run
            self.run_id = run.info.run_id if run is not None else None

            flat_params: dict[str, Any] = {}
            for block_name, block in trainer.config.items():
                if isinstance(block, dict):
                    for k, v in block.items():
                        flat_params[f"{block_name}_{k}"] = v
                else:
                    flat_params[block_name] = block
            mlflow.log_params(flat_params)
        except Exception as exc:
            self._disabled = True
            log.warning(
                "MLflow tracking disabled for this run after initialization failure",
                error=str(exc),
            )

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        if self._disabled:
            return
        try:
            logged_metrics = metrics.copy()
            for i, param_group in enumerate(trainer.optimizer.param_groups):
                lr_key = f"lr_group_{i}" if i > 0 else "lr"
                logged_metrics[lr_key] = param_group["lr"]
            mlflow.log_metrics(logged_metrics, step=epoch - 1)
        except Exception as exc:
            self._disabled = True
            log.warning(
                "MLflow tracking disabled after epoch metric logging failure",
                error=str(exc),
            )

    def on_train_end(self, trainer: Any) -> None:
        if self._disabled:
            self._owns_run = False
            return
        try:
            if self._owns_run and mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception as exc:
            log.warning("MLflow run teardown failed", error=str(exc))
        finally:
            self._owns_run = False


class CarbonTrackerObserver(TrainingObserver):
    """Optionally tracks CO2 emissions via ``codecarbon`` during training."""

    def __init__(self, project_name: str = "chest-xray-tiered") -> None:
        self._project_name = project_name
        self._tracker: Any = None

    def on_train_start(self, trainer: Any) -> None:
        try:
            from codecarbon import EmissionsTracker

            self._tracker = EmissionsTracker(project_name=self._project_name, log_level="warning")
            self._tracker.start()
        except ImportError:
            log.warning("codecarbon not installed; skipping emissions tracking.")
        except Exception as exc:
            log.error("Failed to start EmissionsTracker", error=str(exc))

    def on_train_end(self, trainer: Any) -> None:
        if self._tracker is not None:
            try:
                emissions = self._tracker.stop()
                if emissions is not None:
                    log.info("Carbon emissions recorded", co2_kg=round(float(emissions), 6))
            except Exception as exc:
                log.error("Failed to stop EmissionsTracker", error=str(exc))


class HistoryObserver(TrainingObserver):
    """Persists per-epoch training history, wall-clock duration, and peak VRAM to JSON."""

    def __init__(
        self,
        history_path: str,
        metadata: dict[str, Any] | None = None,
        monitor: str = "val_auc",
        mode: str = "max",
    ) -> None:
        self.history_path = history_path
        self._metadata = dict(metadata or {})
        self._monitor = monitor
        self._mode = mode
        self._train_start: float = 0.0
        self._epoch_start: float = 0.0
        self._epochs: list[dict[str, Any]] = []
        self._best_metric = -float("inf") if mode == "max" else float("inf")
        self._best_epoch = 0
        self.wall_clock_seconds: float = 0.0
        self.peak_vram_mb: float | None = None

    def _current_peak_vram_mb(self, trainer: Any) -> float | None:
        device = getattr(trainer, "device", None)
        if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            return round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 2)
        return None

    def _flush(self, trainer: Any) -> None:
        self.wall_clock_seconds = round(max(0.0, time.perf_counter() - self._train_start), 3)
        self.peak_vram_mb = self._current_peak_vram_mb(trainer)
        payload: dict[str, Any] = {
            **self._metadata,
            "epochs_completed": len(self._epochs),
            "best_epoch": self._best_epoch,
            "best_val_auc": (
                None if self._best_epoch == 0 else round(float(self._best_metric), 6)
            ),
            "wall_clock_seconds": self.wall_clock_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "epochs": self._epochs,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.history_path)), exist_ok=True)
        tmp_path = f"{self.history_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, self.history_path)

    def on_train_start(self, trainer: Any) -> None:
        self._train_start = time.perf_counter()
        device = getattr(trainer, "device", None)
        if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        _ = (epoch, trainer)
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: Any) -> None:
        epoch_seconds = round(max(0.0, time.perf_counter() - self._epoch_start), 3)
        lrs = [float(pg["lr"]) for pg in trainer.optimizer.param_groups]
        record: dict[str, Any] = {
            "epoch": int(epoch),
            "epoch_seconds": epoch_seconds,
            **{k: float(v) for k, v in metrics.items()},
            "lr_backbone": lrs[0] if lrs else None,
            "lr_head": lrs[1] if len(lrs) > 1 else (lrs[0] if lrs else None),
        }
        self._epochs.append(record)

        val_val = metrics.get(self._monitor)
        if val_val is not None:
            improved = (
                (val_val > self._best_metric)
                if self._mode == "max"
                else (val_val < self._best_metric)
            )
            if improved:
                self._best_metric = float(val_val)
                self._best_epoch = int(epoch)

        self._flush(trainer)

    def on_train_end(self, trainer: Any) -> None:
        self._flush(trainer)
        log.info(
            "Saved training history",
            history_path=self.history_path,
            epochs_completed=len(self._epochs),
            best_epoch=self._best_epoch,
            wall_clock_seconds=self.wall_clock_seconds,
            peak_vram_mb=self.peak_vram_mb,
        )
