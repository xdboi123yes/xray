"""MLflow training observer for model tracking.

Hooks into the training process to log metrics, hyperparameters, and active learning
rates to MLflow tracking services.
"""

from typing import Any

import mlflow

from core.interfaces.base_observer import TrainingObserver

import structlog

log = structlog.get_logger(__name__)


class MLflowObserver(TrainingObserver):
    """Observer that logs parameters and metrics to MLflow.

    Automates setting up active runs and logging epoch statistics
    (losses, learning rates, validation AUC) dynamically.
    """

    def __init__(
        self,
        run_name: str | None = None,
        experiment_name: str = "chest_xray_tiered",
    ) -> None:
        """Initialize the MLflow observer.

        Args:
            run_name: The name of this training run (e.g. 'Tier2_EfficientNet').
            experiment_name: The MLflow experiment group name.
        """
        self._run_name = run_name
        self._experiment_name = experiment_name

    def on_train_start(self, trainer: Any) -> None:
        """Called when training starts. Ensures an active MLflow run is open.

        Logs configuration parameters to the run.

        Args:
            trainer: The active Trainer instance.
        """
        try:
            # Setup experiment and start run if not already active
            mlflow.set_experiment(self._experiment_name)
            if mlflow.active_run() is None:
                mlflow.start_run(run_name=self._run_name)

            log.info(
                f"[MLflowObserver] Active run started: '{self._run_name}' "
                f"under experiment '{self._experiment_name}'"
            )

            # Log trainer config flat parameters
            flat_params: dict[str, Any] = {}
            for block_name, block in trainer.config.items():
                if isinstance(block, dict):
                    for k, v in block.items():
                        flat_params[f"{block_name}_{k}"] = v
                else:
                    flat_params[block_name] = block

            mlflow.log_params(flat_params)

        except Exception as e:
            log.error(f"[MLflowObserver] Warning: Failed to initialize run: {e}")

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], trainer: Any
    ) -> None:
        """Called at the end of each epoch after validation to log metrics to MLflow.

        Args:
            epoch: The completed epoch number.
            metrics: Metrics computed for the epoch.
            trainer: The active Trainer instance.
        """
        try:
            # Add active learning rates
            logged_metrics = metrics.copy()
            for i, param_group in enumerate(trainer.optimizer.param_groups):
                lr_key = f"lr_group_{i}" if i > 0 else "lr"
                logged_metrics[lr_key] = param_group["lr"]

            mlflow.log_metrics(logged_metrics, step=epoch - 1)
        except Exception as e:
            log.error(f"[MLflowObserver] Warning: Failed to log epoch metrics: {e}")

    def on_train_end(self, trainer: Any) -> None:
        """Called when training is complete. Closes the active MLflow run.

        Args:
            trainer: The active Trainer instance.
        """
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
                log.info(f"[MLflowObserver] Run '{self._run_name}' ended successfully.")
        except Exception as e:
            log.error(f"[MLflowObserver] Warning: Failed to terminate run cleanly: {e}")
