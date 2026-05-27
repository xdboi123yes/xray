"""Application Service coordinating logit calibration and expected calibration error estimation.

Integrates parametric temperature scaling optimization and generates clinical
reliability diagram visual metrics.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn

from config.settings import get_settings
from core.uncertainty.calibration import (
    ModelWithTemperature,
    compute_ece,
    plot_reliability_diagram,
)

log = structlog.get_logger(__name__)


class CalibrationService:
    """Orchestrates model calibration, temperature scaling, and reliability mapping."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initializes the CalibrationService.

        Loads project path settings to determine where metrics plots are stored.
        """
        self.settings = get_settings()
        self.figures_dir = self.settings.paths.figures
        os.makedirs(self.figures_dir, exist_ok=True)

    def calculate_ece(
        self,
        probabilities: Sequence[float] | np.ndarray[Any, Any],
        labels: Sequence[int] | np.ndarray[Any, Any],
        n_bins: int = 10,
    ) -> float:
        """Wrapper method computing Expected Calibration Error (ECE) for predictions.

        Args:
            probabilities: Probability predictions list or numpy array.
            labels: Ground truth binary labels.
            n_bins: Number of confidence bins.

        Returns:
            The calculated ECE score.
        """
        probs_np = np.asarray(probabilities)
        labels_np = np.asarray(labels)
        return float(compute_ece(probs_np, labels_np, n_bins=n_bins))

    def calibrate_model(
        self,
        model: nn.Module,
        validation_loader: Any,
        device: torch.device | None = None,
    ) -> float:
        """Tunes logit scaling temperatures using L-BFGS optimization on validation set.

        Args:
            model: PyTorch module returning raw logits.
            validation_loader: PyTorch DataLoader for the validation dataset.
            device: Active hardware device.

        Returns:
            The optimal temperature scale value.
        """
        active_device = device or torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        log.info("[CalibrationService] Initializing temperature scaling calibrator...")
        calibrator = ModelWithTemperature(model)
        calibrator.set_temperature(validation_loader, active_device)

        optimal_temp: float = float(calibrator.temperature.item())
        log.info(
            f"[CalibrationService] Calibration completed. Optimal Temp (T) = {optimal_temp:.4f}"
        )
        return optimal_temp

    def save_reliability_diagram(
        self,
        probabilities: Sequence[float] | np.ndarray[Any, Any],
        labels: Sequence[int] | np.ndarray[Any, Any],
        filename: str = "reliability_diagram.png",
        n_bins: int = 10,
    ) -> str:
        """Plots a reliability calibration diagram and saves it to outputs folder.

        Args:
            probabilities: Positive prediction probabilities.
            labels: Ground truth labels.
            filename: Target file name.
            n_bins: Discretization bin count.

        Returns:
            Absolute file path of the saved figure.
        """
        probs_np = np.asarray(probabilities)
        labels_np = np.asarray(labels)

        save_path = os.path.join(self.figures_dir, filename)
        plot_reliability_diagram(
            probs_np,
            labels_np,
            n_bins=n_bins,
            save_path=save_path,
        )
        log.info(f"[CalibrationService] Reliability diagram saved successfully at {save_path}")
        return os.path.abspath(save_path)
