"""Conformal Prediction for Classification.

Ensures rigorous distribution-free predictive sets with statistical coverage
guarantees (1 - alpha).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
import torch

log = structlog.get_logger(__name__)


class ConformalPredictor:
    """Split-conformal predictor mapping probability tensors to predictive sets."""

    def __init__(self, alpha: float = 0.05) -> None:
        """Initializes the ConformalPredictor.

        Args:
            alpha: Target error rate (e.g., 0.05 guarantees 95% coverage).
        """
        self.alpha = alpha
        self.q_hat: float | None = None
        self.classes = ["No Finding", "Pneumothorax"]

    def calibrate(self, model: Any, cal_loader: Any, device: torch.device) -> None:
        """Computes nonconformity threshold q_hat from held-out calibration split."""
        model.eval()
        scores: list[float] = []

        log.info("Calibrating Conformal Predictor...")
        with torch.no_grad():
            for images, labels, _ in cal_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)

                # Extracts the probability of the true label
                true_class_probs = probs[torch.arange(len(labels)), labels]

                # Nonconformity score defined as: 1.0 - P(true_class)
                nc_scores = 1.0 - true_class_probs
                scores.extend(nc_scores.cpu().numpy())

        scores_arr = np.array(scores)
        n = len(scores_arr)

        # Finite-sample correction quantile level calculation
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        if q_level > 1.0:
            q_level = 1.0

        self.q_hat = float(np.quantile(scores_arr, q_level, method="higher"))
        log.info(f"Calibration complete. Nonconformity threshold (q_hat): {self.q_hat:.4f}")

    def predict_set(self, probs: torch.Tensor) -> list[str]:
        """Maps output softmax probabilities to conformal prediction sets.

        Args:
            probs: softmax probability tensor of shape [1, num_classes].

        Returns:
            A list containing active label classes inside the prediction interval.
        """
        if self.q_hat is None:
            raise ValueError("ConformalPredictor is not calibrated yet.")

        threshold = 1.0 - self.q_hat
        probs_np = probs.cpu().numpy()[0]
        prediction_set = []

        for i, prob in enumerate(probs_np):
            if prob >= threshold:
                prediction_set.append(self.classes[i])

        # Graceful fallback: return argmax if threshold sweeps all labels out
        if not prediction_set:
            best_idx = int(np.argmax(probs_np))
            prediction_set.append(self.classes[best_idx])

        return prediction_set

    def save(self, path: str) -> None:
        """Saves conformal thresholds to file.

        Args:
            path: Target file path.
        """
        torch.save({"q_hat": self.q_hat, "alpha": self.alpha}, path)

    def load(self, path: str) -> None:
        """Loads conformal thresholds from file.

        Args:
            path: Source file path.
        """
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.q_hat = data["q_hat"]
        self.alpha = data["alpha"]
