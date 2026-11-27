"""Split-conformal prediction calibrated on the dedicated Calibration Split.

Computes finite-sample nonconformity quantile ``q_hat`` on temperature-scaled
probabilities (with exchangeability preserved by fitting temperature on the
separate Validation Split) and maps softmax distributions to predictive sets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
import torch

log = structlog.get_logger(__name__)


class ConformalPredictor:
    """Split-conformal predictor mapping probability tensors to predictive label sets."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.q_hat: float | None = None
        self.classes = ["No Finding", "Pneumothorax"]

    def calibrate(
        self,
        model: Any,
        cal_loader: Any,
        device: torch.device,
        temperature: float = 1.0,
    ) -> float:
        """Compute finite-sample nonconformity quantile ``q_hat`` on ``cal_loader``.

        Args:
            model: Classifier (or ``ModelWithTemperature``) returning logits.
            cal_loader: DataLoader over the held-out Calibration Split.
            device: Compute device.
            temperature: Optional post-hoc temperature scalar ``T_cal`` applied to
                logits if ``model`` is not already temperature-scaled.

        Returns:
            The calibrated nonconformity quantile ``q_hat``.
        """
        model.eval()
        scores: list[float] = []

        log.info("Calibrating split-conformal predictor", alpha=self.alpha, temperature=temperature)
        with torch.no_grad():
            for batch in cal_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)

                logits = model(images)
                if temperature != 1.0:
                    logits = logits / float(temperature)
                probs = torch.softmax(logits, dim=1)

                true_class_probs = probs[torch.arange(len(labels)), labels]
                nc_scores = 1.0 - true_class_probs
                scores.extend(nc_scores.cpu().numpy().tolist())

        scores_arr = np.asarray(scores, dtype=float)
        n = len(scores_arr)
        if n == 0:
            raise ValueError("Calibration split is empty; cannot compute q_hat.")

        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))
        self.q_hat = float(np.quantile(scores_arr, q_level, method="higher"))
        log.info("Split-conformal calibration complete", q_hat=round(self.q_hat, 4), n=n)
        return self.q_hat

    def predict_set(self, probs: torch.Tensor) -> list[str]:
        """Map ``[1, num_classes]`` softmax probabilities to a conformal prediction set."""
        if self.q_hat is None:
            raise ValueError("ConformalPredictor is not calibrated yet.")

        threshold = 1.0 - self.q_hat
        probs_np = probs.detach().cpu().numpy()[0]
        prediction_set = [self.classes[i] for i, prob in enumerate(probs_np) if prob >= threshold]

        if not prediction_set:
            prediction_set.append(self.classes[int(np.argmax(probs_np))])
        return prediction_set

    def save(self, path: str) -> None:
        """Persist ``q_hat`` and ``alpha`` to disk."""
        torch.save({"q_hat": self.q_hat, "alpha": self.alpha}, path)

    def load(self, path: str) -> None:
        """Load saved conformal threshold state (tolerant of scalar or partial dicts)."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            self.q_hat = data.get("q_hat", self.q_hat)
            self.alpha = data.get("alpha", self.alpha)
        else:
            self.q_hat = float(data)
