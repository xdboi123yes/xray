"""TCR-P (Tiered Confidence-Routing System) cascade coordinator.

Evaluates every incoming radiograph with Tier 1, routes uncertain cases via
the configured router (Escalation) to a single Tier 2 specialist backbone with
MC Dropout + TTA Uncertainty Decomposition and split-conformal prediction sets,
and flags ambiguous or high-variance cases for Human Deferral.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
import torch.nn as nn

from tcrp.models.base import BaseClassifier
from tcrp.routing import BaseRouter, DynamicThresholdRouter


def log_inference(log_path: str, record: dict[str, Any]) -> None:
    """Append one routing telemetry row to ``log_path``, writing headers if absent."""
    file_exists = os.path.isfile(log_path)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


@dataclass
class TieredPrediction:
    """Structured prediction outcome from one TCR-P cascade forward pass."""

    prediction: str
    confidence: float
    mc_variance: float | None
    tier_used: int
    inference_time_ms: float = 0.0
    conformal_set: list[str] | None = None
    conformal_coverage: float = 0.95
    tta_passes: int = 10
    mc_passes: int = 20
    flagged_for_review: bool = False


class TieredSystem(nn.Module):  # type: ignore[misc]
    """Two-stage TCR-P cascade with Escalation and Human Deferral."""

    def __init__(
        self,
        tier1_model: BaseClassifier,
        tier2_model: BaseClassifier,
        config: dict[str, Any],
        conformal_predictor: Any = None,
        router: BaseRouter | None = None,
        variance_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.tier1 = tier1_model
        self.tier2 = tier2_model
        self.config = config
        self.conformal_predictor = conformal_predictor
        self.variance_threshold = variance_threshold

        model_cfg = config.get("model", {})
        self.router: BaseRouter = router or DynamicThresholdRouter(
            initial_threshold=float(model_cfg.get("confidence_threshold", 0.75)),
            window_size=int(model_cfg.get("threshold_window_size", 50)),
            threshold_delta=float(model_cfg.get("threshold_delta", 0.05)),
        )
        self.classes = ["No Finding", "Pneumothorax"]

    @property
    def current_threshold(self) -> float:
        """Expose active routing threshold from the underlying router."""
        return self.router.current_threshold

    def update_threshold(self, conf: float) -> None:
        """Delegate Adaptive Routing Threshold update to the configured router."""
        self.router.update(conf)

    def route(
        self,
        x: torch.Tensor,
        image_id: str = "unknown",
        log_path: str | None = None,
        ground_truth: int | None = None,
    ) -> TieredPrediction:
        """Route one radiograph tensor ``[1, 3, H, W]`` through Tier 1 and optional Tier 2."""
        start_time = time.perf_counter()

        with torch.no_grad():
            t1_logits = self.tier1(x)
            t1_conf_t, t1_pred_t = self.tier1.get_confidence(t1_logits)
            conf_val = float(t1_conf_t.item())
            pred_class = int(t1_pred_t.item())

            self.router.update(conf_val)
            selected_tier = self.router.route(conf_val)

            if selected_tier == 1:
                inference_time = (time.perf_counter() - start_time) * 1000.0
                prediction_result = TieredPrediction(
                    prediction=self.classes[pred_class],
                    confidence=conf_val,
                    mc_variance=None,
                    tier_used=1,
                    inference_time_ms=inference_time,
                )
            else:
                model_cfg = self.config.get("model", {})
                mc_passes = int(model_cfg.get("mc_dropout_passes", 20))
                tta_passes = int(model_cfg.get("tta_passes", 10))

                t2_mean_probs, t2_variance = self.tier2.mc_tta_forward(
                    x, T=mc_passes, n_augments=tta_passes
                )
                t2_conf_t, t2_pred_t = torch.max(t2_mean_probs, dim=1)
                t2_conf_val = float(t2_conf_t.item())
                t2_pred_class = int(t2_pred_t.item())
                var_val = float(t2_variance[0, t2_pred_class].item())

                conformal_set: list[str] | None = None
                conformal_coverage = 0.95
                if self.conformal_predictor and self.conformal_predictor.q_hat is not None:
                    conformal_set = self.conformal_predictor.predict_set(t2_mean_probs)
                    conformal_coverage = 1.0 - float(self.conformal_predictor.alpha)

                inference_time = (time.perf_counter() - start_time) * 1000.0
                flagged = (var_val > self.variance_threshold) or (
                    conformal_set is not None and len(conformal_set) > 1
                )

                prediction_result = TieredPrediction(
                    prediction=self.classes[t2_pred_class],
                    confidence=t2_conf_val,
                    mc_variance=var_val,
                    tier_used=2,
                    inference_time_ms=inference_time,
                    conformal_set=conformal_set,
                    conformal_coverage=conformal_coverage,
                    tta_passes=tta_passes,
                    mc_passes=mc_passes,
                    flagged_for_review=flagged,
                )

            if log_path:
                predicted_label_int = self.classes.index(prediction_result.prediction)
                correct_val = (
                    int(ground_truth == predicted_label_int) if ground_truth is not None else None
                )
                c_set = prediction_result.conformal_set
                c_set_str = ", ".join(c_set) if isinstance(c_set, list) else str(c_set)
                log_inference(
                    log_path,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "image_id": image_id,
                        "tier_used": prediction_result.tier_used,
                        "tier1_confidence": conf_val,
                        "final_prediction": prediction_result.prediction,
                        "ground_truth": ground_truth,
                        "mc_variance": prediction_result.mc_variance,
                        "conformal_set": c_set_str,
                        "inference_time_ms": prediction_result.inference_time_ms,
                        "correct": correct_val,
                    },
                )

            return prediction_result
