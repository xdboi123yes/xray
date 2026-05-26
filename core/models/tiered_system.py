"""Tiered Routing Classifier System for Chest X-Rays.

Combines lightweight screening (Tier 1) and deep specialized models (Tier 2)
using threshold escalation, conformal prediction sets, and MC Dropout uncertainty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from core.interfaces.base_model import BaseClassifier


@dataclass
class TieredPrediction:
    """Diagnostic outcome record from the tiered prediction pipeline."""
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


class TieredSystem(nn.Module):
    """Multi-tiered confidence-routing diagnostic system wrapper."""

    def __init__(
        self,
        tier1_model: BaseClassifier,
        tier2_model: BaseClassifier,
        config: dict[str, Any],
        conformal_predictor: Any = None,
    ) -> None:
        """Initializes the tiered classifier.

        Args:
            tier1_model: Lightweight, high-throughput classifier (e.g. MobileNetV2).
            tier2_model: Deep, high-capacity specialist classifier (e.g. EfficientNet / Ark+).
            config: General settings dict.
            conformal_predictor: Optional conformal inference calibrator.
        """
        super().__init__()
        self.tier1 = tier1_model
        self.tier2 = tier2_model
        self.config = config
        self.conformal_predictor = conformal_predictor

        # Dynamic threshold variables
        self.current_threshold: float = config["model"]["confidence_threshold"]
        self.window_size: int = config["model"]["threshold_window_size"]
        self.threshold_delta: float = config["model"]["threshold_delta"]
        self.recent_confidences: list[float] = []

        self.classes = ["No Finding", "Pneumothorax"]

    def update_threshold(self, conf: float) -> None:
        """Dynamically shifts the routing decision threshold based on confidence streams.

        Args:
            conf: Latest prediction confidence score.
        """
        self.recent_confidences.append(conf)
        if len(self.recent_confidences) > self.window_size:
            self.recent_confidences.pop(0)

            mean_conf = sum(self.recent_confidences) / self.window_size

            # Adaptive shift logic
            if mean_conf < 0.65:
                # Lower threshold to escalate more ambiguous cases to Tier 2
                self.current_threshold = max(0.5, self.current_threshold - self.threshold_delta)
            elif mean_conf > 0.85:
                # Raise threshold to optimize execution speed for high-confidence runs
                self.current_threshold = min(0.95, self.current_threshold + self.threshold_delta)

    def route(
        self,
        x: torch.Tensor,
        image_id: str = "unknown",
        log_path: str | None = None,
        ground_truth: int | None = None,
    ) -> TieredPrediction:
        """Routes a radiograph tensor dynamically through active tiers.

        Args:
            x: Input preprocessed image tensor of shape [1, 3, 224, 224].
            image_id: Unique string identifier for logs.
            log_path: Optional CSV path to record routing telemetry.
            ground_truth: Optional label for validation parity logs.

        Returns:
            A populated TieredPrediction record containing results and metrics.
        """
        start_time = time.time()

        with torch.no_grad():
            # 1. Forward through Tier 1
            t1_logits = self.tier1(x)
            t1_conf_t, t1_pred_t = self.tier1.get_confidence(t1_logits)

            conf_val = float(t1_conf_t.item())
            pred_class = int(t1_pred_t.item())

            self.update_threshold(conf_val)

            # 2. Routing Decision
            if conf_val >= self.current_threshold:
                inference_time = (time.time() - start_time) * 1000.0
                prediction_result = TieredPrediction(
                    prediction=self.classes[pred_class],
                    confidence=conf_val,
                    mc_variance=None,
                    tier_used=1,
                    inference_time_ms=inference_time,
                )
            else:
                # 3. Fallback to specialized Tier 2 Model with MC Dropout + TTA
                mc_passes = self.config["model"].get("mc_dropout_passes", 20)
                tta_passes = self.config["model"].get("tta_passes", 10)

                t2_mean_probs, t2_variance = self.tier2.mc_tta_forward(
                    x, T=mc_passes, n_augments=tta_passes
                )

                t2_conf_t, t2_pred_t = torch.max(t2_mean_probs, dim=1)
                t2_conf_val = float(t2_conf_t.item())
                t2_pred_class = int(t2_pred_t.item())

                # Extract predictive uncertainty variance
                var_val = float(t2_variance[0, t2_pred_class].item())

                # Conformal Prediction set mapping
                conformal_set = None
                conformal_coverage = 0.95
                if self.conformal_predictor and self.conformal_predictor.q_hat is not None:
                    conformal_set = self.conformal_predictor.predict_set(t2_mean_probs)
                    conformal_coverage = 1.0 - self.conformal_predictor.alpha

                inference_time = (time.time() - start_time) * 1000.0

                # High risk review flags (high variance or ambiguous multiple classifications)
                flagged = (var_val > 0.1) or (conformal_set is not None and len(conformal_set) > 1)

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

            # CSV Telemetry Logging if requested
            if log_path:
                from datetime import datetime

                from core.utils.logging_utils import log_inference


                predicted_label_int = self.classes.index(prediction_result.prediction)
                correct_val = None
                if ground_truth is not None:
                    correct_val = int(ground_truth == predicted_label_int)

                c_set = prediction_result.conformal_set
                c_set_str = ", ".join(c_set) if isinstance(c_set, list) else str(c_set)

                record = {
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
                }
                log_inference(log_path, record)

            return prediction_result
