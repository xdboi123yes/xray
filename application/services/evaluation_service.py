"""Application Service orchestrating clinical evaluations and statistical comparisons.

Calculates binary metrics DTOs, performs paired DeLong/McNemar/permutation
significance testing, and exports ROC/decision/calibration visual plots.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import structlog
from sklearn.metrics import roc_curve

from application.dto.metrics_dto import (
    ClassificationMetricsDTO,
    ConfidenceIntervalDTO,
    StatisticalTestResultDTO,
)
from application.services.calibration_service import CalibrationService
from config.settings import get_settings
from core.evaluation.metrics import compute_all_metrics
from core.evaluation.stats import bootstrap_ci, delong_test, mcnemar_test, permutation_test

log = structlog.get_logger(__name__)


class EvaluationService:
    """Orchestrates comprehensive clinical classification and comparative statistical evaluations."""

    def __init__(self) -> None:
        """Initializes the EvaluationService and ensures output directories exist."""
        self.settings = get_settings()
        self.figures_dir = self.settings.paths.figures
        os.makedirs(self.figures_dir, exist_ok=True)
        self.calibration_service = CalibrationService()

    def compute_metrics(
        self,
        y_true: Sequence[int] | np.ndarray[Any, Any],
        y_probs: Sequence[float] | np.ndarray[Any, Any],
        n_bins: int = 10,
    ) -> ClassificationMetricsDTO:
        """Computes all standard binary classification and calibration metrics.

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_probs: Predicted probabilities (0.0 to 1.0).
            n_bins: Confidence bin count for ECE calculation.

        Returns:
            A populated ClassificationMetricsDTO object.
        """
        true_np = np.asarray(y_true)
        probs_np = np.asarray(y_probs)

        # 1. Compute core metrics from core/evaluation/metrics.py
        metrics = compute_all_metrics(true_np, probs_np)

        # 2. Compute Expected Calibration Error
        ece = self.calibration_service.calculate_ece(probs_np, true_np, n_bins=n_bins)

        return ClassificationMetricsDTO(
            auc_roc=float(metrics["auc_roc"]),
            accuracy=float(metrics["accuracy"]),
            sensitivity=float(metrics["sensitivity"]),
            specificity=float(metrics["specificity"]),
            precision=float(metrics["precision"]),
            f1=float(metrics["f1"]),
            mcc=float(metrics["mcc"]),
            ece=ece,
        )

    def compare_models(
        self,
        y_true: Sequence[int] | np.ndarray[Any, Any],
        y_probs1: Sequence[float] | np.ndarray[Any, Any],
        y_probs2: Sequence[float] | np.ndarray[Any, Any],
        threshold: float = 0.75,
    ) -> list[StatisticalTestResultDTO]:
        """Runs hypothesis tests comparing the performance of two models.

        Executes DeLong test (for AUC ROC), McNemar's test (for correctness paired shifts),
        and paired Permutation test (for metric swaps).

        Args:
            y_true: Ground truth binary labels.
            y_probs1: Model 1 prediction probabilities.
            y_probs2: Model 2 prediction probabilities.
            threshold: Decision routing threshold for McNemar contingency.

        Returns:
            A list of StatisticalTestResultDTO containing p-values and interpretations.
        """
        true_np = np.asarray(y_true)
        probs1_np = np.asarray(y_probs1)
        probs2_np = np.asarray(y_probs2)

        results: list[StatisticalTestResultDTO] = []

        # 1. Non-parametric DeLong's test comparing AUC ROCs
        try:
            p_delong = delong_test(true_np, probs1_np, probs2_np)
            diff_auc = float(np.mean(probs2_np[true_np == 1]) - np.mean(probs1_np[true_np == 1])) # Basic delta
            # Retrieve exact AUC values for proper delta comparison
            m1 = compute_all_metrics(true_np, probs1_np)
            m2 = compute_all_metrics(true_np, probs2_np)
            diff_auc = float(m2["auc_roc"] - m1["auc_roc"])

            interpretation = (
                "Statistically significant difference in AUC-ROC."
                if p_delong < 0.05
                else "No statistically significant difference in AUC-ROC."
            )
            results.append(
                StatisticalTestResultDTO(
                    test_name="DeLong AUC Comparison",
                    observed_difference=diff_auc,
                    statistic=None,
                    p_value=p_delong,
                    interpretation=interpretation,
                )
            )
        except Exception as ex:
            log.error(f"[EvaluationService] Warning: DeLong's test failed: {ex}")

        # 2. McNemar's test for paired classification correctness shifts
        try:
            mcnemar_res = mcnemar_test(true_np, probs1_np, probs2_np, threshold=threshold)
            p_mcnemar = float(mcnemar_res["p_value"])
            stat_mcnemar = float(mcnemar_res["statistic"])

            # Compute accuracy difference
            acc1 = float(np.mean((probs1_np >= threshold) == true_np))
            acc2 = float(np.mean((probs2_np >= threshold) == true_np))
            diff_acc = acc2 - acc1

            interpretation = (
                "Significant paired proportion shift in classification correctness."
                if p_mcnemar < 0.05
                else "No significant paired shift in correctness."
            )
            results.append(
                StatisticalTestResultDTO(
                    test_name="McNemar Contingency Check",
                    observed_difference=diff_acc,
                    statistic=stat_mcnemar,
                    p_value=p_mcnemar,
                    interpretation=interpretation,
                )
            )
        except Exception as ex:
            log.error(f"[EvaluationService] Warning: McNemar's test failed: {ex}")

        # 3. Paired Permutation test (paired swaps)
        try:
            perm_res = permutation_test(
                true_np, probs1_np, probs2_np, metric="auc", n_permutations=200
            )
            p_perm = float(perm_res["p_value"])
            obs_diff = float(perm_res["observed_difference"])

            interpretation = (
                "Permutation test confirms statistically significant metric divergence."
                if p_perm < 0.05
                else "Interchangeability hypothesis stands (not significant)."
            )
            results.append(
                StatisticalTestResultDTO(
                    test_name="Paired Permutation (AUC Swap)",
                    observed_difference=obs_diff,
                    statistic=None,
                    p_value=p_perm,
                    interpretation=interpretation,
                )
            )
        except Exception as ex:
            log.error(f"[EvaluationService] Warning: Permutation test failed: {ex}")

        return results

    def get_bootstrap_confidence_interval(
        self,
        y_true: Sequence[int] | np.ndarray[Any, Any],
        y_probs1: Sequence[float] | np.ndarray[Any, Any],
        y_probs2: Sequence[float] | np.ndarray[Any, Any] | None = None,
        metric: str = "auc",
        n_iterations: int = 200,
        alpha: float = 0.05,
    ) -> ConfidenceIntervalDTO:
        """Computes bootstrap empirical confidence interval boundaries.

        Args:
            y_true: Ground truth binary labels.
            y_probs1: Primary model prediction probabilities.
            y_probs2: Secondary comparison model prediction probabilities.
            metric: Metric name.
            n_iterations: Resampling epochs count.
            alpha: Significance level.

        Returns:
            A populated ConfidenceIntervalDTO.
        """
        true_np = np.asarray(y_true)
        probs1_np = np.asarray(y_probs1)
        probs2_np = np.asarray(y_probs2) if y_probs2 is not None else None

        res = bootstrap_ci(
            true_np,
            probs1_np,
            y_probs2=probs2_np,
            metric=metric,
            n_iterations=n_iterations,
            alpha=alpha,
        )

        # If comparing two models, return delta CI details; else return model 1 CI
        if y_probs2 is not None:
            point_est = float(res.get("delta", 0.0))
            lower, upper = res.get("delta_ci", (0.0, 0.0))
            metric_name = f"Delta {metric.upper()}"
        else:
            point_est = float(res.get("model1_metric", 0.0))
            lower, upper = res.get("model1_ci", (0.0, 0.0))
            metric_name = metric.upper()

        return ConfidenceIntervalDTO(
            metric=metric_name,
            point_estimate=point_est,
            lower_bound=float(lower),
            upper_bound=float(upper),
            alpha=alpha,
        )

    def save_roc_comparison(
        self,
        y_true: Sequence[int] | np.ndarray[Any, Any],
        models_probs: dict[str, Sequence[float] | np.ndarray[Any, Any]],
        filename: str = "roc_comparison.png",
    ) -> str:
        """Plots and saves comparison ROC curves for multiple models.

        Args:
            y_true: Ground truth binary labels.
            models_probs: Dictionary of {model_name: probabilities}.
            filename: Target file name to save under figures folder.

        Returns:
            Absolute file path of the saved figure.
        """
        true_np = np.asarray(y_true)

        plt.figure(figsize=(7, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (0.50)")

        for name, probs in models_probs.items():
            probs_np = np.asarray(probs)
            fpr, tpr, _ = roc_curve(true_np, probs_np)
            metrics = compute_all_metrics(true_np, probs_np)
            auc_val = float(metrics["auc_roc"])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Clinical Receiver Operating Characteristic (ROC) Comparison")
        plt.legend(loc="lower right")
        plt.grid(True)

        save_path = os.path.join(self.figures_dir, filename)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        log.info(f"[EvaluationService] ROC Comparison saved successfully at {save_path}")
        return os.path.abspath(save_path)

    def save_decision_curve(
        self,
        y_true: Sequence[int] | np.ndarray[Any, Any],
        y_probs: Sequence[float] | np.ndarray[Any, Any],
        model_name: str = "TieredSystem",
        filename: str = "decision_curve.png",
    ) -> str:
        """Plots and saves Decision Curve Analysis (DCA) representing clinical utility.

        Plots Net Benefit across decision threshold ranges.

        Args:
            y_true: Ground truth binary labels.
            y_probs: Model prediction probabilities.
            model_name: Label representing the model.
            filename: Output filename under figures folder.

        Returns:
            Absolute file path of the saved plot.
        """
        true_np = np.asarray(y_true)
        probs_np = np.asarray(y_probs)

        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefit_model = []
        net_benefit_all = []

        n_samples = len(true_np)
        prevalence = float(np.mean(true_np))

        for pt in thresholds:
            # Model Net Benefit = (TP / N) - (FP / N) * (pt / (1 - pt))
            tp = np.sum((probs_np >= pt) & (true_np == 1))
            fp = np.sum((probs_np >= pt) & (true_np == 0))
            nb_model = (tp / n_samples) - (fp / n_samples) * (pt / (1 - pt))
            net_benefit_model.append(nb_model)

            # "Treat All" Net Benefit
            tp_all = np.sum(true_np == 1)
            fp_all = np.sum(true_np == 0)
            nb_all = (tp_all / n_samples) - (fp_all / n_samples) * (pt / (1 - pt))
            net_benefit_all.append(nb_all)

        plt.figure(figsize=(7, 6))
        plt.plot(thresholds, net_benefit_model, color="blue", label=f"Model ({model_name})")
        plt.plot(thresholds, net_benefit_all, color="gray", linestyle="--", label="Treat All")
        plt.plot(
            thresholds, [0.0] * len(thresholds), color="black", linestyle="-", label="Treat None"
        )

        plt.xlim(0.0, 1.0)
        plt.ylim(-0.05, prevalence + 0.05)
        plt.xlabel("Probability Threshold")
        plt.ylabel("Net Benefit")
        plt.title("Decision Curve Analysis (DCA) for Clinical Utility")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.figures_dir, filename)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()

        log.info(f"[EvaluationService] Decision Curve Analysis saved successfully at {save_path}")
        return os.path.abspath(save_path)
