"""Classification metrics calculation module for chest X-ray classifiers."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import structlog
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

log = structlog.get_logger(__name__)


def compute_all_metrics(y_true: Any, y_probs: Any, threshold: float = 0.5) -> dict[str, float]:
    """Computes basic classification metrics for binary classification.

    Args:
        y_true: Ground truth binary labels.
        y_probs: Predicted probabilities for the positive class.
        threshold: Decision threshold for converting probabilities to binary predictions.

    Returns:
        A dictionary containing accuracy, auc_roc, sensitivity, specificity, f1, precision, mcc.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    # Calculate AUC-ROC safely
    try:
        auc_roc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        auc_roc = float("nan")

    # Extract confusion matrix values safely
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Calculate Sensitivity (Recall) and Specificity
    sensitivity = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    specificity = float(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc": auc_roc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def expected_calibration_error(y_true: Any, y_probs: Any, n_bins: int = 15) -> float:
    """Compute the Expected Calibration Error (ECE) for binary predictions.

    Bins predictions by the confidence of the predicted class (max(p, 1 - p)),
    then accumulates the gap between bin accuracy and bin confidence weighted by
    bin population, following Guo et al. (2017).

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_probs: Predicted probabilities for the positive class.
        n_bins: Number of equal-width confidence bins.

    Returns:
        The ECE as a float in [0, 1], or NaN when the input is empty.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_probs_arr = np.asarray(y_probs, dtype=float)
    n = y_true_arr.size
    if n == 0:
        return float("nan")

    predictions = (y_probs_arr >= 0.5).astype(float)
    confidences = np.maximum(y_probs_arr, 1.0 - y_probs_arr)
    accuracies = (predictions == y_true_arr).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in pairwise(bin_edges):
        in_bin = (confidences > low) & (confidences <= high)
        bin_count = int(in_bin.sum())
        if bin_count == 0:
            continue
        bin_accuracy = float(accuracies[in_bin].mean())
        bin_confidence = float(confidences[in_bin].mean())
        ece += (bin_count / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def brier_score(y_true: Any, y_probs: Any) -> float:
    """Compute the Brier score for binary predictions.

    The Brier score is the mean squared error between the predicted
    positive-class probability and the binary outcome; lower is better, with a
    perfectly accurate and confident classifier scoring 0.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_probs: Predicted probabilities for the positive class.

    Returns:
        The Brier score as a float in [0, 1], or NaN when the input is empty.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_probs_arr = np.asarray(y_probs, dtype=float)
    if y_true_arr.size == 0:
        return float("nan")
    return float(np.mean((y_probs_arr - y_true_arr) ** 2))


def negative_log_likelihood(y_true: Any, y_probs: Any, eps: float = 1e-12) -> float:
    """Compute binary negative log-likelihood (log loss) for positive-class probabilities."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_probs, dtype=float), eps, 1.0 - eps)
    if y.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def classwise_ece(y_true: Any, y_probs: Any, n_bins: int = 15) -> dict[str, float]:
    """Compute class-wise ECE (equal-width bins) for binary classification.

    Evaluates calibration separately for class 0 (Normal, predicted probability
    ``1 - p``) and class 1 (Pneumothorax, predicted probability ``p``), plus their
    mean (class-wise ECE).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_probs, dtype=float)
    n = y.size
    if n == 0:
        return {"ece_class0": float("nan"), "ece_class1": float("nan"), "classwise_ece": float("nan")}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _one_class_ece(target_indicator: np.ndarray[Any, Any], class_prob: np.ndarray[Any, Any]) -> float:
        ece_val = 0.0
        for idx, (low, high) in enumerate(pairwise(bin_edges)):
            in_bin = (class_prob >= low) & (class_prob <= high) if idx == 0 else (class_prob > low) & (class_prob <= high)
            cnt = int(in_bin.sum())
            if cnt == 0:
                continue
            freq = float(target_indicator[in_bin].mean())
            conf = float(class_prob[in_bin].mean())
            ece_val += (cnt / n) * abs(freq - conf)
        return float(ece_val)

    ece0 = _one_class_ece((y == 0).astype(float), 1.0 - p)
    ece1 = _one_class_ece((y == 1).astype(float), p)
    return {
        "ece_class0": ece0,
        "ece_class1": ece1,
        "classwise_ece": 0.5 * (ece0 + ece1),
    }


def reliability_bin_stats(y_true: Any, y_probs: Any, n_bins: int = 15) -> list[dict[str, float | int]]:
    """Return per-bin statistics across ``n_bins`` equal-width intervals on [0, 1]."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_probs, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for idx, (low, high) in enumerate(pairwise(bin_edges)):
        in_bin = (p >= low) & (p <= high) if idx == 0 else (p > low) & (p <= high)
        cnt = int(in_bin.sum())
        rows.append(
            {
                "bin_index": idx,
                "bin_lower": float(low),
                "bin_upper": float(high),
                "count": cnt,
                "mean_prob": float(p[in_bin].mean()) if cnt > 0 else float("nan"),
                "empirical_pos_rate": float(y[in_bin].mean()) if cnt > 0 else float("nan"),
            }
        )
    return rows


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute the two-sided Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    half = (z * np.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * total)) / total)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def youden_threshold(y_true: Any, y_probs: Any) -> float:
    """Return the decision threshold maximising Youden's J (TPR - FPR); highest cut on ties."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(np.asarray(y_true, dtype=int), np.asarray(y_probs, dtype=float))
    finite = np.isfinite(thresholds)
    j = np.where(finite, tpr - fpr, -np.inf)
    return float(thresholds[int(np.argmax(j))])


def operating_point(y_true: Any, y_probs: Any, threshold: float) -> dict[str, Any]:
    """Confusion counts, sensitivity/specificity/FNR/accuracy, and 95% Wilson CIs at ``threshold``."""
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(y_probs, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_pos, n_neg = tp + fn, tn + fp
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": tp / n_pos if n_pos else float("nan"),
        "specificity": tn / n_neg if n_neg else float("nan"),
        "fnr": fn / n_pos if n_pos else float("nan"),
        "accuracy": (tp + tn) / max(1, len(y)),
        "sensitivity_wilson_95ci": list(wilson_ci(tp, n_pos)),
        "specificity_wilson_95ci": list(wilson_ci(tn, n_neg)),
        "accuracy_wilson_95ci": list(wilson_ci(tp + tn, len(y))),
    }


def calibration_slope_intercept(
    y_true: Any,
    y_probs: Any,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Estimate the calibration slope and intercept (calibration-in-the-large).

    Fits the logistic recalibration model
    ``logit(P(y=1)) = intercept + slope * logit(p_hat)`` by Newton-Raphson
    (iteratively reweighted least squares). A perfectly calibrated model yields
    a slope of 1 and an intercept of 0. A slope below 1 indicates
    over-confident predictions (probabilities too extreme); a non-zero
    intercept indicates a systematic over- or under-estimation of risk.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_probs: Predicted probabilities for the positive class.
        max_iter: Maximum number of Newton-Raphson iterations.
        tol: Convergence tolerance on the parameter update.

    Returns:
        A ``(slope, intercept)`` tuple, or ``(nan, nan)`` when the input is
        empty, single-class, or the fit fails to converge.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_probs, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan"), float("nan")

    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))
    design = np.column_stack([np.ones_like(logits), logits])  # [intercept, slope]
    beta = np.zeros(2)

    for _ in range(max_iter):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(mu * (1.0 - mu), eps, None)
        gradient = design.T @ (y - mu)
        hessian = design.T @ (design * weights[:, None])
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta += delta
        if float(np.max(np.abs(delta))) < tol:
            break

    intercept, slope = float(beta[0]), float(beta[1])
    return slope, intercept


def find_optimal_threshold(y_true: Any, y_probs: Any) -> float:
    """Finds the optimal decision threshold that maximizes the F1 score.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_probs: Predicted probabilities.

    Returns:
        The optimal F1 threshold value.
    """
    best_threshold = 0.5
    best_f1 = -1.0
    best_sensitivity = 0.0

    # Sweep thresholds from 0.1 to 0.9 with a step of 0.01
    for threshold in np.arange(0.1, 0.91, 0.01):
        metrics = compute_all_metrics(y_true, y_probs, threshold=float(threshold))
        f1 = metrics["f1"]

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_sensitivity = metrics["sensitivity"]

    if best_sensitivity < 0.7:
        log.warning(
            f"WARNING: Sensitivity at optimal threshold ({best_threshold:.2f}) "
            f"is low: {best_sensitivity:.2f} (< 0.7)"
        )

    return best_threshold
