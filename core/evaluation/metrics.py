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
