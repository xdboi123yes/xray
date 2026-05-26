"""Robust statistical testing module for chest X-ray classifiers.

Implements exact parametric and non-parametric comparisons including DeLong,
bootstrap confidence intervals, McNemar, and paired permutation tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats  # type: ignore[import-untyped]

from core.evaluation.metrics import compute_all_metrics


def delong_test(
    y_true: np.ndarray[Any, Any],
    y_probs1: np.ndarray[Any, Any],
    y_probs2: np.ndarray[Any, Any],
) -> float:
    """Computes the p-value comparing the AUC-ROC of two correlated classifiers.

    Based on the nonparametric approach by DeLong, DeLong, and Clarke-Pearson (1988).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs1: Predicted probabilities from the first model.
        y_probs2: Predicted probabilities from the second model.

    Returns:
        Two-tailed p-value for the difference in AUC-ROC.

    Raises:
        ValueError: If input arrays have mismatched shapes or do not contain both classes.
    """
    y_true = np.asarray(y_true)
    y_probs1 = np.asarray(y_probs1)
    y_probs2 = np.asarray(y_probs2)

    if len(y_true) != len(y_probs1) or len(y_true) != len(y_probs2):
        raise ValueError("Input arrays must have the same length.")

    unique_classes = np.unique(y_true)
    if len(unique_classes) != 2 or not np.array_equal(unique_classes, [0, 1]):
        raise ValueError("Ground truth labels must contain exactly both 0 and 1 classes.")

    # Split into positive and negative predictions
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos < 2 or n_neg < 2:
        raise ValueError("At least 2 positive and 2 negative samples are required.")

    pos1 = y_probs1[pos_mask]
    neg1 = y_probs1[neg_mask]
    pos2 = y_probs2[pos_mask]
    neg2 = y_probs2[neg_mask]

    # Compute structural components
    # We broadcast pos (n_pos, 1) and neg (1, n_neg) to get diff (n_pos, n_neg)
    diff1 = pos1[:, None] - neg1[None, :]
    v10_1 = np.mean((diff1 > 0).astype(float) + 0.5 * (diff1 == 0).astype(float), axis=1)
    v01_1 = np.mean((diff1 > 0).astype(float) + 0.5 * (diff1 == 0).astype(float), axis=0)

    diff2 = pos2[:, None] - neg2[None, :]
    v10_2 = np.mean((diff2 > 0).astype(float) + 0.5 * (diff2 == 0).astype(float), axis=1)
    v01_2 = np.mean((diff2 > 0).astype(float) + 0.5 * (diff2 == 0).astype(float), axis=0)

    auc1 = np.mean(v10_1)
    auc2 = np.mean(v10_2)

    # Compute covariance matrices of structural components
    # s_10 is positive structural component covariance (shape 2x2)
    # s_01 is negative structural component covariance (shape 2x2)
    s_10 = np.cov(np.vstack([v10_1, v10_2]), ddof=1)
    s_01 = np.cov(np.vstack([v01_1, v01_2]), ddof=1)

    # Variance of the difference:
    # Var(AUC1 - AUC2) = (s_10[0,0] + s_10[1,1] - 2*s_10[0,1]) / n_pos + (s_01[0,0] + s_01[1,1] - 2*s_01[0,1]) / n_neg
    var_diff = (
        (s_10[0, 0] + s_10[1, 1] - 2 * s_10[0, 1]) / n_pos
        + (s_01[0, 0] + s_01[1, 1] - 2 * s_01[0, 1]) / n_neg
    )

    if var_diff <= 0:
        return 1.0

    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
    return float(p_value)


def _calculate_metric(
    y_true: np.ndarray[Any, Any], y_probs: np.ndarray[Any, Any], metric: str
) -> float:
    """Helper function to calculate a classification metric from y_true and y_probs."""
    metrics = compute_all_metrics(y_true, y_probs)
    key = metric.lower()
    if key == "auc":
        key = "auc_roc"
    if key not in metrics:
        raise ValueError(
            f"Unsupported metric: {metric}. Choose from: auc, accuracy, sensitivity, specificity, f1, precision, mcc."
        )
    return float(metrics[key])


def bootstrap_ci(
    y_true: np.ndarray[Any, Any],
    y_probs1: np.ndarray[Any, Any],
    y_probs2: np.ndarray[Any, Any] | None = None,
    metric: str = "auc",
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Computes bootstrap confidence intervals for classification metrics.

    If y_probs2 is provided, also computes confidence interval for the difference
    (delta = model2 - model1) and includes the analytic DeLong p-value for AUC.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs1: Predicted probabilities from the first model.
        y_probs2: Optional predicted probabilities from the second model.
        metric: Metric name ('auc', 'accuracy', 'sensitivity', 'specificity', 'f1', 'precision', 'mcc').
        n_iterations: Number of bootstrap resamples.
        alpha: Significance level (e.g. 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        A dictionary containing empirical estimates, confidence intervals, delta, and p-value if applicable.
    """
    np.random.seed(seed)
    y_true = np.asarray(y_true)
    y_probs1 = np.asarray(y_probs1)
    n_samples = len(y_true)

    val1_base = _calculate_metric(y_true, y_probs1, metric)

    val1_list = []
    val2_list = []
    delta_list = []

    has_two = y_probs2 is not None
    if has_two:
        assert y_probs2 is not None
        y_probs2 = np.asarray(y_probs2)
        val2_base = _calculate_metric(y_true, y_probs2, metric)
        delta_base = val2_base - val1_base
    else:
        val2_base = None
        delta_base = None

    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_true = y_true[indices]
        boot_probs1 = y_probs1[indices]

        # Check if boot_true has both classes (necessary for AUC and others)
        if len(np.unique(boot_true)) != 2:
            continue

        try:
            val1 = _calculate_metric(boot_true, boot_probs1, metric)
            val1_list.append(val1)

            if has_two:
                assert y_probs2 is not None
                boot_probs2 = y_probs2[indices]
                val2 = _calculate_metric(boot_true, boot_probs2, metric)
                val2_list.append(val2)
                delta_list.append(val2 - val1)
        except Exception:
            # Handle possible divide-by-zero or numerical issues in small boot samples
            continue

    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    ci1 = (
        float(np.percentile(val1_list, lower_pct)),
        float(np.percentile(val1_list, upper_pct)),
    )

    results: dict[str, Any] = {
        "model1_metric": val1_base,
        "model1_ci": ci1,
    }

    if has_two:
        assert y_probs2 is not None
        ci2 = (
            float(np.percentile(val2_list, lower_pct)),
            float(np.percentile(val2_list, upper_pct)),
        )
        ci_delta = (
            float(np.percentile(delta_list, lower_pct)),
            float(np.percentile(delta_list, upper_pct)),
        )

        # Two-sided bootstrap p-value for the difference:
        # twice the proportion of bootstrap differences that cross zero
        delta_arr = np.array(delta_list)
        p_val_boot = float(2 * min(float(np.mean(delta_arr <= 0)), float(np.mean(delta_arr >= 0))))
        p_val_boot = max(p_val_boot, 1.0 / n_iterations)  # Avoid reporting exact 0.0

        results.update(
            {
                "model2_metric": val2_base,
                "model2_ci": ci2,
                "delta": delta_base,
                "delta_ci": ci_delta,
                "p_value_bootstrap": float(p_val_boot),
            }
        )

        if metric.lower() in ("auc", "auc_roc"):
            try:
                p_val_delong = delong_test(y_true, y_probs1, y_probs2)
                results["p_value_delong"] = p_val_delong
            except Exception:
                results["p_value_delong"] = float("nan")

    return results


def mcnemar_test(
    y_true: np.ndarray[Any, Any],
    y_probs1: np.ndarray[Any, Any],
    y_probs2: np.ndarray[Any, Any],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Computes McNemar's test for paired binary classifications.

    Uses a contingency table comparing matched-pair prediction correctness.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs1: Predicted probabilities from the first model.
        y_probs2: Predicted probabilities from the second model.
        threshold: Decision threshold to convert probabilities to binary predictions.

    Returns:
        A dictionary containing the contingency table, chi2 statistic, and p-value.
    """
    y_true = np.asarray(y_true)
    y_pred1 = (np.asarray(y_probs1) >= threshold).astype(int)
    y_pred2 = (np.asarray(y_probs2) >= threshold).astype(int)

    correct1 = y_pred1 == y_true
    correct2 = y_pred2 == y_true

    # Contingency table:
    #                 Model 2 Correct    Model 2 Incorrect
    # Model 1 Correct       a                   b
    # Model 1 Incorrect     c                   d
    a = np.sum(correct1 & correct2)
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    d = np.sum(~correct1 & ~correct2)

    # McNemar's test statistic with continuity correction: (|b - c| - 1)^2 / (b + c)
    denominator = b + c
    if denominator == 0:
        stat = 0.0
        p_value = 1.0
    else:
        stat = ((np.abs(b - c) - 1.0) ** 2) / denominator
        p_value = scipy.stats.chi2.sf(stat, df=1)

    return {
        "contingency_table": {"a": int(a), "b": int(b), "c": int(c), "d": int(d)},
        "statistic": float(stat),
        "p_value": float(p_value),
    }


def permutation_test(
    y_true: np.ndarray[Any, Any],
    y_probs1: np.ndarray[Any, Any],
    y_probs2: np.ndarray[Any, Any],
    metric: str = "auc",
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Computes a paired permutation test to compare two models.

    Under the null hypothesis, the predictions of Model 1 and Model 2 are interchangeable.
    For each permutation, we randomly swap the predictions of Model 1 and Model 2 for a subset of samples.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs1: Predicted probabilities from the first model.
        y_probs2: Predicted probabilities from the second model.
        metric: Metric name to compare.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary with the observed difference, p-value, and permuted differences.
    """
    np.random.seed(seed)
    y_true = np.asarray(y_true)
    y_probs1 = np.asarray(y_probs1)
    y_probs2 = np.asarray(y_probs2)

    n_samples = len(y_true)
    obs1 = _calculate_metric(y_true, y_probs1, metric)
    obs2 = _calculate_metric(y_true, y_probs2, metric)
    observed_diff = obs2 - obs1

    count_extreme = 0
    perm_diffs = []

    for _ in range(n_permutations):
        # Generate random boolean mask to swap (True means swap, False means keep)
        swap_mask = np.random.rand(n_samples) < 0.5

        perm_probs1 = np.where(swap_mask, y_probs2, y_probs1)
        perm_probs2 = np.where(swap_mask, y_probs1, y_probs2)

        try:
            p1 = _calculate_metric(y_true, perm_probs1, metric)
            p2 = _calculate_metric(y_true, perm_probs2, metric)
            diff = p2 - p1
            perm_diffs.append(diff)

            if np.abs(diff) >= np.abs(observed_diff):
                count_extreme += 1
        except Exception:
            continue

    p_value = count_extreme / len(perm_diffs) if perm_diffs else 1.0

    return {
        "observed_difference": observed_diff,
        "p_value": float(p_value),
        "permuted_differences": perm_diffs,
    }
