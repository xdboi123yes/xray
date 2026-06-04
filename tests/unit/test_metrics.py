"""Unit tests for core evaluation metrics."""

from __future__ import annotations

import math

import numpy as np

from core.evaluation.metrics import (
    brier_score,
    calibration_slope_intercept,
    compute_all_metrics,
    expected_calibration_error,
    find_optimal_threshold,
)


def test_compute_all_metrics_basic() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_all_metrics(y_true, y_probs)
    assert m["accuracy"] == 1.0
    assert m["auc_roc"] == 1.0


def test_compute_all_metrics_single_class_gives_nan_auc() -> None:
    y_true = np.array([0, 0, 0])
    y_probs = np.array([0.1, 0.2, 0.3])
    m = compute_all_metrics(y_true, y_probs)
    assert math.isnan(m["auc_roc"])


def test_expected_calibration_error_perfectly_calibrated_is_low() -> None:
    # Confident and correct predictions: confidence ~1.0, accuracy 1.0 -> ECE ~0.
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.0, 0.0, 1.0, 1.0])
    ece = expected_calibration_error(y_true, y_probs)
    assert 0.0 <= ece < 1e-9


def test_expected_calibration_error_confidently_wrong_is_high() -> None:
    # Maximally confident but always wrong: confidence 1.0, accuracy 0.0 -> ECE 1.0.
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([1.0, 1.0, 0.0, 0.0])
    ece = expected_calibration_error(y_true, y_probs)
    assert math.isclose(ece, 1.0, abs_tol=1e-9)


def test_expected_calibration_error_empty_is_nan() -> None:
    assert math.isnan(expected_calibration_error([], []))


def test_brier_score_perfect_is_zero() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.0, 0.0, 1.0, 1.0])
    assert math.isclose(brier_score(y_true, y_probs), 0.0, abs_tol=1e-12)


def test_brier_score_worst_is_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([1.0, 1.0, 0.0, 0.0])
    assert math.isclose(brier_score(y_true, y_probs), 1.0, abs_tol=1e-12)


def test_brier_score_empty_is_nan() -> None:
    assert math.isnan(brier_score([], []))


def test_calibration_slope_intercept_well_calibrated() -> None:
    # Draw labels from their own predicted probability -> calibrated by design,
    # so the recalibration slope ~ 1 and intercept ~ 0.
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert math.isclose(slope, 1.0, abs_tol=0.15)
    assert math.isclose(intercept, 0.0, abs_tol=0.15)


def test_calibration_slope_intercept_overconfident_slope_below_one() -> None:
    # Sharpen calibrated probabilities away from 0.5 -> over-confident -> slope < 1.
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    logit = np.log(p / (1.0 - p))
    p_overconfident = 1.0 / (1.0 + np.exp(-2.0 * logit))
    slope, _ = calibration_slope_intercept(y, p_overconfident)
    assert slope < 0.85


def test_calibration_slope_intercept_single_class_is_nan() -> None:
    slope, intercept = calibration_slope_intercept([1, 1, 1], [0.6, 0.7, 0.8])
    assert math.isnan(slope) and math.isnan(intercept)


def test_find_optimal_threshold_returns_valid_range() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    t = find_optimal_threshold(y_true, y_probs)
    assert 0.1 <= t <= 0.91


def test_find_optimal_threshold_low_sensitivity_logs_warning() -> None:
    y_true = np.array([0, 0, 0, 1])
    y_probs = np.array([0.9, 0.9, 0.9, 0.05])
    t = find_optimal_threshold(y_true, y_probs)
    assert isinstance(t, float)
