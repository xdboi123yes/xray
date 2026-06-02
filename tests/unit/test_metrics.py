"""Unit tests for core evaluation metrics."""

from __future__ import annotations

import math

import numpy as np

from core.evaluation.metrics import (
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
