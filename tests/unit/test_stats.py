"""Unit tests for the statistical testing module.

Verifies mathematical soundness, error handling, and output formats.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.evaluation.stats import bootstrap_ci, delong_test, mcnemar_test, permutation_test


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates standard toy binary targets and model predictions."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Model 1: decent accuracy (~0.8 AUC)
    y_probs1 = np.array([0.1, 0.2, 0.4, 0.3, 0.6, 0.7, 0.5, 0.8, 0.9, 0.9])
    # Model 2: slightly better (~0.9 AUC)
    y_probs2 = np.array([0.05, 0.1, 0.2, 0.3, 0.15, 0.8, 0.75, 0.9, 0.95, 0.85])
    return y_true, y_probs1, y_probs2


def test_delong_test_valid(
    sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Verifies that delong_test returns a valid p-value for distinct models."""
    y_true, y_probs1, y_probs2 = sample_data
    p_value = delong_test(y_true, y_probs1, y_probs2)
    assert isinstance(p_value, float)
    assert 0.0 <= p_value <= 1.0


def test_delong_test_identical(
    sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Verifies that delong_test returns 1.0 when comparing a model to itself."""
    y_true, y_probs1, _ = sample_data
    p_value = delong_test(y_true, y_probs1, y_probs1)
    assert p_value == 1.0


def test_delong_test_invalid_inputs() -> None:
    """Verifies error cases for mismatched lengths or missing classes."""
    y_true = np.array([0, 1, 0])
    y_probs1 = np.array([0.2, 0.8])
    y_probs2 = np.array([0.3, 0.7, 0.5])

    # Mismatched length
    with pytest.raises(ValueError, match="Input arrays must have the same length"):
        delong_test(y_true, y_probs1, y_probs2)

    # Missing class
    y_true_single = np.array([1, 1, 1])
    y_probs_a = np.array([0.7, 0.8, 0.9])
    y_probs_b = np.array([0.6, 0.7, 0.8])
    with pytest.raises(
        ValueError, match="Ground truth labels must contain exactly both 0 and 1 classes"
    ):
        delong_test(y_true_single, y_probs_a, y_probs_b)


def test_bootstrap_ci_single_model(
    sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Verifies bootstrap CI outputs for a single model."""
    y_true, y_probs1, _ = sample_data
    results = bootstrap_ci(
        y_true, y_probs1, metric="auc", n_iterations=50, seed=42
    )

    assert "model1_metric" in results
    assert "model1_ci" in results
    assert len(results["model1_ci"]) == 2
    assert results["model1_ci"][0] <= results["model1_ci"][1]
    assert "model2_metric" not in results


def test_bootstrap_ci_paired_models(
    sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Verifies bootstrap CI outputs comparing two models."""
    y_true, y_probs1, y_probs2 = sample_data
    results = bootstrap_ci(
        y_true,
        y_probs1,
        y_probs2,
        metric="accuracy",
        n_iterations=50,
        seed=42,
    )

    assert "model1_metric" in results
    assert "model2_metric" in results
    assert "delta" in results
    assert "delta_ci" in results
    assert "p_value_bootstrap" in results
    assert results["delta_ci"][0] <= results["delta_ci"][1]


def test_mcnemar_test(
    sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Verifies McNemar test contingency table and output logic."""
    y_true, y_probs1, y_probs2 = sample_data
    results = mcnemar_test(y_true, y_probs1, y_probs2, threshold=0.5)

    assert "contingency_table" in results
    assert "statistic" in results
    assert "p_value" in results

    table = results["contingency_table"]
    assert "a" in table
    assert "b" in table
    assert "c" in table
    assert "d" in table
    assert sum(table.values()) == len(y_true)


def test_permutation_test(
    sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Verifies paired permutation test calculations."""
    y_true, y_probs1, y_probs2 = sample_data
    results = permutation_test(
        y_true, y_probs1, y_probs2, metric="auc", n_permutations=50, seed=42
    )

    assert "observed_difference" in results
    assert "p_value" in results
    assert "permuted_differences" in results
    assert len(results["permuted_differences"]) == 50
    assert 0.0 <= results["p_value"] <= 1.0
