"""Unit tests for Chest X-Ray application services.

Tests ECE computation, model calibration, and statistical hypothesis testing.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from application.dto.metrics_dto import (
    ClassificationMetricsDTO,
    ConfidenceIntervalDTO,
    StatisticalTestResultDTO,
)
from application.services.calibration_service import CalibrationService
from application.services.evaluation_service import EvaluationService


@pytest.fixture
def dummy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates dummy ground-truth labels and prediction probabilities for two models."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_probs1 = np.array([0.1, 0.2, 0.4, 0.3, 0.7, 0.8, 0.6, 0.9])
    y_probs2 = np.array([0.15, 0.1, 0.3, 0.25, 0.8, 0.9, 0.75, 0.85])
    return y_true, y_probs1, y_probs2


def test_calibration_service_ece(dummy_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Verifies that ECE is calculated correctly by CalibrationService."""
    y_true, y_probs1, _ = dummy_data
    service = CalibrationService()

    ece = service.calculate_ece(y_probs1, y_true, n_bins=5)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_calibration_service_plot(
    dummy_data: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Any
) -> None:
    """Tests that the reliability diagram is plotted and saved to the correct path."""
    y_true, y_probs1, _ = dummy_data
    service = CalibrationService()

    # Override figures dir for testing isolation
    service.figures_dir = str(tmp_path)
    filename = "test_reliability.png"

    saved_path = service.save_reliability_diagram(y_probs1, y_true, filename=filename, n_bins=5)
    assert os.path.exists(saved_path)
    assert saved_path == os.path.abspath(os.path.join(tmp_path, filename))


def test_calibration_service_fit() -> None:
    """Tests that calibrate_model correctly optimizes logit scaling temperature."""
    service = CalibrationService()

    # Create dummy model returning simple logits
    class SimpleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)
            # Fix weights for deterministic logit outputs
            self.linear.weight.data.fill_(1.0)
            self.linear.bias.data.fill_(0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = SimpleModel()

    # Create dummy dataloader
    x_val = torch.randn(10, 2)
    y_val = torch.randint(0, 2, (10,))
    dataset = TensorDataset(x_val, y_val, torch.zeros(10))  # Yields image_tensor, label, index
    loader = DataLoader(dataset, batch_size=2)

    device = torch.device("cpu")
    opt_temp = service.calibrate_model(model, loader, device=device)

    assert isinstance(opt_temp, float)
    assert opt_temp > 0.0


def test_evaluation_service_metrics(dummy_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Verifies that compute_metrics returns a fully populated metrics DTO."""
    y_true, y_probs1, _ = dummy_data
    service = EvaluationService()

    dto = service.compute_metrics(y_true, y_probs1, n_bins=5)
    assert isinstance(dto, ClassificationMetricsDTO)
    assert 0.0 <= dto.auc_roc <= 1.0
    assert 0.0 <= dto.accuracy <= 1.0
    assert 0.0 <= dto.ece <= 1.0


def test_evaluation_service_compare(dummy_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Tests comparison testing between two models."""
    y_true, y_probs1, y_probs2 = dummy_data
    service = EvaluationService()

    test_results = service.compare_models(y_true, y_probs1, y_probs2, threshold=0.5)
    assert isinstance(test_results, list)
    assert len(test_results) > 0

    for res in test_results:
        assert isinstance(res, StatisticalTestResultDTO)
        assert res.p_value >= 0.0
        assert len(res.test_name) > 0


def test_evaluation_service_bootstrap(
    dummy_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Tests bootstrap confidence interval generation."""
    y_true, y_probs1, y_probs2 = dummy_data
    service = EvaluationService()

    # Test single-model CI
    dto_single = service.get_bootstrap_confidence_interval(
        y_true, y_probs1, metric="auc", n_iterations=50
    )
    assert isinstance(dto_single, ConfidenceIntervalDTO)
    assert dto_single.metric == "AUC"
    assert dto_single.lower_bound <= dto_single.point_estimate <= dto_single.upper_bound

    # Test comparison CI (delta)
    dto_delta = service.get_bootstrap_confidence_interval(
        y_true, y_probs1, y_probs2=y_probs2, metric="auc", n_iterations=50
    )
    assert isinstance(dto_delta, ConfidenceIntervalDTO)
    assert dto_delta.metric == "Delta AUC"
    assert dto_delta.lower_bound <= dto_delta.point_estimate <= dto_delta.upper_bound


def test_evaluation_service_plots(
    dummy_data: tuple[np.ndarray, np.ndarray, np.ndarray], tmp_path: Any
) -> None:
    """Tests plotting of comparative ROC and DCA plots."""
    y_true, y_probs1, y_probs2 = dummy_data
    service = EvaluationService()
    service.figures_dir = str(tmp_path)

    # Test ROC plot comparison
    roc_path = service.save_roc_comparison(
        y_true, {"Model 1": y_probs1, "Model 2": y_probs2}, filename="test_roc.png"
    )
    assert os.path.exists(roc_path)

    # Test Decision Curve plot
    dca_path = service.save_decision_curve(y_true, y_probs1, filename="test_dca.png")
    assert os.path.exists(dca_path)
