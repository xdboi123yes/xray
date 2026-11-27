"""Unit tests for TieredSystem cascade routing, Escalation, and Human Deferral."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from tcrp.models import TieredPrediction, TieredSystem
from tcrp.routing import DynamicThresholdRouter


def test_tiered_prediction_dataclass() -> None:
    pred = TieredPrediction(
        prediction="No Finding",
        confidence=0.92,
        mc_variance=None,
        tier_used=1,
    )
    assert pred.prediction == "No Finding"
    assert pred.confidence == 0.92
    assert pred.mc_variance is None
    assert pred.tier_used == 1
    assert pred.inference_time_ms == 0.0
    assert pred.conformal_set is None
    assert pred.conformal_coverage == 0.95
    assert pred.tta_passes == 10
    assert pred.mc_passes == 20
    assert not pred.flagged_for_review


def test_tiered_system_initialization() -> None:
    config = {
        "model": {
            "confidence_threshold": 0.75,
            "threshold_window_size": 5,
            "threshold_delta": 0.05,
            "mc_dropout_passes": 20,
            "tta_passes": 10,
        }
    }
    system = TieredSystem(MagicMock(), MagicMock(), config)
    assert system.current_threshold == 0.75
    assert isinstance(system.router, DynamicThresholdRouter)


def test_tiered_system_update_threshold() -> None:
    config = {
        "model": {
            "confidence_threshold": 0.75,
            "threshold_window_size": 3,
            "threshold_delta": 0.05,
        }
    }
    system_low = TieredSystem(MagicMock(), MagicMock(), config)
    for conf in (0.50, 0.55, 0.60, 0.58):
        system_low.update_threshold(conf)
    assert system_low.current_threshold == pytest.approx(0.70)

    system_high = TieredSystem(MagicMock(), MagicMock(), config)
    for conf in (0.90, 0.92, 0.95, 0.89):
        system_high.update_threshold(conf)
    assert system_high.current_threshold == pytest.approx(0.80)


def test_tiered_system_route_tier1() -> None:
    t1 = MagicMock()
    t1.return_value = torch.tensor([[0.1, 0.9]])
    t1.get_confidence.return_value = (torch.tensor(0.85), torch.tensor(1))

    config = {
        "model": {
            "confidence_threshold": 0.75,
            "threshold_window_size": 3,
            "threshold_delta": 0.05,
        }
    }
    system = TieredSystem(t1, MagicMock(), config)
    prediction = system.route(torch.zeros(1, 3, 224, 224))
    assert prediction.prediction == "Pneumothorax"
    assert prediction.confidence == pytest.approx(0.85)
    assert prediction.tier_used == 1
    assert prediction.mc_variance is None


def test_tiered_system_route_tier2() -> None:
    t1 = MagicMock()
    t1.get_confidence.return_value = (torch.tensor(0.60), torch.tensor(0))

    t2 = MagicMock()
    t2.mc_tta_forward.return_value = (torch.tensor([[0.2, 0.8]]), torch.tensor([[0.01, 0.02]]))

    config = {
        "model": {
            "confidence_threshold": 0.75,
            "threshold_window_size": 3,
            "threshold_delta": 0.05,
            "mc_dropout_passes": 20,
            "tta_passes": 10,
        }
    }
    system = TieredSystem(t1, t2, config)
    prediction = system.route(torch.zeros(1, 3, 224, 224))
    assert prediction.prediction == "Pneumothorax"
    assert prediction.confidence == pytest.approx(0.8)
    assert prediction.tier_used == 2
    assert prediction.mc_variance == pytest.approx(0.02)
    assert not prediction.flagged_for_review


def test_tiered_system_route_tier2_flagged() -> None:
    t1 = MagicMock()
    t1.get_confidence.return_value = (torch.tensor(0.60), torch.tensor(0))

    t2 = MagicMock()
    t2.mc_tta_forward.return_value = (torch.tensor([[0.2, 0.8]]), torch.tensor([[0.01, 0.12]]))

    config = {
        "model": {
            "confidence_threshold": 0.75,
            "threshold_window_size": 3,
            "threshold_delta": 0.05,
            "mc_dropout_passes": 20,
            "tta_passes": 10,
        }
    }
    system = TieredSystem(t1, t2, config)
    prediction = system.route(torch.zeros(1, 3, 224, 224))
    assert prediction.prediction == "Pneumothorax"
    assert prediction.mc_variance == pytest.approx(0.12)
    assert prediction.flagged_for_review
