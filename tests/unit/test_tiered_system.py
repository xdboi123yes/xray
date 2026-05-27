"""Unit tests for the TieredSystem wrapper class.

Verifies dynamic threshold updates and routing decisions for Tier 1 and Tier 2 models.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import torch

from core.models.tiered_system import TieredSystem, TieredPrediction


def test_tiered_prediction_dataclass() -> None:
    """Verifies that TieredPrediction sets correct default parameters."""
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
    """Test that TieredSystem initializes configuration variables correctly."""
    t1 = MagicMock()
    t2 = MagicMock()
    config = {
        "model": {
            "confidence_threshold": 0.75,
            "threshold_window_size": 5,
            "threshold_delta": 0.05,
            "mc_dropout_passes": 20,
            "tta_passes": 10,
        }
    }
    system = TieredSystem(t1, t2, config)
    assert system.current_threshold == 0.75
    assert system.window_size == 5
    assert system.threshold_delta == 0.05
    assert len(system.recent_confidences) == 0


def test_tiered_system_update_threshold() -> None:
    """Test adaptive threshold shifts on high and low confidence streams."""
    t1 = MagicMock()
    t2 = MagicMock()
    config = {
        "model": {
            "confidence_threshold": 0.75,
            "threshold_window_size": 3,
            "threshold_delta": 0.05,
        }
    }
    system = TieredSystem(t1, t2, config)

    # 1. Test low confidence shift
    system.update_threshold(0.50)
    system.update_threshold(0.55)
    system.update_threshold(0.60)
    # 4th update shifts window, mean = (0.50+0.55+0.60)/3 = 0.55 (< 0.65)
    system.update_threshold(0.58)
    assert system.current_threshold == 0.70

    # Reset
    system.current_threshold = 0.75
    system.recent_confidences = []

    # 2. Test high confidence shift
    system.update_threshold(0.90)
    system.update_threshold(0.92)
    system.update_threshold(0.95)
    # 4th update shifts window, mean = (0.90+0.92+0.95)/3 = 0.923 (> 0.85)
    system.update_threshold(0.89)
    assert system.current_threshold == 0.80


def test_tiered_system_route_tier1() -> None:
    """Test routing when Tier 1 model has confidence above threshold."""
    t1 = MagicMock()
    t1.return_value = torch.tensor([[0.1, 0.9]])
    t1.get_confidence.return_value = (torch.tensor(0.85), torch.tensor(1))

    t2 = MagicMock()
    config = {
        "model": {
            "confidence_threshold": 0.75,
            "threshold_window_size": 3,
            "threshold_delta": 0.05,
        }
    }
    system = TieredSystem(t1, t2, config)
    x = torch.zeros(1, 3, 224, 224)

    import pytest
    prediction = system.route(x)
    assert prediction.prediction == "Pneumothorax"
    assert prediction.confidence == pytest.approx(0.85)
    assert prediction.tier_used == 1
    assert prediction.mc_variance is None


def test_tiered_system_route_tier2() -> None:
    """Test routing escalation to Tier 2 when Tier 1 confidence is low."""
    t1 = MagicMock()
    t1.get_confidence.return_value = (torch.tensor(0.60), torch.tensor(0))

    t2 = MagicMock()
    t2_mean_probs = torch.tensor([[0.2, 0.8]])
    t2_variance = torch.tensor([[0.01, 0.02]])
    t2.mc_tta_forward.return_value = (t2_mean_probs, t2_variance)

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
    x = torch.zeros(1, 3, 224, 224)

    import pytest
    prediction = system.route(x)
    assert prediction.prediction == "Pneumothorax"
    assert prediction.confidence == pytest.approx(0.8)
    assert prediction.tier_used == 2
    assert prediction.mc_variance == pytest.approx(0.02)
    assert not prediction.flagged_for_review


def test_tiered_system_route_tier2_flagged() -> None:
    """Test high risk review flagging on Tier 2 predictions with high variance."""
    t1 = MagicMock()
    t1.get_confidence.return_value = (torch.tensor(0.60), torch.tensor(0))

    t2 = MagicMock()
    t2_mean_probs = torch.tensor([[0.2, 0.8]])
    t2_variance = torch.tensor([[0.01, 0.12]])  # variance > 0.1
    t2.mc_tta_forward.return_value = (t2_mean_probs, t2_variance)

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
    x = torch.zeros(1, 3, 224, 224)

    import pytest
    prediction = system.route(x)
    assert prediction.prediction == "Pneumothorax"
    assert prediction.mc_variance == pytest.approx(0.12)
    assert prediction.flagged_for_review
