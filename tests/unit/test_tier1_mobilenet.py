"""Unit tests for the Tier1MobileNet classifier.

Verifies shapes, initialization, confidence scoring, MC dropout, TTA, and
joint MC-TTA forward flows.
"""

import torch

from core.interfaces.base_model import BaseClassifier
from core.models.tier1_mobilenet import Tier1MobileNet


def test_tier1_mobilenet_init() -> None:
    """Test that the Tier 1 MobileNet model initializes correctly."""
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    assert isinstance(model, BaseClassifier)
    assert isinstance(model, torch.nn.Module)
    assert model.backbone.classifier[3].out_features == 2


def test_tier1_mobilenet_forward(dummy_xray_tensor: torch.Tensor) -> None:
    """Test standard forward pass and output shape.

    Args:
        dummy_xray_tensor: Pytest fixture representing a chest X-ray image batch.
    """
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(dummy_xray_tensor)

    assert logits.shape == (1, 2)


def test_tier1_mobilenet_get_confidence() -> None:
    """Test confidence and prediction class extraction from logits."""
    model = Tier1MobileNet(num_classes=2, pretrained=False)

    # Let class 1 be highly active: logit values [0.0, 10.0]
    logits = torch.tensor([[0.0, 10.0]])
    confidences, predictions = model.get_confidence(logits)

    assert predictions.item() == 1
    assert confidences.item() > 0.999


def test_tier1_mobilenet_mc_forward(dummy_xray_tensor: torch.Tensor) -> None:
    """Test MC dropout forward pass output shape and stats.

    Args:
        dummy_xray_tensor: Pytest fixture representing a chest X-ray image batch.
    """
    model = Tier1MobileNet(num_classes=2, pretrained=False)

    # Put in eval mode to see if mc_forward successfully forces training mode inside
    model.eval()

    mean_probs, variance = model.mc_forward(dummy_xray_tensor, T=5)

    assert mean_probs.shape == (1, 2)
    assert variance.shape == (1, 2)
    assert not model.training  # training state is restored to eval


def test_tier1_mobilenet_tta_forward(dummy_xray_tensor: torch.Tensor) -> None:
    """Test Test-Time Augmentation (TTA) forward pass output shape.

    Args:
        dummy_xray_tensor: Pytest fixture representing a chest X-ray image batch.
    """
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    model.eval()

    mean_probs = model.tta_forward(dummy_xray_tensor, n_augments=3)

    assert mean_probs.shape == (1, 2)


def test_tier1_mobilenet_mc_tta_forward(dummy_xray_tensor: torch.Tensor) -> None:
    """Test combined MC Dropout and TTA forward pass output shape.

    Args:
        dummy_xray_tensor: Pytest fixture representing a chest X-ray image batch.
    """
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    model.eval()

    mean_probs, variance = model.mc_tta_forward(dummy_xray_tensor, T=3, n_augments=3)

    assert mean_probs.shape == (1, 2)
    assert variance.shape == (1, 2)
