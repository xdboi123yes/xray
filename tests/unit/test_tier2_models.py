"""Unit tests for the Tier 2 models.

Verifies initialization, unfreezing hooks, shapes, MC Dropout, TTA,
and combined MC-TTA behaviors for Tier2EfficientNet and Tier2ArkPlus.
"""

import torch

from core.interfaces.base_model import BaseClassifier
from core.models.tier2_ark import Tier2ArkPlus
from core.models.tier2_efficientnet import Tier2EfficientNet


def test_tier2_efficientnet_init() -> None:
    """Test that Tier 2 EfficientNet initializes correctly."""
    model = Tier2EfficientNet(num_classes=2, pretrained=False)
    assert isinstance(model, BaseClassifier)
    assert isinstance(model, torch.nn.Module)
    assert model.backbone.classifier[3].out_features == 2


def test_tier2_efficientnet_forward(dummy_xray_tensor: torch.Tensor) -> None:
    """Test standard forward shape for Tier 2 EfficientNet."""
    model = Tier2EfficientNet(num_classes=2, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(dummy_xray_tensor)

    assert logits.shape == (1, 2)


def test_tier2_efficientnet_mc_tta(dummy_xray_tensor: torch.Tensor) -> None:
    """Test MC, TTA, and joint MC-TTA shapes for Tier 2 EfficientNet."""
    model = Tier2EfficientNet(num_classes=2, pretrained=False)
    model.eval()

    mean_probs, variance = model.mc_tta_forward(dummy_xray_tensor, T=3, n_augments=3)
    assert mean_probs.shape == (1, 2)
    assert variance.shape == (1, 2)


def test_tier2_ark_init() -> None:
    """Test that Tier 2 Ark+ Swin model initializes correctly."""
    # Using 'tiny' variant to keep memory and CPU loads minimal for testing
    model = Tier2ArkPlus(num_classes=2, variant="tiny", pretrained=False, freeze_epochs=3)
    assert isinstance(model, BaseClassifier)
    assert model._frozen
    assert model.classifier[3].out_features == 2


def test_tier2_ark_unfreezing() -> None:
    """Test progressive backbone unfreezing in Tier 2 Ark+."""
    model = Tier2ArkPlus(num_classes=2, variant="tiny", pretrained=False, freeze_epochs=3)
    assert model._frozen

    # Epoch 2 < 3: should remain frozen
    model.unfreeze_at_epoch(2)
    assert model._frozen

    # Epoch 3 >= 3: should unfreeze backbone
    model.unfreeze_at_epoch(3)
    assert not model._frozen


def test_tier2_ark_forward(dummy_xray_tensor: torch.Tensor) -> None:
    """Test forward and MC-TTA shapes for Tier 2 Ark+."""
    model = Tier2ArkPlus(num_classes=2, variant="tiny", pretrained=False, freeze_epochs=0)
    model.eval()

    with torch.no_grad():
        logits = model(dummy_xray_tensor)
    assert logits.shape == (1, 2)

    mean_probs, variance = model.mc_tta_forward(dummy_xray_tensor, T=2, n_augments=2)
    assert mean_probs.shape == (1, 2)
    assert variance.shape == (1, 2)
