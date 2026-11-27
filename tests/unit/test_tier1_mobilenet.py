"""Unit tests for Tier1MobileNet and inherited BaseClassifier Uncertainty Decomposition."""

from __future__ import annotations

import torch
import torch.nn as nn
from tcrp.models import BaseClassifier, Tier1MobileNet


def test_tier1_mobilenet_init() -> None:
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    assert isinstance(model, BaseClassifier)
    assert isinstance(model, torch.nn.Module)
    assert model.backbone.classifier[3].out_features == 2


def test_tier1_mobilenet_forward(dummy_xray_tensor: torch.Tensor) -> None:
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(dummy_xray_tensor)
    assert logits.shape == (1, 2)


def test_tier1_mobilenet_get_confidence() -> None:
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    confidences, predictions = model.get_confidence(torch.tensor([[0.0, 10.0]]))
    assert predictions.item() == 1
    assert confidences.item() > 0.999


def test_tier1_mobilenet_mc_forward_preserves_batchnorm_eval(
    dummy_xray_tensor: torch.Tensor,
) -> None:
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    model.eval()

    bn_states_during: list[bool] = []

    def _hook(module: nn.Module, _inp: object, _out: object) -> None:
        bn_states_during.append(module.training)

    first_bn = next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    handle = first_bn.register_forward_hook(_hook)
    try:
        mean_probs, variance = model.mc_forward(dummy_xray_tensor, T=4)
    finally:
        handle.remove()

    assert mean_probs.shape == (1, 2)
    assert variance.shape == (1, 2)
    assert all(is_train is False for is_train in bn_states_during)
    assert not model.training


def test_tier1_mobilenet_tta_and_decomposed_uncertainty(
    dummy_xray_tensor: torch.Tensor,
) -> None:
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    model.eval()

    tta_probs = model.tta_forward(dummy_xray_tensor, n_augments=2)
    assert tta_probs.shape == (1, 2)

    mean_probs, total_var, epistemic_var, aleatoric_var = model.mc_tta_forward_decomposed(
        dummy_xray_tensor, T=2, n_augments=2
    )
    assert mean_probs.shape == (1, 2)
    assert torch.allclose(total_var, epistemic_var + aleatoric_var, atol=1e-6)
