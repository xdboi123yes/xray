"""Unit tests for XRayGradCAMPlusPlus (Grad-CAM++ visualizations)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core.explainability.gradcam_pp import XRayGradCAMPlusPlus


class ToyConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU())
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def test_gradcam_pp_pipeline() -> None:
    """Verify that Grad-CAM++ generates saliency maps and overlays correctly."""
    model = ToyConvModel()
    target_layer = model.features[0]

    explainer = XRayGradCAMPlusPlus(model=model, target_layers=[target_layer])

    dummy_input = torch.randn(1, 3, 224, 224)
    cam = explainer.generate(dummy_input)

    assert cam.shape == (224, 224)
    assert np.min(cam) >= 0.0
    assert np.max(cam) <= 1.0

    # Test overlay
    dummy_rgb = np.random.rand(224, 224, 3)
    overlayed = explainer.overlay(dummy_rgb, cam)
    assert overlayed.shape == (224, 224, 3)
