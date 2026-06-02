"""Unit tests for HiResCAM helpers: target resolution and background masking."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from core.explainability.hirescam import (
    _swin_reshape_transform,
    resolve_target_layers,
    suppress_background,
)


class _ConvModel(nn.Module):
    """Stand-in for a torchvision conv backbone exposing ``backbone.features``."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.features = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv2d(4, 4, 3))


class _SwinModel(nn.Module):
    """Stand-in for a timm Swin backbone exposing ``backbone.layers[*].blocks[*].norm2``."""

    def __init__(self) -> None:
        super().__init__()
        block = nn.Module()
        block.norm2 = nn.LayerNorm(8)
        stage = nn.Module()
        stage.blocks = nn.ModuleList([block])
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([stage])


def test_resolve_conv_backbone_has_no_reshape() -> None:
    layers, reshape = resolve_target_layers(_ConvModel())
    assert len(layers) == 1
    assert reshape is None


def test_resolve_swin_backbone_has_reshape() -> None:
    layers, reshape = resolve_target_layers(_SwinModel())
    assert len(layers) == 1
    assert reshape is not None


def test_resolve_unsupported_backbone_raises() -> None:
    class _Bad(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Linear(2, 2)

    with pytest.raises(ValueError):
        resolve_target_layers(_Bad())


def test_suppress_background_zeros_black_corners() -> None:
    cam = np.ones((224, 224), dtype=np.float32)
    img = np.ones((224, 224, 3), dtype=np.float32)
    img[:24, :24] = 0.0  # a black collimation corner
    masked = suppress_background(cam, img)
    assert masked[:12, :12].mean() < 0.2  # corner attribution suppressed
    assert masked.min() >= 0.0 and masked.max() <= 1.0


def test_swin_reshape_handles_grid_and_sequence() -> None:
    assert _swin_reshape_transform(torch.randn(1, 7, 7, 8)).shape == (1, 8, 7, 7)
    assert _swin_reshape_transform(torch.randn(1, 49, 8)).shape == (1, 8, 7, 7)
