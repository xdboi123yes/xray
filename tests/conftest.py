"""Pytest configuration and shared fixtures for unit and integration testing.

Defines common testing fixtures like dummy chest X-ray tensors and configuration mocks.
"""

from typing import Any

import pytest
import torch


@pytest.fixture
def dummy_xray_tensor() -> torch.Tensor:
    """Create a dummy chest X-ray image tensor.

    Shape: [1, 3, 224, 224] representing a batch of 1 RGB image.

    Returns:
        A random PyTorch float tensor with values normalized between 0 and 1.
    """
    return torch.rand(1, 3, 224, 224)


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Create a mock configuration dictionary mirroring config.yaml layout.

    Returns:
        A dictionary with typical model and training configurations.
    """
    return {
        "model": {
            "tier1_backbone": "mobilenet_v2",
            "tier2_backbone": "efficientnet_b4",
            "confidence_threshold": 0.75,
            "threshold_window_size": 50,
            "threshold_delta": 0.05,
            "mc_dropout_passes": 20,
            "tta_passes": 10,
        },
        "training": {
            "batch_size": 32,
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
            "epochs": 50,
            "early_stopping_patience": 7,
            "seed": 42,
        },
        "data": {
            "image_size": 224,
            "train_split": 0.70,
            "val_split": 0.15,
            "test_split": 0.15,
            "synthetic_augmentation": True,
        },
    }
