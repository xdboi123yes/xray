"""Unit tests for the image augmentation strategies.

Verifies shapes, types, batch mixing, and composite orchestrations
for ClassicalAugmentation, MixupCutmixAugmentation, and AugmentationPipeline.
"""

from typing import Any

import numpy as np
import torch

from core.augmentation.classical import ClassicalAugmentation
from core.augmentation.composer import AugmentationPipeline
from core.augmentation.mixup_cutmix import MixupCutmixAugmentation


def test_classical_augmentation_ndarray() -> None:
    """Test ClassicalAugmentation on a numpy array input."""
    # Create mock 224x224 RGB image
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    aug = ClassicalAugmentation(image_size=224, is_training=True)
    result = aug.apply(arr)

    # Result should be a normalized PyTorch tensor due to ToTensorV2
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)


def test_classical_augmentation_tensor(dummy_xray_tensor: torch.Tensor) -> None:
    """Test ClassicalAugmentation on a PyTorch tensor input.

    Args:
        dummy_xray_tensor: Pytest fixture representing a chest X-ray image batch.
    """
    # ClassicalAugmentation expects shape [C, H, W] for single images,
    # dummy_xray_tensor is shape [1, C, H, W]
    single_img = dummy_xray_tensor[0]

    aug = ClassicalAugmentation(image_size=128, is_training=False)
    result = aug.apply(single_img)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 128, 128)


def test_mixup_cutmix_single_image() -> None:
    """Test that MixupCutmix is a no-op on single images."""
    arr = np.zeros((3, 224, 224))
    aug = MixupCutmixAugmentation(prob=1.0)

    # Should return unmodified input
    assert aug.apply(arr) is arr


def test_mixup_cutmix_batch() -> None:
    """Test batch-level mixup and cutmix execution."""
    images = torch.ones(4, 3, 224, 224)
    targets = torch.tensor([0, 1, 0, 1])

    # Probability 1.0 ensures mixing is applied
    aug = MixupCutmixAugmentation(mixup_alpha=0.4, cutmix_alpha=1.0, prob=1.0)
    mixed_imgs, mixed_targets, perm_targets, lam = aug.apply_batch(images, targets)

    assert mixed_imgs.shape == (4, 3, 224, 224)
    assert mixed_targets.shape == (4,)
    assert perm_targets.shape == (4,)
    assert 0.0 <= lam <= 1.0


def test_augmentation_pipeline_composer(mock_config: dict[str, Any]) -> None:
    """Test composite AugmentationPipeline construction from config.

    Args:
        mock_config: Pytest fixture representing config.yaml parameters.
    """
    # Enable classical and mixup, exclude diffusion for unit test
    cfg = mock_config.copy()
    cfg["data"]["synthetic_augmentation"] = False
    cfg["data"]["mixup_cutmix_regularization"] = True
    cfg["mixup_cutmix"] = {"mixup_alpha": 0.2, "cutmix_alpha": 0.8, "prob": 0.5}

    pipeline = AugmentationPipeline.from_config(cfg)

    # Should contain ClassicalAugmentation and MixupCutmixAugmentation
    assert len(pipeline.augmentations) == 2
    assert isinstance(pipeline.augmentations[0], ClassicalAugmentation)
    assert isinstance(pipeline.augmentations[1], MixupCutmixAugmentation)

    # Test applying classical step of pipeline on mock image
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    result = pipeline.apply(arr)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)
