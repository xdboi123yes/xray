"""Unit tests for classical and Mixup/CutMix augmentation strategies."""

from __future__ import annotations

import numpy as np
import torch
from tcrp.data.augmentation import ClassicalAugmentation, MixupCutmixAugmentation


def test_classical_augmentation_ndarray() -> None:
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    aug = ClassicalAugmentation(image_size=224, is_training=True)
    result = aug.apply(arr)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)


def test_classical_augmentation_tensor(dummy_xray_tensor: torch.Tensor) -> None:
    single_img = dummy_xray_tensor[0]
    aug = ClassicalAugmentation(image_size=128, is_training=False)
    result = aug.apply(single_img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 128, 128)


def test_mixup_cutmix_single_image() -> None:
    arr = np.zeros((3, 224, 224))
    aug = MixupCutmixAugmentation(prob=1.0)
    assert aug.apply(arr) is arr


def test_mixup_cutmix_batch() -> None:
    images = torch.ones(4, 3, 224, 224)
    targets = torch.tensor([0, 1, 0, 1])
    aug = MixupCutmixAugmentation(mixup_alpha=0.4, cutmix_alpha=1.0, prob=1.0)
    mixed_imgs, mixed_targets, perm_targets, lam = aug.apply_batch(images, targets)
    assert mixed_imgs.shape == (4, 3, 224, 224)
    assert mixed_targets.shape == (4,)
    assert perm_targets.shape == (4,)
    assert 0.0 <= lam <= 1.0
