"""Classical Albumentations transforms and batch-level Mixup/CutMix regularization."""

from __future__ import annotations

from typing import Any, cast

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


class ClassicalAugmentation:
    """Albumentations pipeline for chest radiograph training and evaluation."""

    def __init__(self, image_size: int = 224, is_training: bool = True) -> None:
        self._image_size = image_size
        self._is_training = is_training
        self._pipeline = self._build_pipeline()

    @property
    def pipeline(self) -> A.Compose:
        """Return the underlying Albumentations Compose transform."""
        return self._pipeline

    def _build_pipeline(self) -> A.Compose:
        if self._is_training:
            return A.Compose(
                [
                    A.Resize(height=self._image_size, width=self._image_size),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=10, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.GaussNoise(p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        return A.Compose(
            [
                A.Resize(height=self._image_size, width=self._image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def apply(
        self, image: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Apply classical transforms to a numpy ``[H, W, C]`` array or ``[C, H, W]`` tensor."""
        arr = (
            image.detach().cpu().numpy().transpose(1, 2, 0)
            if isinstance(image, torch.Tensor)
            else image
        )
        transformed = self._pipeline(image=arr)["image"]
        if isinstance(transformed, torch.Tensor):
            return transformed
        return cast(np.ndarray[Any, Any] | torch.Tensor, transformed)


class MixupCutmixAugmentation:
    """Batch-level Mixup and CutMix regularization for radiograph training."""

    def __init__(
        self,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
    ) -> None:
        self._mixup_alpha = mixup_alpha
        self._cutmix_alpha = cutmix_alpha
        self._prob = prob

    def apply(
        self, image: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Single-image no-op (Mixup/CutMix operates on mini-batches)."""
        return image

    def apply_batch(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Randomly apply Mixup or CutMix to a batch of tensors and labels."""
        if float(torch.rand(1).item()) > self._prob or len(images) < 2:
            return images, targets, targets, 1.0

        use_cutmix = torch.rand(1).item() > 0.5
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        if use_cutmix:
            lam = float(torch.distributions.Beta(self._cutmix_alpha, self._cutmix_alpha).sample())
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images_mixed = images.clone()
            images_mixed[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-2) * images.size(-1)))
        else:
            lam = float(torch.distributions.Beta(self._mixup_alpha, self._mixup_alpha).sample())
            images_mixed = lam * images + (1.0 - lam) * images[index]

        return images_mixed, targets, targets[index], lam

    def _rand_bbox(self, size: torch.Size, lam: float) -> tuple[int, int, int, int]:
        w = size[-2]
        h = size[-1]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))

        x1 = int(np.clip(cx - cut_w // 2, 0, w))
        y1 = int(np.clip(cy - cut_h // 2, 0, h))
        x2 = int(np.clip(cx + cut_w // 2, 0, w))
        y2 = int(np.clip(cy + cut_h // 2, 0, h))
        return x1, y1, x2, y2
