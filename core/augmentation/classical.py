"""Classical image augmentation strategy for Chest X-Ray classification.

Wraps Albumentations pipelines for training and validation transforms.
"""

from typing import Any, cast

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from core.interfaces.base_augmentation import BaseAugmentation


class ClassicalAugmentation(BaseAugmentation):
    """Classical medical image augmentation strategy.

    Configures and executes Albumentations pipelines for medical X-ray resizing,
    jitter, spatial translations, and normalization.
    """

    def __init__(self, image_size: int = 224, is_training: bool = True) -> None:
        """Initialize classical augmentations.

        Args:
            image_size: Resolution to resize target images.
            is_training: If True, apply training augmentations, else validation transforms.
        """
        self._image_size = image_size
        self._is_training = is_training
        self._pipeline = self._build_pipeline()

    def _build_pipeline(self) -> A.Compose:
        """Construct the Albumentations Compose pipeline.

        Returns:
            An Albumentations Compose object.
        """
        if self._is_training:
            return A.Compose(
                [
                    A.Resize(height=self._image_size, width=self._image_size),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0, scale_limit=0.1, rotate_limit=10, p=0.5
                    ),
                    A.RandomBrightnessContrast(p=0.5),
                    A.GaussNoise(p=0.2),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(height=self._image_size, width=self._image_size),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                    ToTensorV2(),
                ]
            )

    def apply(
        self, image: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Apply classical resizing and transformations.

        Args:
            image: Input image as a numpy array [H, W, C] or PyTorch tensor.

        Returns:
            The augmented and normalized image as a tensor or array.
        """
        if isinstance(image, torch.Tensor):
            # Convert PyTorch tensor back to numpy [H, W, C] for Albumentations if needed
            # Assuming tensor shape is [C, H, W]
            arr = image.detach().cpu().numpy().transpose(1, 2, 0)
        else:
            arr = image

        result = self._pipeline(image=arr)
        transformed = result["image"]

        if isinstance(transformed, torch.Tensor):
            return transformed
        else:
            return cast(np.ndarray[Any, Any] | torch.Tensor, transformed)
