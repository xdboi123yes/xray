"""Composite augmentation pipeline for Chest X-Ray classification.

Implements the Composite pattern to orchestrate multiple sequential
augmentation strategies based on training configurations or ablation settings.
"""

from typing import Any

import numpy as np
import torch

from core.interfaces.base_augmentation import BaseAugmentation


class AugmentationPipeline(BaseAugmentation):
    """Composite augmentation strategy.

    Maintains a sequence of BaseAugmentation objects and applies them in order,
    allowing runtime modification of training regularizations.
    """

    def __init__(self, augmentations: list[BaseAugmentation]) -> None:
        """Initialize the composite pipeline.

        Args:
            augmentations: List of concrete BaseAugmentation strategies.
        """
        self._augmentations = augmentations

    def apply(
        self, image: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Apply all pipeline augmentations sequentially to the input image.

        Args:
            image: Input image as a numpy array or PyTorch tensor.

        Returns:
            The fully augmented image.
        """
        result = image
        for aug in self._augmentations:
            result = aug.apply(result)
        return result

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AugmentationPipeline":
        """Factory method to construct a pipeline dynamically from configuration.

        Args:
            config: The active configuration dictionary.

        Returns:
            An initialized AugmentationPipeline composite.
        """
        from core.augmentation.classical import ClassicalAugmentation
        from core.augmentation.diffusion import DiffusionAugmentation
        from core.augmentation.mixup_cutmix import MixupCutmixAugmentation

        augs: list[BaseAugmentation] = []

        # Read enabled strategies from config
        data_cfg = config.get("data", {})
        image_size = data_cfg.get("image_size", 224)

        # 1. Classical transforms (always enabled for standardization/resize)
        augs.append(ClassicalAugmentation(image_size=image_size, is_training=True))

        # 2. Diffusion-based synthetic data zenhiglestirmesi
        if data_cfg.get("synthetic_augmentation", False):
            # Safe wrapper to only append if diffusion is explicitly configured
            augs.append(DiffusionAugmentation(config=config))

        # 3. Mixup/Cutmix regularization (optional, default: False or configured)
        if data_cfg.get("mixup_cutmix_regularization", False):
            mixup_cfg = config.get("mixup_cutmix", {})
            augs.append(
                MixupCutmixAugmentation(
                    mixup_alpha=mixup_cfg.get("mixup_alpha", 0.4),
                    cutmix_alpha=mixup_cfg.get("cutmix_alpha", 1.0),
                    prob=mixup_cfg.get("prob", 0.5),
                )
            )

        return cls(augs)

    @property
    def augmentations(self) -> list[BaseAugmentation]:
        """Get the underlying list of active augmentation strategies.

        Returns:
            List of BaseAugmentation strategies.
        """
        return self._augmentations
