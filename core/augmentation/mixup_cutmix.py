"""Mixup and Cutmix regularization strategy for Chest X-Ray classification.

Implements batch-level medical augmentations that mix images and labels
to reduce overfitting and improve model generalization.
"""

from typing import Any

import numpy as np
import torch

from core.interfaces.base_augmentation import BaseAugmentation


class MixupCutmixAugmentation(BaseAugmentation):
    """Mixup and Cutmix batch regularization strategy.

    Applies Mixup (linear combination of pairs) or Cutmix (patch replacement)
    on a batch of PyTorch tensors during model training.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
    ) -> None:
        """Initialize Mixup and Cutmix.

        Args:
            mixup_alpha: Alpha parameter for Beta distribution in Mixup (usually 0.2 to 0.4).
            cutmix_alpha: Alpha parameter for Beta distribution in Cutmix (usually 1.0).
            prob: Probability of applying either Mixup or Cutmix to a batch.
        """
        self._mixup_alpha = mixup_alpha
        self._cutmix_alpha = cutmix_alpha
        self._prob = prob

    def apply(
        self, image: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Apply augmentation to a single image (no-op for batch operations).

        Mixup and Cutmix are batch-level regularizations. For single images,
        this method acts as a no-op fallback.

        Args:
            image: Input image.

        Returns:
            The unmodified input image.
        """
        return image

    def apply_batch(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply either Mixup or Cutmix to a batch of images and targets.

        Decides with probability `self._prob` whether to apply mixing. If applied,
        randomly selects between Mixup and Cutmix with equal probability.

        Args:
            images: Batch image tensor of shape [B, C, H, W].
            targets: Batch label tensor of shape [B].

        Returns:
            A tuple of (mixed_images, targets, permuted_targets, lam_value) where:
            - mixed_images: Augmented batch.
            - targets: Original labels.
            - permuted_targets: Mixed labels.
            - lam_value: The calculated Beta mixing coefficient lambda.
        """
        if float(torch.rand(1).item()) > self._prob or len(images) < 2:
            return images, targets, targets, 1.0

        # Select randomly between Mixup (0) and Cutmix (1)
        use_cutmix = torch.rand(1).item() > 0.5
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        if use_cutmix:
            lam = float(
                torch.distributions.Beta(
                    self._cutmix_alpha, self._cutmix_alpha
                ).sample()
            )
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)

            # Apply Cutmix patch replacement
            images_mixed = images.clone()
            images_mixed[:, :, bbx1:bbx2, bby1:bby2] = images[
                index, :, bbx1:bbx2, bby1:bby2
            ]

            # Adjust lambda based on actual patch area ratio
            lam = 1.0 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size(-2) * images.size(-1))
            )
        else:
            lam = float(
                torch.distributions.Beta(
                    self._mixup_alpha, self._mixup_alpha
                ).sample()
            )

            # Apply Mixup linear blending
            images_mixed = lam * images + (1.0 - lam) * images[index]

        return images_mixed, targets, targets[index], lam

    def _rand_bbox(
        self, size: torch.Size, lam: float
    ) -> tuple[int, int, int, int]:
        """Generate a random bounding box patch for Cutmix.

        Args:
            size: Dimension profile of the batch tensor.
            lam: Mixing coefficient lambda.

        Returns:
            A tuple of (x1, y1, x2, y2) crop coordinates.
        """
        w = size[-2]
        h = size[-1]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Uniformly select patch center
        cx = int(np.random.randint(w))
        cy = int(np.random.randint(h))

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        return int(x1), int(y1), int(x2), int(y2)
