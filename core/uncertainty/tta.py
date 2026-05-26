"""Test-Time Augmentation (TTA) Uncertainty Quantification Module.

Manages data transformations and inference averaging pipelines to calculate 
predictive variance and improve model robustness under distribution shifts.
"""

from __future__ import annotations

from typing import Any
import torch
import torchvision.transforms as T

class TestTimeAugmenter:
    """Manages test-time augmentation ensembles for chest radiograph predictions."""

    def __init__(self, n_augments: int = 10) -> None:
        """Initialize TestTimeAugmenter.

        Args:
            n_augments: Number of augmented samples to generate per image.
        """
        self.n_augments = n_augments
        # Define clean clinical-safe spatial transformations
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def generate_augments(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Generates TTA variations of the input image tensor.

        Args:
            x: Input image tensor of shape [1, C, H, W].

        Returns:
            List of augmented tensors.
        """
        augments = [x]  # Always include original unmodified image
        for _ in range(self.n_augments - 1):
            augments.append(self.transforms(x))
        return augments

    def predict_with_tta(self, model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs model predictions with Test-Time Augmentation.

        Args:
            model: PyTorch model.
            x: Input image tensor.

        Returns:
            Tuple of (mean_probs, variance).
        """
        model.eval()
        aug_inputs = self.generate_augments(x)
        probs_list = []

        with torch.no_grad():
            for aug_x in aug_inputs:
                logits = model(aug_x)
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)

        probs_stack = torch.stack(probs_list)  # [N, B, NumClasses]
        mean_probs = probs_stack.mean(dim=0)
        variance = probs_stack.var(dim=0).mean(dim=-1)

        return mean_probs, variance
