"""Base classifier interface for the Chest X-Ray classification system.

This module defines the abstract base class for all classifiers within the tiered
inference architecture.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseClassifier(nn.Module, ABC):
    """Abstract base class for all tiered classifiers.

    Enforces standard PyTorch forward API along with uncertainty estimation methods
    including Monte Carlo Dropout and Test-Time Augmentation (TTA).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits.

        Args:
            x: Input image tensor of shape [batch_size, channels, height, width].

        Returns:
            Raw logits tensor of shape [batch_size, num_classes].
        """
        pass

    @abstractmethod
    def get_confidence(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute predicted class indices and corresponding confidence scores.

        Args:
            logits: Raw logits tensor of shape [batch_size, num_classes].

        Returns:
            A tuple of (confidences, predictions) tensors.
        """
        pass

    @abstractmethod
    def mc_forward(self, x: torch.Tensor, T: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform T forward passes with active dropout to estimate epistemic uncertainty.

        Args:
            x: Input image tensor of shape [batch_size, channels, height, width].
            T: The number of stochastic forward passes.

        Returns:
            A tuple of (mean_probabilities, variances) tensors.
        """
        pass

    @abstractmethod
    def tta_forward(self, x: torch.Tensor, n_augments: int = 10) -> torch.Tensor:
        """Perform Test-Time Augmentation (TTA) forward pass over multiple variations.

        Args:
            x: Input image tensor of shape [batch_size, channels, height, width].
            n_augments: The number of augmented variations (including the original).

        Returns:
            Mean probabilities tensor of shape [batch_size, num_classes].
        """
        pass

    @abstractmethod
    def mc_tta_forward(
        self, x: torch.Tensor, T: int = 20, n_augments: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine Monte Carlo Dropout and Test-Time Augmentation.

        Args:
            x: Input image tensor of shape [batch_size, channels, height, width].
            T: The number of stochastic dropout runs per augmented image.
            n_augments: The number of augmented variations.

        Returns:
            A tuple of (mean_probabilities, variances) tensors.
        """
        pass
