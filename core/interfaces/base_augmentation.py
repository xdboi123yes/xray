"""Base augmentation strategy interface for the Chest X-Ray classification system.

This module defines the abstract base class for augmentation strategies applied
to chest X-ray images during training or evaluation.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseAugmentation(ABC):
    """Abstract base class for all image augmentation strategies.

    Enforces a common interface for applying transformations (classical,
    diffusion-based synthetic additions, regularization) to image data.
    """

    @abstractmethod
    def apply(
        self, image: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Apply the specific augmentation to the input image.

        Args:
            image: Input image as a numpy array or PyTorch tensor.

        Returns:
            The augmented image of the same type (numpy array or PyTorch tensor).
        """
        pass
