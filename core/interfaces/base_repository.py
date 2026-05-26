"""Base repository interface for data access in the Chest X-Ray classification system.

This module defines the abstract base class for repositories managing access to various
splits of data (NIH, CheXpert, Synthetic).
"""

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class BaseRepository(ABC):
    """Abstract base class for data repositories.

    Defines a unified interface to retrieve train, validation, test, and
    calibration splits, abstracting the source data implementation.
    """

    @abstractmethod
    def get_train_dataset(self) -> Dataset[Any]:
        """Retrieve the dataset split designated for training.

        Returns:
            A PyTorch Dataset instance containing training data.
        """
        pass

    @abstractmethod
    def get_val_dataset(self) -> Dataset[Any]:
        """Retrieve the dataset split designated for validation.

        Returns:
            A PyTorch Dataset instance containing validation data.
        """
        pass

    @abstractmethod
    def get_test_dataset(self) -> Dataset[Any]:
        """Retrieve the dataset split designated for testing.

        Returns:
            A PyTorch Dataset instance containing testing data.
        """
        pass

    @abstractmethod
    def get_calibration_dataset(self) -> Dataset[Any]:
        """Retrieve the dataset split designated for conformal prediction calibration.

        Returns:
            A PyTorch Dataset instance containing calibration data.
        """
        pass
