"""Repository for managing CheXpert out-of-distribution evaluation splits.

Implements the BaseRepository interface to decouple data access details from systems.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from core.interfaces.base_repository import BaseRepository
from infrastructure.data.chexpert_dataset import CheXpertDataset


class EmptyMockDataset(Dataset[Any]):  # type: ignore[misc]
    """Empty mock container to satisfy type bounds for splits not utilized in OOD."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        raise IndexError("Mock dataset has zero elements")


class CheXpertRepository(BaseRepository):
    """Repository accessing CheXpert dataset splits for zero-shot testing."""

    def __init__(
        self,
        test_csv: str = "data/processed/chexpert_test.csv",
        image_dir: str | None = None,
        transform: Any = None,
    ) -> None:
        """Initialize the CheXpertRepository.

        Args:
            test_csv: Path to the processed CheXpert test/validation CSV.
            image_dir: Folder containing raw CheXpert images.
            transform: Transformations applied to images (resize, normalize, etc.).
        """
        self.test_csv = test_csv
        self.image_dir = image_dir
        self.transform = transform

    def get_train_dataset(self) -> Dataset[Any]:
        """Retrieve training split (not used in zero-shot).

        Returns:
            An empty mock dataset container.
        """
        return EmptyMockDataset()

    def get_val_dataset(self) -> Dataset[Any]:
        """Retrieve validation split (not used in zero-shot).

        Returns:
            An empty mock dataset container.
        """
        return EmptyMockDataset()

    def get_test_dataset(self) -> Dataset[Any]:
        """Retrieve the CheXpert zero-shot testing dataset.

        Returns:
            A populated CheXpertDataset instance for evaluation.
        """
        return CheXpertDataset(
            csv_file=self.test_csv,
            image_dir=self.image_dir,
            transform=self.transform,
        )

    def get_calibration_dataset(self) -> Dataset[Any]:
        """Retrieve conformal calibration split (not used in zero-shot).

        Returns:
            An empty mock dataset container.
        """
        return EmptyMockDataset()
