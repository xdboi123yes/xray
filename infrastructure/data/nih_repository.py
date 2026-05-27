"""Repository and Dataset for managing NIH ChestX-ray14 out-of-distribution evaluation splits.

Implements the BaseRepository interface to decouple data access details from systems.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from core.interfaces.base_repository import BaseRepository


class EmptyMockDataset(Dataset[Any]):
    """Empty mock container to satisfy type bounds for splits not utilized in OOD."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        raise IndexError("Mock dataset has zero elements")

class NIHDataset(Dataset[Any]):
    """PyTorch Dataset loading NIH ChestX-ray14 radiographs for evaluation."""

    def __init__(
        self,
        csv_file: str,
        image_dir: str | None = None,
        transform: Any = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        if not os.path.exists(csv_file):
            self.data_frame = pd.DataFrame(columns=["Path", "Label", "Image Index"])
            return

        self.data_frame = pd.read_csv(csv_file)

        # Ensure dynamic binary mapping
        # Pneumothorax = 1.0 -> Class 1 (Positive)
        # No Finding = 1.0 -> Class 0 (Negative)
        if "Label" not in self.data_frame.columns:
            p_col = "Pneumothorax"
            nf_col = "No Finding"

            if p_col in self.data_frame.columns and nf_col in self.data_frame.columns:
                mask = (self.data_frame[p_col] == 1.0) | (self.data_frame[nf_col] == 1.0)
                self.data_frame = self.data_frame[mask].copy()
                self.data_frame["Label"] = self.data_frame[p_col].apply(
                    lambda x: 1 if x == 1.0 else 0
                )
            else:
                self.data_frame["Label"] = 0

        if "Image Index" not in self.data_frame.columns:
            self.data_frame["Image Index"] = self.data_frame["Path"].apply(
                lambda x: os.path.basename(str(x))
            )

        self.data_frame["Image Path"] = self.data_frame.apply(
            lambda row: os.path.join(self.image_dir, str(row["Image Index"]))
            if self.image_dir
            else str(row["Path"]),
            axis=1,
        )

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple[Any, torch.Tensor, str]:
        row = self.data_frame.iloc[idx]
        img_path = row["Image Path"]
        image_id = row["Image Index"]
        label_val = int(row["Label"])

        if not os.path.exists(img_path):
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = augmented["image"]

        label = torch.tensor(label_val, dtype=torch.long)
        return image, label, image_id

class NIHRepository(BaseRepository):
    """Repository accessing NIH ChestX-ray14 dataset splits for zero-shot testing."""

    def __init__(
        self,
        test_csv: str = "data/processed/nih_test.csv",
        image_dir: str | None = None,
        transform: Any = None,
    ) -> None:
        self.test_csv = test_csv
        self.image_dir = image_dir
        self.transform = transform

    def get_train_dataset(self) -> Dataset[Any]:
        return EmptyMockDataset()

    def get_val_dataset(self) -> Dataset[Any]:
        return EmptyMockDataset()

    def get_test_dataset(self) -> Dataset[Any]:
        return NIHDataset(
            csv_file=self.test_csv,
            image_dir=self.image_dir,
            transform=self.transform,
        )

    def get_calibration_dataset(self) -> Dataset[Any]:
        return EmptyMockDataset()
