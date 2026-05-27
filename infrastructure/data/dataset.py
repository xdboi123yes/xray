"""NIH Chest X-Ray dataset class for PyTorch models.

Provides batch loading, Pneumothorax filtering, and synthetic image augmentation.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
from PIL import Image
from torch.utils.data import Dataset

log = structlog.get_logger(__name__)


def _read_image_dir() -> str | None:
    """Reads processed image directory path from data/processed/image_dir.txt if it exists.

    Returns:
        The processed image directory path or None.
    """
    txt_path = os.path.join("data", "processed", "image_dir.txt")
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            return f.read().strip()
    return None


class NIHChestXrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Dataset implementation for NIH Chest X-Rays with optional synthetic additions."""

    def __init__(
        self,
        csv_file: str,
        image_dir: str | None = None,
        transform: Callable[[np.ndarray[Any, Any]], dict[str, Any]] | None = None,
        use_synthetic: bool = False,
        synthetic_dir: str | None = None,
        synthetic_csv: str | None = None,
    ) -> None:
        """Initializes the NIHChestXrayDataset.

        Args:
            csv_file: Path to the main CSV file containing annotations.
            image_dir: Path to directory containing raw images.
            transform: Optional Albumentations transform pipeline.
            use_synthetic: True to merge synthetic Pneumothorax X-Rays.
            synthetic_dir: Optional directory with synthetic image files.
            synthetic_csv: Optional metadata CSV for synthetic images.
        """
        # Resolve image_dir: explicit param > image_dir.txt > fallback
        if image_dir is None:
            image_dir = _read_image_dir()
        if image_dir is None:
            image_dir = "data/raw/images"
        self.image_dir = image_dir
        self.transform = transform

        # Load main CSV
        self.data_frame = pd.read_csv(csv_file)

        # If 'Label' column already exists (preprocessed CSV), use it directly
        if "Label" not in self.data_frame.columns:
            # Filter for Pneumothorax (str.contains) and No Finding (exact match)
            mask = self.data_frame["Finding Labels"].str.contains("Pneumothorax", na=False) | (
                self.data_frame["Finding Labels"] == "No Finding"
            )
            self.data_frame = self.data_frame[mask].copy()

            # Create labels: 1 if Pneumothorax anywhere in Finding Labels, else 0
            self.data_frame["Label"] = self.data_frame["Finding Labels"].apply(
                lambda x: 1 if "Pneumothorax" in str(x) else 0
            )

        # Build image paths from single flat directory
        self.data_frame["Image Path"] = self.data_frame["Image Index"].apply(
            lambda x: os.path.join(self.image_dir, x)
        )

        # Optionally add synthetic data
        if use_synthetic and synthetic_dir and synthetic_csv:
            if os.path.exists(synthetic_csv):
                synth_df = pd.read_csv(synthetic_csv)
                synth_df["Label"] = 1
                synth_df["Finding Labels"] = "Pneumothorax"
                synth_df["Image Path"] = synth_df["Image Index"].apply(
                    lambda x: os.path.join(str(synthetic_dir), x)
                )
                self.data_frame = pd.concat([self.data_frame, synth_df], ignore_index=True)
            else:
                log.warning(
                    f"Warning: synthetic CSV {synthetic_csv} not found. "
                    "Proceeding without synthetic data."
                )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Loads and processes the dataset sample at the given index.

        Args:
            idx: Index of the sample to fetch.

        Returns:
            A tuple of (image tensor, label tensor, image index string).
        """
        if torch.is_tensor(idx):
            idx_list = idx.tolist()  # type: ignore[union-attr]
            idx = int(idx_list[0]) if isinstance(idx_list, list) else int(idx_list)  # type: ignore[arg-type]
        else:
            idx = int(idx)

        img_path: str = self.data_frame.iloc[idx]["Image Path"]
        image_id: str = self.data_frame.iloc[idx]["Image Index"]
        label: int = int(self.data_frame.iloc[idx]["Label"])

        # NIH images are grayscale, but convert to RGB for standard CNN backbones
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            # Albumentations expects numpy arrays
            image_np = np.array(image)
            augmented = self.transform(image=image_np)  # type: ignore[call-arg]
            image_val = augmented["image"]
            if isinstance(image_val, torch.Tensor):
                image_tensor = image_val
            else:
                image_tensor = torch.from_numpy(np.array(image_val))
        else:
            # Fallback PIL to PyTorch float tensor conversion
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        # Ensure label is a tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor, image_id
