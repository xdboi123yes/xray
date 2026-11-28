"""PyTorch datasets and radiograph preprocessing utilities for NIH and CheXpert."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import structlog
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from tcrp.config import guard_mock

log = structlog.get_logger(__name__)


def _read_image_dir() -> str | None:
    """Read persisted image directory from ``data/processed/image_dir.txt`` if present."""
    txt_path = os.path.join("data", "processed", "image_dir.txt")
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            return f.read().strip()
    return None


class ImagePreprocessor:
    """Radiograph preprocessing utilities (CLAHE contrast enhancement and resize)."""

    def __init__(self, target_size: tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size
        self.clahe_transform = A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0)

    def apply_clahe(self, image: Image.Image) -> Image.Image:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        img_np = np.array(image.convert("L"))
        augmented = self.clahe_transform(image=img_np)
        return Image.fromarray(augmented["image"])

    def apply_histogram_matching(self, source: Image.Image, reference: Image.Image) -> Image.Image:
        """Normalize grayscale intensity range using autocontrast fallback."""
        _ = reference
        return ImageOps.autocontrast(source.convert("L"), cutoff=1)

    def preprocess_pipeline(self, image: Image.Image) -> Image.Image:
        """Resize, convert to grayscale, and apply CLAHE enhancement."""
        resized = image.resize(self.target_size, Image.Resampling.BILINEAR)
        return self.apply_clahe(resized.convert("L"))


class NIHChestXrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):  # type: ignore[misc]
    """NIH ChestX-ray14 dataset with optional accepted Synthetic Augmentation Batch merge."""

    def __init__(
        self,
        csv_file: str,
        image_dir: str | None = None,
        transform: Callable[[np.ndarray[Any, Any]], dict[str, Any]] | None = None,
        use_synthetic: bool = False,
        synthetic_dir: str | None = None,
        synthetic_csv: str | None = None,
    ) -> None:
        if image_dir is None:
            image_dir = _read_image_dir() or "data/raw/images"
        self.image_dir = image_dir
        self.transform = transform

        self.data_frame = pd.read_csv(csv_file)

        if "Label" not in self.data_frame.columns:
            mask = self.data_frame["Finding Labels"].str.contains("Pneumothorax", na=False) | (
                self.data_frame["Finding Labels"] == "No Finding"
            )
            self.data_frame = self.data_frame[mask].copy()
            self.data_frame["Label"] = self.data_frame["Finding Labels"].apply(
                lambda x: 1 if "Pneumothorax" in str(x) else 0
            )

        self.data_frame["Image Path"] = self.data_frame["Image Index"].apply(
            lambda x: os.path.join(self.image_dir, str(x))
        )

        if use_synthetic and synthetic_dir and synthetic_csv:
            if os.path.exists(synthetic_csv):
                synth_df = pd.read_csv(synthetic_csv)
                synth_df["Label"] = 1
                synth_df["Finding Labels"] = "Pneumothorax"
                synth_df["Image Path"] = synth_df["Image Index"].apply(
                    lambda x: os.path.join(str(synthetic_dir), str(x))
                )
                self.data_frame = pd.concat([self.data_frame, synth_df], ignore_index=True)
            else:
                log.warning(
                    "Synthetic metadata CSV not found; proceeding with real images only.",
                    synthetic_csv=synthetic_csv,
                )

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        if torch.is_tensor(idx):
            idx_list = idx.tolist()  # type: ignore[union-attr]
            idx = int(idx_list[0]) if isinstance(idx_list, list) else int(idx_list)  # type: ignore[arg-type, call-overload]
        else:
            idx = int(idx)

        row = self.data_frame.iloc[idx]
        img_path: str = str(row["Image Path"])
        image_id: str = str(row["Image Index"])
        label: int = int(row["Label"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)  # type: ignore[call-arg]
            image_val = augmented["image"]
            image_tensor = (
                image_val
                if isinstance(image_val, torch.Tensor)
                else torch.from_numpy(np.array(image_val))
            )
        else:
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        return image_tensor, torch.tensor(label, dtype=torch.long), image_id


class CheXpertDataset(Dataset[Any]):  # type: ignore[misc]
    """CheXpert zero-shot external validation dataset mapped to binary pneumothorax."""

    def __init__(
        self,
        csv_file: str,
        image_dir: str | None = None,
        transform: Any = None,
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        if not os.path.exists(csv_file):
            log.warning("CheXpert CSV not found; initializing empty dataset.", csv_file=csv_file)
            self.data_frame = pd.DataFrame(columns=["Path", "Label", "Image Index"])
            return

        self.data_frame = pd.read_csv(csv_file)

        if "Label" not in self.data_frame.columns:
            p_col = "Pneumothorax"
            if p_col in self.data_frame.columns:
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
        img_path = str(row["Image Path"])
        image_id = str(row["Image Index"])
        label_val = int(row["Label"])

        if not os.path.exists(img_path):
            guard_mock(f"[CheXpertDataset] image not found at '{img_path}'")
            if idx == 0:
                log.warning(
                    "MOCK MODE: missing CheXpert image replaced by black placeholder.",
                    img_path=img_path,
                )
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            image = Image.open(img_path).convert("RGB")

        out_img: Any = image
        if self.transform:
            augmented = self.transform(image=np.array(image))
            out_img = augmented["image"]

        return out_img, torch.tensor(label_val, dtype=torch.long), image_id
