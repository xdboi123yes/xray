"""Unit tests for the CheXpert dataset and repository components.

Verifies correct binary label mapping, image dimensions, mock fallbacks,
and BaseRepository splits interface compliance.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pytest
import torch

from core.interfaces.base_repository import BaseRepository
from infrastructure.data.chexpert_dataset import CheXpertDataset
from infrastructure.data.chexpert_repository import CheXpertRepository


def test_chexpert_dataset_label_mapping(tmp_path: Any) -> None:
    """Test that CheXpertDataset maps multi-label headers to clean binary outputs.

    Args:
        tmp_path: Pytest temporary directory path.
    """
    dummy_csv = os.path.join(tmp_path, "chexpert_metadata.csv")

    # 1. Pneumothorax = 1.0, No Finding = 0.0 -> Label 1
    # 2. Pneumothorax = 0.0, No Finding = 1.0 -> Label 0
    df = pd.DataFrame(
        {
            "Path": ["path1.png", "path2.png"],
            "Pneumothorax": [1.0, 0.0],
            "No Finding": [0.0, 1.0],
        }
    )
    df.to_csv(dummy_csv, index=False)

    dataset = CheXpertDataset(csv_file=dummy_csv)
    assert len(dataset) == 2

    # Verify index mapping
    assert dataset.data_frame.iloc[0]["Label"] == 1
    assert dataset.data_frame.iloc[1]["Label"] == 0


def test_chexpert_dataset_getitem_mock(tmp_path: Any, monkeypatch: Any) -> None:
    """Missing images fail loudly in a real run; only XRAY_ALLOW_MOCK=1 permits a placeholder.

    Args:
        tmp_path: Pytest temporary directory path.
        monkeypatch: Pytest environment patching fixture.
    """
    dummy_csv = os.path.join(tmp_path, "chexpert_metadata.csv")
    df = pd.DataFrame(
        {
            "Path": [os.path.join(tmp_path, "missing_img.png")],
            "Pneumothorax": [1.0],
            "No Finding": [0.0],
        }
    )
    df.to_csv(dummy_csv, index=False)

    dataset = CheXpertDataset(csv_file=dummy_csv)

    # Without the opt-in, a missing image must raise rather than fabricate a black frame.
    monkeypatch.delenv("XRAY_ALLOW_MOCK", raising=False)
    with pytest.raises(RuntimeError):
        _ = dataset[0]

    # With XRAY_ALLOW_MOCK=1, the dry-run black placeholder is allowed.
    monkeypatch.setenv("XRAY_ALLOW_MOCK", "1")
    image, label, image_id = dataset[0]

    assert image is not None
    # Depending on transforms, image is either PIL.Image or torch.Tensor
    from PIL import Image as PILImage

    assert isinstance(image, PILImage.Image | torch.Tensor)
    assert label.item() == 1
    assert image_id == "missing_img.png"


def test_chexpert_repository_splits(tmp_path: Any) -> None:
    """Test that CheXpertRepository conforms to BaseRepository interface.

    Args:
        tmp_path: Pytest temporary directory path.
    """
    dummy_csv = os.path.join(tmp_path, "chexpert_metadata.csv")
    df = pd.DataFrame(
        {
            "Path": ["path1.png"],
            "Pneumothorax": [1.0],
            "No Finding": [0.0],
        }
    )
    df.to_csv(dummy_csv, index=False)

    repo = CheXpertRepository(test_csv=dummy_csv)
    assert isinstance(repo, BaseRepository)

    # Empty mock containers
    assert len(repo.get_train_dataset()) == 0
    assert len(repo.get_val_dataset()) == 0
    assert len(repo.get_calibration_dataset()) == 0

    # Populated split
    test_ds = repo.get_test_dataset()
    assert isinstance(test_ds, CheXpertDataset)
    assert len(test_ds) == 1
