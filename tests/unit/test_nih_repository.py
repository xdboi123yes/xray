"""Unit tests for NIHRepository (NIH dataset loader repository)."""

from __future__ import annotations

import os

import pandas as pd
import torch

from infrastructure.data.nih_repository import NIHDataset, NIHRepository


def test_nih_repository_empty_splits() -> None:
    """Verify that train, val, and calibration datasets return EmptyMockDataset."""
    repo = NIHRepository()
    assert len(repo.get_train_dataset()) == 0
    assert len(repo.get_val_dataset()) == 0
    assert len(repo.get_calibration_dataset()) == 0

def test_nih_dataset_mock_fallback() -> None:
    """Verify that NIHDataset handles missing files gracefully with high-fidelity mock."""
    dummy_csv = "tests/dummy_nih.csv"
    df = pd.DataFrame([
        {"Path": "non_existent_img.png", "Pneumothorax": 1.0, "No Finding": 0.0}
    ])
    df.to_csv(dummy_csv, index=False)

    try:
        dataset = NIHDataset(csv_file=dummy_csv)
        assert len(dataset) == 1
        
        image, label, image_id = dataset[0]
        assert isinstance(image, torch.Tensor) or hasattr(image, "size") # PIL or Tensor
        assert label.item() == 1
        assert image_id == "non_existent_img.png"
    finally:
        if os.path.exists(dummy_csv):
            os.remove(dummy_csv)
