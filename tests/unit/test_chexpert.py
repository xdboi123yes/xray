"""Unit tests for the CheXpert zero-shot external validation dataset."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pytest
import torch
from PIL import Image as PILImage
from tcrp.data.datasets import CheXpertDataset


def test_chexpert_dataset_label_mapping(tmp_path: Any) -> None:
    dummy_csv = os.path.join(tmp_path, "chexpert_metadata.csv")
    pd.DataFrame(
        {
            "Path": ["path1.png", "path2.png"],
            "Pneumothorax": [1.0, 0.0],
            "No Finding": [0.0, 1.0],
        }
    ).to_csv(dummy_csv, index=False)

    dataset = CheXpertDataset(csv_file=dummy_csv)
    assert len(dataset) == 2
    assert dataset.data_frame.iloc[0]["Label"] == 1
    assert dataset.data_frame.iloc[1]["Label"] == 0


def test_chexpert_dataset_keeps_all_frontals_as_screening_negatives(tmp_path: Any) -> None:
    dummy_csv = os.path.join(tmp_path, "chexpert_full.csv")
    pd.DataFrame(
        {
            "Path": ["p1.png", "p2.png", "p3.png", "p4.png", "p5.png"],
            "Pneumothorax": [1.0, 0.0, 0.0, float("nan"), -1.0],
            "No Finding": [0.0, 1.0, 0.0, 0.0, 0.0],
        }
    ).to_csv(dummy_csv, index=False)

    dataset = CheXpertDataset(csv_file=dummy_csv)
    assert len(dataset) == 5
    assert dataset.data_frame["Label"].tolist() == [1, 0, 0, 0, 0]


def test_chexpert_dataset_getitem_mock(tmp_path: Any, monkeypatch: Any) -> None:
    dummy_csv = os.path.join(tmp_path, "chexpert_metadata.csv")
    pd.DataFrame(
        {
            "Path": [os.path.join(tmp_path, "missing_img.png")],
            "Pneumothorax": [1.0],
            "No Finding": [0.0],
        }
    ).to_csv(dummy_csv, index=False)

    dataset = CheXpertDataset(csv_file=dummy_csv)

    monkeypatch.delenv("XRAY_ALLOW_MOCK", raising=False)
    with pytest.raises(RuntimeError):
        _ = dataset[0]

    monkeypatch.setenv("XRAY_ALLOW_MOCK", "1")
    image, label, image_id = dataset[0]
    assert isinstance(image, PILImage.Image | torch.Tensor)
    assert label.item() == 1
    assert image_id == "missing_img.png"
