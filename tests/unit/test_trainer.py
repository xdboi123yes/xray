"""Unit tests for the Trainer engine.

Verifies creation, custom optimizer and criteria configurations, and observer
registrations for the Trainer class.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from tcrp.config import Settings
from tcrp.models import BaseClassifier, Tier1MobileNet
from tcrp.training import MLflowObserver, Trainer, TrainingConfig, TrainingObserver, train_model
from torch.utils.data import DataLoader, TensorDataset


class MockObserver(TrainingObserver):
    """Mock observer to verify event registrations."""

    def __init__(self) -> None:
        self.start_called = False

    def on_train_start(self, trainer: Any) -> None:
        self.start_called = True


def test_trainer_init(mock_config: dict[str, Any]) -> None:
    """Test Trainer initialization and state setups.

    Args:
        mock_config: Pytest fixture representing mock parameters.
    """
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=mock_config,
        use_amp=False,
        accumulate_grad_batches=2,
        gradient_clip_val=1.5,
    )

    assert trainer.model is model
    assert trainer.optimizer is optimizer
    assert trainer.criterion is criterion
    assert trainer.device == device
    assert trainer.accumulate_grad_batches == 2
    assert trainer.gradient_clip_val == 1.5
    assert not trainer.use_amp


def test_trainer_observer_registration(mock_config: dict[str, Any]) -> None:
    """Test that observers can be registered cleanly to the Trainer engine.

    Args:
        mock_config: Pytest fixture representing mock parameters.
    """
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=mock_config,
    )

    obs = MockObserver()
    trainer.add_observer(obs)

    assert obs in trainer._observers


class _ConstantLogitClassifier(BaseClassifier):
    """Tiny classifier outputting fixed logits [0, 0] (exact CrossEntropyLoss = ln 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 2, bias=True)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.view(x.size(0), -1)[:, :4])


def test_trainer_epoch_loss_accurate_with_drop_last_and_empty_loader() -> None:
    model = _ConstantLogitClassifier()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        config={"training": {"epochs": 1}},
        use_amp=False,
    )

    # 3 samples with batch_size=2 and drop_last=True -> 2 processed samples, each loss = ln(2)
    ds = TensorDataset(torch.zeros(3, 4), torch.tensor([0, 1, 0]))
    train_loader = DataLoader(ds, batch_size=2, drop_last=True)
    empty_loader = DataLoader(TensorDataset(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)), batch_size=2)

    train_metrics = trainer._train_epoch(train_loader)
    assert train_metrics["train_loss"] == pytest.approx(float(np.log(2.0)), abs=1e-5)

    val_metrics = trainer._val_epoch(empty_loader)
    assert val_metrics["val_loss"] == 0.0


def test_train_model_loads_accepted_synthetic_metadata(tmp_path: Path) -> None:
    img_dir = tmp_path / "raw_images"
    synth_root = tmp_path / "synthetic"
    accepted_dir = synth_root / "accepted"
    models_dir = tmp_path / "models"
    for d in (img_dir, accepted_dir, models_dir):
        d.mkdir(parents=True)

    rows = []
    for i in range(4):
        fname = f"0000000{i}_000.png"
        Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8)).save(img_dir / fname)
        rows.append({"Image Index": fname, "Finding Labels": "Pneumothorax" if i % 2 == 0 else "No Finding", "Label": i % 2})
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    pd.DataFrame(rows[:2]).to_csv(val_csv, index=False)

    # Accepted synthetic radiographs matching generate_synthetics.py layout
    synth_rows = []
    for j in range(2):
        sname = f"synth_{j}.png"
        Image.fromarray(np.full((32, 32, 3), 150, dtype=np.uint8)).save(accepted_dir / sname)
        synth_rows.append({"Image Index": sname, "Finding Labels": "Pneumothorax"})
    pd.DataFrame(synth_rows).to_csv(synth_root / "synthetic_metadata.csv", index=False)

    settings = Settings(
        data={"image_size": 32},
        paths={"models": str(models_dir), "data_synthetic": str(synth_root)},
    )
    cfg = TrainingConfig(
        backbone="mobilenet_v2",
        run_name="TestSynthWire",
        batch_size=2,
        epochs=1,
        use_synthetic=True,
        use_classical_augmentation=False,
    )

    seen_train_len: list[int] = []
    orig_train = Trainer.train

    def _spy_train(self: Trainer, train_loader: DataLoader[Any], val_loader: DataLoader[Any], resume_from: str | None = None) -> dict[str, float]:
        seen_train_len.append(len(train_loader.dataset))  # type: ignore[arg-type]
        return orig_train(self, train_loader, val_loader, resume_from=resume_from)

    with patch.object(Trainer, "train", _spy_train):
        train_model(
            cfg,
            train_csv=str(train_csv),
            val_csv=str(val_csv),
            dry_run=True,
            settings=settings,
            device=torch.device("cpu"),
            pretrained=False,
            image_dir=str(img_dir),
            enable_external_observers=False,
        )

    assert seen_train_len == [6]  # 4 real + 2 accepted synthetic


def test_train_model_persists_training_history_and_wall_clock(tmp_path: Path) -> None:
    import json

    img_dir = tmp_path / "raw_images"
    models_dir = tmp_path / "models"
    img_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    rows = []
    for i in range(4):
        fname = f"0000001{i}_000.png"
        Image.fromarray(np.full((32, 32, 3), 80 + i * 20, dtype=np.uint8)).save(img_dir / fname)
        rows.append(
            {
                "Image Index": fname,
                "Finding Labels": "Pneumothorax" if i % 2 == 0 else "No Finding",
                "Label": i % 2,
            }
        )
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    pd.DataFrame(rows[:2]).to_csv(val_csv, index=False)

    settings = Settings(
        data={"image_size": 32},
        paths={"models": str(models_dir)},
    )
    cfg = TrainingConfig(
        backbone="mobilenet_v2",
        run_name="HistorySmoke",
        batch_size=2,
        epochs=2,
        use_synthetic=False,
        use_classical_augmentation=False,
    )

    result = train_model(
        cfg,
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        dry_run=False,
        settings=settings,
        device=torch.device("cpu"),
        pretrained=False,
        image_dir=str(img_dir),
        enable_external_observers=False,
    )

    history_path = models_dir / "HistorySmoke" / "training_history.json"
    assert history_path.is_file()
    assert result["history_path"] == str(history_path)
    assert float(result["wall_clock_seconds"]) >= 0.0

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert payload["run_name"] == "HistorySmoke"
    assert payload["backbone"] == "mobilenet_v2"
    assert payload["use_synthetic"] is False
    assert payload["train_samples"] == 4
    assert payload["val_samples"] == 2
    assert payload["epochs_completed"] == 2
    assert len(payload["epochs"]) == 2
    assert payload["wall_clock_seconds"] >= 0.0
    for idx, entry in enumerate(payload["epochs"], start=1):
        assert entry["epoch"] == idx
        assert "train_loss" in entry and "val_auc" in entry and "epoch_seconds" in entry


def test_mlflow_observer_only_ends_runs_it_started() -> None:
    dummy_trainer = MagicMock(config={"training": {"epochs": 1}})

    # Case 1: external run already active -> must NOT close it on train end
    with patch("tcrp.training.observers.mlflow") as mock_mlflow:
        mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="outer_run"))
        obs = MLflowObserver(run_name="inner")
        obs.on_train_start(dummy_trainer)
        obs.on_train_end(dummy_trainer)
        mock_mlflow.start_run.assert_not_called()
        mock_mlflow.end_run.assert_not_called()

    # Case 2: no active run -> starts and closes its own run
    with patch("tcrp.training.observers.mlflow") as mock_mlflow:
        started_run = MagicMock(info=MagicMock(run_id="owned_run"))
        mock_mlflow.active_run.side_effect = [None, started_run]
        mock_mlflow.start_run.return_value = started_run
        obs = MLflowObserver(run_name="owned")
        obs.on_train_start(dummy_trainer)
        obs.on_train_end(dummy_trainer)
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()
