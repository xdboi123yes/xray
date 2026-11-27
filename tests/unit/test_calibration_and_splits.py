"""Unit tests for Patient-Disjoint Partitioning, NIH datasets, calibration, conformal, and training config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from pydantic import ValidationError
from tcrp.config import Settings, get_settings, guard_mock
from tcrp.data.datasets import NIHChestXrayDataset
from tcrp.data.splits import build_patient_disjoint_splits, verify_patient_disjointness
from tcrp.training import (
    ONNXModelExporter,
    TrainingConfig,
    get_class_weighted_loss,
    get_scheduler,
)
from tcrp.uncertainty.calibration import (
    ModelWithTemperature,
    calibrate_temperature,
    compute_ece,
    plot_reliability_diagram,
)
from tcrp.uncertainty.conformal import ConformalPredictor
from torch.utils.data import DataLoader, TensorDataset


def test_verify_patient_disjointness_detects_overlap() -> None:
    df_a = pd.DataFrame({"Image Index": ["00000001_000.png", "00000002_000.png"]})
    df_b = pd.DataFrame({"Image Index": ["00000003_000.png", "00000004_000.png"]})
    overlaps = verify_patient_disjointness({"train": df_a, "val": df_b})
    assert overlaps == {"train__val": 0}

    df_leaky = pd.DataFrame({"Image Index": ["00000002_001.png", "00000005_000.png"]})
    with pytest.raises(ValueError, match="Patient-Disjoint Partition violation"):
        verify_patient_disjointness({"train": df_a, "val": df_leaky})


def test_build_patient_disjoint_splits_and_nih_dataset(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows: list[dict[str, str]] = []
    for i in range(24):
        pid = f"{i + 1:08d}"
        finding = "Pneumothorax|Effusion" if i % 3 == 0 else "No Finding"
        for s in range(2):
            fname = f"{pid}_{s:03d}.png"
            Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(img_dir / fname)
            rows.append({"Image Index": fname, "Finding Labels": finding})
    # Excluded non-pneumothorax pathology row must be dropped
    Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(img_dir / "99999999_000.png")
    rows.append({"Image Index": "99999999_000.png", "Finding Labels": "Cardiomegaly"})

    csv_path = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out_dir = tmp_path / "processed"
    summary = build_patient_disjoint_splits(
        csv_path=str(csv_path),
        image_dir=str(img_dir),
        output_dir=str(out_dir),
        train_split=0.50,
        val_split=0.30,
        test_split=0.20,
        seed=42,
        calibration_fraction=0.50,
    )
    assert all(v == 0 for v in summary.pairwise_patient_overlaps.values())
    total_samples = sum(c.samples for c in summary.splits.values())
    assert total_samples == 48  # Cardiomegaly excluded

    ds = NIHChestXrayDataset(str(out_dir / "train.csv"), image_dir=str(img_dir))
    img_t, label_t, img_id = ds[0]
    assert img_t.shape == (3, 32, 32)
    assert label_t.item() in (0, 1)
    assert img_id.endswith(".png")

    loss_fn = get_class_weighted_loss(str(out_dir / "train.csv"))
    assert loss_fn.weight is not None
    assert pytest.approx(float(loss_fn.weight.sum().item()), abs=1e-4) == 2.0


def test_temperature_calibration_and_conformal_prediction(tmp_path: Path) -> None:
    torch.manual_seed(0)

    class _LinearLogitModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = _LinearLogitModel()
    x_val = torch.randn(16, 4)
    y_val = torch.randint(0, 2, (16,))
    val_loader = DataLoader(TensorDataset(x_val, y_val, torch.arange(16)), batch_size=4)

    cal_model, t_cal = calibrate_temperature(model, val_loader, device=torch.device("cpu"))
    assert t_cal > 0.0

    probs = torch.softmax(cal_model(x_val), dim=1)[:, 1].detach().numpy()
    ece = compute_ece(probs, y_val.numpy(), n_bins=5)
    assert 0.0 <= ece <= 1.0

    fig_path = plot_reliability_diagram(
        probs, y_val.numpy(), n_bins=5, save_path=str(tmp_path / "rel.png")
    )
    assert fig_path is not None and Path(fig_path).exists()

    cp = ConformalPredictor(alpha=0.1)
    with pytest.raises(ValueError, match="not calibrated"):
        cp.predict_set(torch.tensor([[0.6, 0.4]]))

    q_hat = cp.calibrate(model, val_loader, device=torch.device("cpu"), temperature=t_cal)
    assert 0.0 <= q_hat <= 1.0
    pred_set = cp.predict_set(torch.tensor([[0.8, 0.2]]))
    assert len(pred_set) >= 1

    save_path = str(tmp_path / "cp.pt")
    cp.save(save_path)
    loaded = ConformalPredictor(alpha=0.2)
    loaded.load(save_path)
    assert loaded.q_hat == q_hat
    assert loaded.alpha == 0.1


def test_training_config_settings_scheduler_and_guard_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = TrainingConfig(backbone="efficientnet_b4", run_name="Run_01", epochs=10)
    assert cfg.batch_size == 32
    with pytest.raises(ValidationError):
        TrainingConfig(backbone="efficientnet_b4", run_name="Bad", batch_size=0)

    settings = get_settings()
    assert isinstance(settings, Settings)
    model = nn.Linear(4, 2)
    sched = get_scheduler(torch.optim.Adam(model.parameters(), lr=1e-3), settings)
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    monkeypatch.delenv("XRAY_ALLOW_MOCK", raising=False)
    with pytest.raises(RuntimeError, match="refusing to fabricate"):
        guard_mock("unit-test")
    monkeypatch.setenv("XRAY_ALLOW_MOCK", "1")
    guard_mock("unit-test")


def test_onnx_exporter_invokes_torch_onnx_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_export(model: nn.Module, args: object, f: str, **kwargs: object) -> None:
        calls.append({"model": model, "args": args, "f": f, "kwargs": kwargs})
        Path(f).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", _fake_export)
    tiny = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    onnx_out = tmp_path / "tiny.onnx"
    ONNXModelExporter.export(tiny, str(onnx_out), input_size=(1, 3, 8, 8))
    assert onnx_out.exists()
    assert len(calls) == 1
    assert calls[0]["f"] == str(onnx_out)


def test_compute_ece_includes_exact_zero_probability() -> None:
    # A sample with probability 0.0 and true label 1 has error |0.0 - 1.0| = 1.0
    assert compute_ece([0.0], [1], n_bins=15) == pytest.approx(1.0)
    assert compute_ece([0.0, 1.0], [0, 1], n_bins=15) == pytest.approx(0.0)


def test_model_with_temperature_clamps_non_positive_temperature() -> None:
    base = nn.Linear(2, 2)
    wrapped = ModelWithTemperature(base)
    logits = torch.tensor([[1.0, -2.0]])

    with torch.no_grad():
        wrapped.temperature.fill_(-0.5)
    scaled = wrapped.temperature_scale(logits)
    assert torch.isfinite(scaled).all()
    assert scaled[0, 0].item() > 0.0
    assert scaled[0, 1].item() < 0.0


def test_build_patient_disjoint_splits_missing_image_dir_requires_opt_out(tmp_path: Path) -> None:
    rows = []
    for i in range(24):
        pid = f"{i + 1:08d}"
        finding = "Pneumothorax" if i % 2 == 0 else "No Finding"
        rows.append({"Image Index": f"{pid}_000.png", "Finding Labels": finding})
    csv_path = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    missing_dir = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="image_dir"):
        build_patient_disjoint_splits(
            csv_path=str(csv_path),
            image_dir=missing_dir,
            output_dir=str(tmp_path / "out_fail"),
        )

    summary = build_patient_disjoint_splits(
        csv_path=str(csv_path),
        image_dir=missing_dir,
        output_dir=str(tmp_path / "out_ok"),
        train_split=0.50,
        val_split=0.30,
        test_split=0.20,
        calibration_fraction=0.50,
        verify_images=False,
    )
    assert sum(c.samples for c in summary.splits.values()) == 24
