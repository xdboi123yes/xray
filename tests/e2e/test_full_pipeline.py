"""End-to-End CPU research pipeline test for the TCR-P screening system.

Exercises the complete causal research seam on a small patient-grouped dummy
cohort without GPU or external downloads:
  1. Patient-Disjoint Partition (zero pairwise patient overlap + manifest)
  2. Mocked Synthetic Augmentation Batch generation and FID gate acceptance
  3. Tier 1 (MobileNetV2) and Tier 2 (EfficientNet-B4) training + checkpoint reload
  4. Validation Split post-hoc temperature scaling (T_cal)
  5. Disjoint Calibration Split split-conformal q_hat on temperature-scaled logits
  6. TCR-P inference exercising Tier 1 resolution, Escalation, and Human Deferral
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tcrp.data.augmentation import ClassicalAugmentation
from tcrp.data.datasets import NIHChestXrayDataset
from tcrp.data.splits import build_patient_disjoint_splits
from tcrp.data.synthetic import SyntheticBatchPipeline
from tcrp.models import ModelFactory, TieredSystem
from tcrp.routing import StaticThresholdRouter
from tcrp.training import CheckpointObserver, Trainer
from tcrp.uncertainty.calibration import calibrate_temperature
from tcrp.uncertainty.conformal import ConformalPredictor
from torch.utils.data import DataLoader


class _FakeGenerator:
    """In-memory fake generator adapter emitting deterministic synthetic radiographs."""

    def generate(
        self,
        init_image_path: str,
        num_variations: int = 3,
        strength: float = 0.3,
        severity: str | None = None,
        location: str | None = None,
    ) -> list[Image.Image]:
        base = np.asarray(Image.open(init_image_path).convert("RGB"), dtype=np.uint8)
        return [Image.fromarray(np.clip(base + i, 0, 255).astype(np.uint8)) for i in range(num_variations)]


class _FakeFID:
    """In-memory fake FID adapter returning a passing FID score."""

    def __init__(self, score: float = 28.5) -> None:
        self.score = score

    def compute_fid(self, real_dir: str, synthetic_dir: str, batch_size: int = 32) -> float:
        return self.score


def _create_dummy_nih_cohort(base_dir: Path, n_patients: int = 32) -> tuple[Path, Path]:
    """Create a patient-grouped dummy NIH CSV and PNG radiographs on disk."""
    img_dir = base_dir / "raw_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    rows: list[dict[str, str]] = []
    for p_idx in range(n_patients):
        patient_id = f"{p_idx + 1:08d}"
        is_pneumo = p_idx % 2 == 0
        finding = "Pneumothorax" if is_pneumo else "No Finding"
        # Two radiographs per patient to verify patient-level grouping
        for study in range(2):
            fname = f"{patient_id}_{study:03d}.png"
            arr = rng.integers(20, 235, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_dir / fname)
            rows.append({"Image Index": fname, "Finding Labels": finding})

    csv_path = base_dir / "Data_Entry_2017.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, img_dir


def test_e2e_research_pipeline(tmp_path: Path) -> None:
    """Verify full causal TCR-P pipeline end-to-end on CPU."""
    device = torch.device("cpu")
    csv_path, img_dir = _create_dummy_nih_cohort(tmp_path)
    processed_dir = tmp_path / "processed"

    # 1. Patient-Disjoint Partition
    summary = build_patient_disjoint_splits(
        csv_path=str(csv_path),
        image_dir=str(img_dir),
        output_dir=str(processed_dir),
        train_split=0.50,
        val_split=0.30,
        test_split=0.20,
        seed=42,
        max_ratio=5,
        calibration_fraction=0.50,
    )
    assert all(overlap == 0 for overlap in summary.pairwise_patient_overlaps.values())
    manifest = json.loads((processed_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["splits"].keys()) == {"train", "val", "calibration", "test"}

    # 2. Mocked Synthetic Augmentation Batch (strictly from train pneumothorax seeds)
    train_df = pd.read_csv(processed_dir / "train.csv")
    seed_fname = str(train_df[train_df["Label"] == 1].iloc[0]["Image Index"])
    synth_batch_dir = tmp_path / "synth_tmp"
    synth_accepted_dir = tmp_path / "synth_accepted"
    synth_rejected_dir = tmp_path / "synth_rejected"
    synth_batch_dir.mkdir(parents=True, exist_ok=True)

    synth_pipeline = SyntheticBatchPipeline(
        generator=_FakeGenerator(),
        fid_evaluator=_FakeFID(score=25.0),
        fid_threshold=50.0,
    )
    variations = synth_pipeline.generate_variations(str(img_dir / seed_fname), num_variations=2)
    for idx, img in enumerate(variations):
        img.save(synth_batch_dir / f"synth_{idx}.png")

    batch_record = synth_pipeline.evaluate_and_filter_batch(
        real_dir=str(img_dir),
        synthetic_batch_dir=str(synth_batch_dir),
        accepted_dir=str(synth_accepted_dir),
        rejected_dir=str(synth_rejected_dir),
    )
    assert batch_record.accepted is True
    assert len(batch_record.moved_files) == 2

    synth_csv = synth_accepted_dir / "synthetic.csv"
    pd.DataFrame(
        [{"Image Index": Path(p).name, "Finding Labels": "Pneumothorax"} for p in batch_record.moved_files]
    ).to_csv(synth_csv, index=False)

    # 3. Load datasets and train mini Tier 1 + Tier 2 models
    eval_tf = ClassicalAugmentation(image_size=64, is_training=False).pipeline
    train_ds = NIHChestXrayDataset(
        csv_file=str(processed_dir / "train.csv"),
        image_dir=str(img_dir),
        transform=eval_tf,
        use_synthetic=True,
        synthetic_dir=str(synth_accepted_dir),
        synthetic_csv=str(synth_csv),
    )
    val_ds = NIHChestXrayDataset(str(processed_dir / "val.csv"), str(img_dir), transform=eval_tf)
    cal_ds = NIHChestXrayDataset(
        str(processed_dir / "calibration.csv"), str(img_dir), transform=eval_tf
    )
    test_ds = NIHChestXrayDataset(str(processed_dir / "test.csv"), str(img_dir), transform=eval_tf)
    assert len(train_ds) == len(train_df) + 2

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    cal_loader = DataLoader(cal_ds, batch_size=4, shuffle=False)

    t1_ckpt = str(tmp_path / "tier1_best.pth")
    t1_model = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    t1_trainer = Trainer(
        model=t1_model,
        optimizer=torch.optim.Adam(t1_model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config={"training": {"epochs": 1, "seed": 42}},
        use_amp=False,
    )
    t1_trainer.add_observer(CheckpointObserver(checkpoint_path=t1_ckpt))
    t1_trainer.train(train_loader, val_loader)
    assert Path(t1_ckpt).exists()

    t2_ckpt = str(tmp_path / "tier2_best.pth")
    t2_model = ModelFactory.create("efficientnet_b4", num_classes=2, pretrained=False).to(device)
    t2_trainer = Trainer(
        model=t2_model,
        optimizer=torch.optim.Adam(t2_model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config={"training": {"epochs": 1, "seed": 42}},
        use_amp=False,
    )
    t2_trainer.add_observer(CheckpointObserver(checkpoint_path=t2_ckpt))
    t2_trainer.train(train_loader, val_loader)
    assert Path(t2_ckpt).exists()

    # Reload weights from disk
    t1_reloaded = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    t1_reloaded.load_state_dict(
        torch.load(t1_ckpt, map_location=device, weights_only=False)["model_state_dict"]
    )
    t1_reloaded.eval()

    t2_reloaded = ModelFactory.create("efficientnet_b4", num_classes=2, pretrained=False).to(device)
    t2_reloaded.load_state_dict(
        torch.load(t2_ckpt, map_location=device, weights_only=False)["model_state_dict"]
    )
    t2_reloaded.eval()

    # 4. Validation Split temperature scaling BEFORE Calibration Split conformal q_hat
    temp_model, t_cal = calibrate_temperature(t2_reloaded, val_loader, device=device)
    assert t_cal > 0.0

    # 5. Disjoint Calibration Split conformal q_hat on temperature-scaled model
    cp_path = str(tmp_path / "q_hat.pt")
    cp = ConformalPredictor(alpha=0.1)
    q_hat = cp.calibrate(temp_model, cal_loader, device)
    assert 0.0 <= q_hat <= 1.0
    cp.save(cp_path)

    reloaded_cp = ConformalPredictor(alpha=0.1)
    reloaded_cp.load(cp_path)
    assert reloaded_cp.q_hat == q_hat

    # 6. TCR-P inference: verify Escalation to Tier 2 and Human Deferral metadata
    sample_x, sample_y, sample_id = test_ds[0]
    sample_batch = sample_x.unsqueeze(0)

    escalating_system = TieredSystem(
        tier1_model=t1_reloaded,
        tier2_model=t2_reloaded,
        config={"model": {"mc_dropout_passes": 2, "tta_passes": 2}},
        conformal_predictor=reloaded_cp,
        router=StaticThresholdRouter(threshold=1.01),
    )
    log_csv = str(tmp_path / "routing_log.csv")
    esc_result = escalating_system.route(
        sample_batch,
        image_id=sample_id,
        log_path=log_csv,
        ground_truth=int(sample_y.item()),
    )
    assert esc_result.tier_used == 2
    assert esc_result.prediction in ("Pneumothorax", "No Finding")
    assert esc_result.mc_variance is not None and esc_result.mc_variance >= 0.0
    assert esc_result.conformal_set is not None and len(esc_result.conformal_set) >= 1
    assert isinstance(esc_result.flagged_for_review, bool)
    assert Path(log_csv).exists()

    # Verify Tier-1 fast resolution path
    fast_system = TieredSystem(
        tier1_model=t1_reloaded,
        tier2_model=t2_reloaded,
        config={"model": {"mc_dropout_passes": 2, "tta_passes": 2}},
        conformal_predictor=reloaded_cp,
        router=StaticThresholdRouter(threshold=0.0),
    )
    fast_result = fast_system.route(sample_batch, image_id=sample_id)
    assert fast_result.tier_used == 1
    assert fast_result.mc_variance is None
