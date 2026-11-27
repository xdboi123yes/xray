"""End-to-End CPU research pipeline test for the TCR-P screening system.

Exercises the complete causal research seam on a small patient-grouped dummy
cohort without GPU or external downloads:
  1. Patient-Disjoint Partition (zero pairwise patient overlap + manifest)
  2. Mocked Synthetic Augmentation Batch generation and FID gate acceptance
  3. Tier 1 (MobileNetV2) and Tier 2 (EfficientNet-B4) training + checkpoint reload
  4. Validation Split post-hoc temperature scaling (T_cal -> temperature.pt)
  5. Disjoint Calibration Split split-conformal q_hat on temperature-scaled logits
     + marginal/class-conditional coverage and set-size diagnostics
  6. TCR-P inference exercising Tier 1 resolution, Escalation, and Human Deferral
  7. Provenance manifest build, SHA-256 verification, and tamper detection
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from tcrp.data.augmentation import ClassicalAugmentation
from tcrp.data.datasets import NIHChestXrayDataset
from tcrp.data.splits import build_patient_disjoint_splits, verify_patient_disjointness
from tcrp.data.synthetic import SyntheticBatchPipeline, SyntheticGenerationConfig
from tcrp.evaluation.provenance import build_provenance_manifest, verify_provenance_manifest
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
        seed: int | None = None,
    ) -> list[Image.Image]:
        base = np.asarray(Image.open(init_image_path).convert("RGB"), dtype=np.uint8)
        offset = 0 if seed is None else (seed % 7)
        return [
            Image.fromarray(np.clip(base.astype(int) + i + offset, 0, 255).astype(np.uint8))
            for i in range(num_variations)
        ]


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
    """Verify full causal TCR-P pipeline end-to-end on CPU including provenance verification."""
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
    split_manifest_path = processed_dir / "split_manifest.json"
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["splits"].keys()) == {"train", "val", "calibration", "test"}

    # 2. Mocked multi-batch Synthetic Augmentation Batch (strictly from train pneumothorax seeds)
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    cal_df = pd.read_csv(processed_dir / "calibration.csv")
    assert verify_patient_disjointness({"val": val_df, "calibration": cal_df}) == {
        "calibration__val": 0
    }

    synth_dir = tmp_path / "synthetic"
    synth_pipeline = SyntheticBatchPipeline(
        generator=_FakeGenerator(),
        fid_evaluator=_FakeFID(score=25.0),
        fid_threshold=50.0,
    )
    synth_summary = synth_pipeline.run_multi_batch(
        SyntheticGenerationConfig(
            train_csv=str(processed_dir / "train.csv"),
            real_images_dir=str(img_dir),
            output_dir=str(synth_dir),
            seed_scope="all",
            batch_size=4,
            num_variations=1,
            max_seeds=2,
            fid_threshold=50.0,
            seed=42,
            resume=False,
        )
    )
    assert synth_summary.batches_accepted == 1
    assert synth_summary.accepted_images == 2
    synth_accepted_dir = synth_dir / "accepted"
    synth_csv = synth_dir / "synthetic_metadata.csv"
    fid_log_csv = synth_dir / "fid_log.csv"
    assert synth_csv.is_file() and fid_log_csv.is_file()

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
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    t1_ckpt = tmp_path / "tier1_best.pth"
    t1_model = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    t1_trainer = Trainer(
        model=t1_model,
        optimizer=torch.optim.Adam(t1_model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config={"training": {"epochs": 1, "seed": 42}},
        use_amp=False,
    )
    t1_trainer.add_observer(CheckpointObserver(checkpoint_path=str(t1_ckpt)))
    t1_trainer.train(train_loader, val_loader)
    assert t1_ckpt.exists()

    t2_ckpt = tmp_path / "tier2_best.pth"
    t2_model = ModelFactory.create("efficientnet_b4", num_classes=2, pretrained=False).to(device)
    t2_trainer = Trainer(
        model=t2_model,
        optimizer=torch.optim.Adam(t2_model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config={"training": {"epochs": 1, "seed": 42}},
        use_amp=False,
    )
    t2_trainer.add_observer(CheckpointObserver(checkpoint_path=str(t2_ckpt)))
    t2_trainer.train(train_loader, val_loader)
    assert t2_ckpt.exists()

    # Reload weights from disk
    t1_reloaded = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    t1_reloaded.load_state_dict(
        torch.load(str(t1_ckpt), map_location=device, weights_only=False)["model_state_dict"]
    )
    t1_reloaded.eval()

    t2_reloaded = ModelFactory.create("efficientnet_b4", num_classes=2, pretrained=False).to(device)
    t2_reloaded.load_state_dict(
        torch.load(str(t2_ckpt), map_location=device, weights_only=False)["model_state_dict"]
    )
    t2_reloaded.eval()

    # 4. Validation Split temperature scaling BEFORE Calibration Split conformal q_hat
    _, t_cal = calibrate_temperature(t2_reloaded, val_loader, device=device)
    assert t_cal > 0.0
    temp_path = tmp_path / "temperature.pt"
    torch.save({"temperature": float(t_cal), "split": "val"}, temp_path)

    # 5. Disjoint Calibration Split conformal q_hat consuming Validation-Split T_cal
    cp_path = tmp_path / "q_hat.pt"
    cp = ConformalPredictor(alpha=0.1)
    q_hat = cp.calibrate(t2_reloaded, cal_loader, device, temperature=t_cal)
    assert 0.0 <= q_hat <= 1.0
    assert cp.temperature == pytest.approx(t_cal)
    assert cp.last_calibration_report is not None
    assert 0.0 <= cp.last_calibration_report.marginal_coverage <= 1.0
    assert 0.0 <= cp.last_calibration_report.singleton_rate <= 1.0
    assert 0.0 <= cp.last_calibration_report.two_class_abstention_rate <= 1.0
    assert 0.0 <= cp.last_calibration_report.empty_fallback_rate <= 1.0
    cp.save(str(cp_path))

    reloaded_cp = ConformalPredictor(alpha=0.1)
    reloaded_cp.load(str(cp_path))
    assert reloaded_cp.q_hat == pytest.approx(q_hat)
    assert reloaded_cp.temperature == pytest.approx(t_cal)

    test_cp_report = reloaded_cp.evaluate(t2_reloaded, test_loader, device, split_name="test")
    assert test_cp_report.n_samples == len(test_ds)
    assert test_cp_report.singleton_rate + test_cp_report.two_class_abstention_rate == pytest.approx(
        1.0
    )
    cp_summary_path = tmp_path / "conformal_calibration_summary.json"
    cp_summary_path.write_text(json.dumps(test_cp_report.to_dict(), indent=2), encoding="utf-8")

    # 6. TCR-P inference: verify Tier-1 fast resolution, Escalation to Tier 2, and Human Deferral
    sample_x, sample_y, sample_id = test_ds[0]
    sample_batch = sample_x.unsqueeze(0)

    escalating_system = TieredSystem(
        tier1_model=t1_reloaded,
        tier2_model=t2_reloaded,
        config={"model": {"mc_dropout_passes": 2, "tta_passes": 2}},
        conformal_predictor=reloaded_cp,
        router=StaticThresholdRouter(threshold=1.01),
        temperature=t_cal,
    )
    log_csv = tmp_path / "routing_log.csv"
    esc_result = escalating_system.route(
        sample_batch,
        image_id=sample_id,
        log_path=str(log_csv),
        ground_truth=int(sample_y.item()),
    )
    assert esc_result.tier_used == 2
    assert esc_result.prediction in ("Pneumothorax", "No Finding")
    assert esc_result.mc_variance is not None and esc_result.mc_variance >= 0.0
    assert esc_result.conformal_set is not None and len(esc_result.conformal_set) >= 1
    assert isinstance(esc_result.flagged_for_review, bool)
    assert log_csv.exists()

    # Verify explicit Human Deferral trigger (strict variance threshold or two-class conformal set)
    deferral_cp = ConformalPredictor(alpha=0.05)
    deferral_cp.q_hat = 1.0  # threshold = 0.0 -> both classes included -> 2-class abstention
    deferring_system = TieredSystem(
        tier1_model=t1_reloaded,
        tier2_model=t2_reloaded,
        config={"model": {"mc_dropout_passes": 2, "tta_passes": 2}},
        conformal_predictor=deferral_cp,
        router=StaticThresholdRouter(threshold=1.01),
        temperature=t_cal,
    )
    defer_result = deferring_system.route(sample_batch, image_id=sample_id)
    assert defer_result.tier_used == 2
    assert defer_result.conformal_set == ["No Finding", "Pneumothorax"]
    assert defer_result.flagged_for_review is True

    # Verify Tier-1 fast resolution path
    fast_system = TieredSystem(
        tier1_model=t1_reloaded,
        tier2_model=t2_reloaded,
        config={"model": {"mc_dropout_passes": 2, "tta_passes": 2}},
        conformal_predictor=reloaded_cp,
        router=StaticThresholdRouter(threshold=0.0),
        temperature=t_cal,
    )
    fast_result = fast_system.route(sample_batch, image_id=sample_id)
    assert fast_result.tier_used == 1
    assert fast_result.mc_variance is None

    # 7. Provenance manifest verification (build, verify, and confirm tamper detection)
    provenance_manifest_path = tmp_path / "rerun_manifest.json"
    tracked_artifacts = [
        synth_csv,
        fid_log_csv,
        t1_ckpt,
        t2_ckpt,
        temp_path,
        cp_path,
        cp_summary_path,
        log_csv,
    ]
    build_provenance_manifest(
        output_path=provenance_manifest_path,
        artifact_paths=tracked_artifacts,
        protocol="patient-disjoint-v1",
        split_manifest_path=split_manifest_path,
        calibration_metadata={
            "temperature_split": "val",
            "conformal_split": "calibration",
            "temperature_before_conformal": True,
            "temperature": float(t_cal),
            "q_hat": float(q_hat),
        },
        base_dir=tmp_path,
    )
    ver = verify_provenance_manifest(
        manifest_path=provenance_manifest_path,
        base_dir=tmp_path,
        expected_protocol="patient-disjoint-v1",
        required_artifacts=[
            "processed/split_manifest.json",
            "synthetic/fid_log.csv",
            "temperature.pt",
            "q_hat.pt",
            "conformal_calibration_summary.json",
        ],
        strict=True,
    )
    assert ver.valid is True
    assert ver.verified_artifacts == len(tracked_artifacts) + 1  # + split_manifest.json

    # Tampering with any tracked artifact must fail the provenance gate
    cp_summary_path.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_provenance_manifest(
            manifest_path=provenance_manifest_path,
            base_dir=tmp_path,
            expected_protocol="patient-disjoint-v1",
            strict=True,
        )
