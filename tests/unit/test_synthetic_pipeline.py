"""Unit tests for the deepened SyntheticBatchPipeline seam with fake adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from tcrp.data.synthetic import (
    BatchFIDRecord,
    MultiBatchGenerationSummary,
    PyTorchFIDAdapter,
    SyntheticBatchPipeline,
    SyntheticGenerationConfig,
)


class _StubGenerator:
    """Fake generator returning deterministic images and recording prompt args."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        init_image_path: str,
        num_variations: int = 3,
        strength: float = 0.3,
        severity: str | None = None,
        location: str | None = None,
    ) -> list[Image.Image]:
        self.calls.append(
            {
                "init_image_path": init_image_path,
                "num_variations": num_variations,
                "strength": strength,
                "severity": severity,
                "location": location,
            }
        )
        return [
            Image.fromarray(np.full((64, 64, 3), fill_value=10 * (i + 1), dtype=np.uint8))
            for i in range(num_variations)
        ]


class _StubFID:
    """Fake FID evaluator returning a preset FID score."""

    def __init__(self, score: float) -> None:
        self.score = score

    def compute_fid(self, real_dir: str, synthetic_dir: str, batch_size: int = 32) -> float:
        return self.score


def test_synthetic_pipeline_generates_and_handles_missing_seed(tmp_path: Path) -> None:
    seed_img = tmp_path / "seed.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(seed_img)

    gen = _StubGenerator()
    pipeline = SyntheticBatchPipeline(generator=gen, fid_evaluator=_StubFID(20.0), fid_threshold=50.0)

    imgs = pipeline.generate_variations(
        str(seed_img), num_variations=3, severity="mild", location="apex"
    )
    assert len(imgs) == 3
    assert gen.calls[0]["severity"] == "mild"
    assert gen.calls[0]["location"] == "apex"

    assert pipeline.generate_variations(str(tmp_path / "does_not_exist.png")) == []


def test_synthetic_pipeline_fid_gate_accept_and_reject(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    batch_ok = tmp_path / "batch_ok"
    batch_bad = tmp_path / "batch_bad"
    accepted_dir = tmp_path / "accepted"
    rejected_dir = tmp_path / "rejected"
    for d in (real_dir, batch_ok, batch_bad):
        d.mkdir(parents=True)

    for i in range(2):
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(batch_ok / f"ok_{i}.png")
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(batch_bad / f"bad_{i}.png")

    accept_pipeline = SyntheticBatchPipeline(
        generator=_StubGenerator(), fid_evaluator=_StubFID(31.4), fid_threshold=50.0
    )
    rec_ok = accept_pipeline.evaluate_and_filter_batch(
        str(real_dir), str(batch_ok), str(accepted_dir), str(rejected_dir)
    )
    assert isinstance(rec_ok, BatchFIDRecord)
    assert rec_ok.accepted is True
    assert rec_ok.fid_score == 31.4
    assert len(rec_ok.moved_files) == 2
    assert len(list(accepted_dir.glob("*.png"))) == 2

    reject_pipeline = SyntheticBatchPipeline(
        generator=_StubGenerator(), fid_evaluator=_StubFID(72.8), fid_threshold=50.0
    )
    rec_bad = reject_pipeline.evaluate_and_filter_batch(
        str(real_dir), str(batch_bad), str(accepted_dir), str(rejected_dir)
    )
    assert rec_bad.accepted is False
    assert rec_bad.fid_score == 72.8
    assert len(list(rejected_dir.glob("*.png"))) == 2


def test_pytorch_fid_adapter_handles_missing_or_tiny_dirs(tmp_path: Path, monkeypatch: object) -> None:
    adapter = PyTorchFIDAdapter()
    # Missing directories must return +inf (never falsely accept a batch)
    assert adapter.compute_fid(str(tmp_path / "missing_a"), str(tmp_path / "missing_b")) == float("inf")


class _SequenceFID:
    """Fake FID adapter returning a predefined score sequence and recording real/synth counts."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = list(scores)
        self.calls: list[dict[str, object]] = []

    def compute_fid(self, real_dir: str, synthetic_dir: str, batch_size: int = 32) -> float:
        real_files = sorted(f for f in Path(real_dir).iterdir() if f.is_file() or f.is_symlink())
        synth_files = sorted(f for f in Path(synthetic_dir).iterdir() if f.is_file())
        idx = len(self.calls)
        score = self._scores[idx] if idx < len(self._scores) else self._scores[-1]
        self.calls.append(
            {
                "real_files": [f.name for f in real_files],
                "synth_files": [f.name for f in synth_files],
                "score": score,
            }
        )
        return score


def _build_training_split_fixture(base_dir: Path) -> tuple[Path, Path]:
    """Create a Patient-Disjoint training split with isolated, comorbid, and normal images."""
    img_dir = base_dir / "real_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"Image Index": "00000001_000.png", "Finding Labels": "Pneumothorax", "Label": 1, "Patient ID": "00000001"},
        {"Image Index": "00000002_000.png", "Finding Labels": "Pneumothorax|Effusion", "Label": 1, "Patient ID": "00000002"},
        {"Image Index": "00000003_000.png", "Finding Labels": "Pneumothorax", "Label": 1, "Patient ID": "00000003"},
        {"Image Index": "00000004_000.png", "Finding Labels": "Infiltration|Pneumothorax", "Label": 1, "Patient ID": "00000004"},
        {"Image Index": "00000005_000.png", "Finding Labels": "Pneumothorax", "Label": 1, "Patient ID": "00000005"},
        {"Image Index": "00000006_000.png", "Finding Labels": "No Finding", "Label": 0, "Patient ID": "00000006"},
        {"Image Index": "00000007_000.png", "Finding Labels": "No Finding", "Label": 0, "Patient ID": "00000007"},
    ]
    for idx, row in enumerate(rows):
        arr = np.full((64, 64, 3), fill_value=30 + idx * 20, dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / row["Image Index"])

    train_csv = base_dir / "train.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    return train_csv, img_dir


def test_multi_batch_processes_all_isolated_and_comorbid_seeds_with_fid_gating(
    tmp_path: Path,
) -> None:
    train_csv, img_dir = _build_training_split_fixture(tmp_path)
    out_dir = tmp_path / "synthetic_all"
    grid_path = tmp_path / "figures" / "synthetic_quality_grid.png"

    gen = _StubGenerator()
    # 5 pneumothorax seeds with batch_size=2 -> 3 batches (2, 2, 1 seeds)
    # Batch 1 accepted (28.5 <= 50), Batch 2 rejected (64.2 > 50), Batch 3 accepted (49.9 <= 50)
    fid = _SequenceFID([28.5, 64.2, 49.9])
    pipeline = SyntheticBatchPipeline(generator=gen, fid_evaluator=fid, fid_threshold=50.0)

    summary = pipeline.run_multi_batch(
        SyntheticGenerationConfig(
            train_csv=str(train_csv),
            real_images_dir=str(img_dir),
            output_dir=str(out_dir),
            seed_scope="all",
            batch_size=2,
            num_variations=3,
            seed=42,
            fid_threshold=50.0,
            qualitative_grid_path=str(grid_path),
        )
    )

    assert isinstance(summary, MultiBatchGenerationSummary)
    assert summary.total_seeds == 5
    assert summary.processed_seeds == 5
    assert summary.skipped_seeds == 0
    assert summary.total_batches == 3
    assert summary.batches_executed == 3
    assert summary.batches_accepted == 2
    assert summary.batches_rejected == 1
    assert summary.accepted_images == 9  # (2 + 1) seeds * 3 variations
    assert summary.rejected_images == 6  # 2 seeds * 3 variations

    # Only pneumothorax images (never No Finding) are used as seeds or FID real references
    generated_seed_names = [Path(str(c["init_image_path"])).name for c in gen.calls]
    assert generated_seed_names == [
        "00000001_000.png",
        "00000002_000.png",
        "00000003_000.png",
        "00000004_000.png",
        "00000005_000.png",
    ]
    assert len(fid.calls) == 3
    for call in fid.calls:
        assert call["real_files"] == [
            "00000001_000.png",
            "00000002_000.png",
            "00000003_000.png",
            "00000004_000.png",
            "00000005_000.png",
        ]

    # Routed image files match accept/reject decisions
    assert len(list((out_dir / "accepted").glob("*.png"))) == 9
    assert len(list((out_dir / "rejected").glob("*.png"))) == 6

    # FID log records batch index, seed count, generated count, FID score, and accept/reject decision
    fid_df = pd.read_csv(out_dir / "fid_log.csv")
    assert list(fid_df["batch"]) == [1, 2, 3]
    assert list(fid_df["seed_count"]) == [2, 2, 1]
    assert list(fid_df["generated_count"]) == [6, 6, 3]
    assert list(fid_df["fid"]) == pytest.approx([28.5, 64.2, 49.9])
    assert list(fid_df["accepted"]) == [True, False, True]
    assert list(fid_df["decision"]) == ["accepted", "rejected", "accepted"]

    # Accepted metadata tracks only accepted images and preserves seed provenance
    meta_df = pd.read_csv(out_dir / "synthetic_metadata.csv")
    assert len(meta_df) == 9
    assert set(meta_df["Finding Labels"]) == {"Pneumothorax"}
    assert set(meta_df["Seed Image"]) == {
        "00000001_000.png",
        "00000002_000.png",
        "00000005_000.png",
    }

    # Qualitative comparison grid was persisted
    assert grid_path.is_file()
    assert summary.qualitative_grid_path == str(grid_path)


def test_multi_batch_isolated_seed_scope_filters_out_comorbid_cases(tmp_path: Path) -> None:
    train_csv, img_dir = _build_training_split_fixture(tmp_path)
    out_dir = tmp_path / "synthetic_isolated"

    gen = _StubGenerator()
    fid = _SequenceFID([25.0])
    pipeline = SyntheticBatchPipeline(generator=gen, fid_evaluator=fid, fid_threshold=50.0)

    summary = pipeline.run_multi_batch(
        SyntheticGenerationConfig(
            train_csv=str(train_csv),
            real_images_dir=str(img_dir),
            output_dir=str(out_dir),
            seed_scope="isolated",
            batch_size=10,
            num_variations=2,
            fid_threshold=30.0,
        )
    )

    assert summary.total_seeds == 3
    assert summary.accepted_images == 6
    assert [Path(str(c["init_image_path"])).name for c in gen.calls] == [
        "00000001_000.png",
        "00000003_000.png",
        "00000005_000.png",
    ]


class _SeedRecordingInterruptibleGenerator:
    """Fake generator that records per-seed RNG seeds and optionally crashes on a target seed."""

    def __init__(self, fail_on_seed_name: str | None = None) -> None:
        self.fail_on_seed_name = fail_on_seed_name
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        init_image_path: str,
        num_variations: int = 3,
        strength: float = 0.3,
        severity: str | None = None,
        location: str | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        name = Path(init_image_path).name
        if self.fail_on_seed_name is not None and name == self.fail_on_seed_name:
            raise RuntimeError(f"Simulated interruption on {name}")
        self.calls.append({"seed_name": name, "num_variations": num_variations, "seed": seed})
        base_val = 20 if seed is None else (seed % 200)
        return [
            Image.fromarray(np.full((32, 32, 3), fill_value=(base_val + i) % 255, dtype=np.uint8))
            for i in range(num_variations)
        ]


def test_multi_batch_crash_resume_and_idempotency(tmp_path: Path) -> None:
    train_csv, img_dir = _build_training_split_fixture(tmp_path)
    out_dir = tmp_path / "synthetic_resume"

    cfg = SyntheticGenerationConfig(
        train_csv=str(train_csv),
        real_images_dir=str(img_dir),
        output_dir=str(out_dir),
        seed_scope="all",
        batch_size=2,
        num_variations=3,
        seed=100,
        fid_threshold=50.0,
        resume=True,
    )

    # Run 1: Batch 1 completes (seeds 1 & 2, accepted); crashes mid-batch 2 on seed 4
    gen_crash = _SeedRecordingInterruptibleGenerator(fail_on_seed_name="00000004_000.png")
    fid_run1 = _SequenceFID([22.0])
    pipe_run1 = SyntheticBatchPipeline(generator=gen_crash, fid_evaluator=fid_run1, fid_threshold=50.0)

    with pytest.raises(RuntimeError, match="Simulated interruption"):
        pipe_run1.run_multi_batch(cfg)

    # Batch 1 persisted cleanly; batch 2 was NOT logged as completed, and completion marker absent
    assert list(pd.read_csv(out_dir / "fid_log.csv")["batch"]) == [1]
    assert len(pd.read_csv(out_dir / "synthetic_metadata.csv")) == 6
    assert len(list((out_dir / "accepted").glob("*.png"))) == 6
    assert not (out_dir / "generation_complete.json").exists()

    # Run 2: Resume completes batches 2 (rejected) and 3 (accepted) without re-running batch 1
    gen_resume = _SeedRecordingInterruptibleGenerator()
    fid_run2 = _SequenceFID([61.5, 34.0])
    pipe_run2 = SyntheticBatchPipeline(generator=gen_resume, fid_evaluator=fid_run2, fid_threshold=50.0)
    summary_run2 = pipe_run2.run_multi_batch(cfg)
    assert (out_dir / "generation_complete.json").is_file()

    assert [c["seed_name"] for c in gen_resume.calls] == [
        "00000003_000.png",
        "00000004_000.png",
        "00000005_000.png",
    ]
    # Deterministic per-seed RNG seeds continue cleanly across resume boundary
    assert [c["seed"] for c in gen_resume.calls] == [106, 109, 112]
    assert summary_run2.total_seeds == 5
    assert summary_run2.processed_seeds == 5
    assert summary_run2.skipped_seeds == 2
    assert summary_run2.batches_executed == 2
    assert summary_run2.batches_accepted == 2
    assert summary_run2.batches_rejected == 1
    assert summary_run2.accepted_images == 9
    assert summary_run2.rejected_images == 6

    fid_df = pd.read_csv(out_dir / "fid_log.csv")
    assert list(fid_df["batch"]) == [1, 2, 3]
    assert list(fid_df["fid"]) == pytest.approx([22.0, 61.5, 34.0])
    assert list(fid_df["accepted"]) == [True, False, True]

    meta_df = pd.read_csv(out_dir / "synthetic_metadata.csv")
    assert len(meta_df) == 9
    assert meta_df["Image Index"].nunique() == 9

    # Run 3: Re-invoking after completion with resume=True is a strict zero-work no-op
    gen_noop = _SeedRecordingInterruptibleGenerator()
    fid_noop = _SequenceFID([99.9])
    pipe_noop = SyntheticBatchPipeline(generator=gen_noop, fid_evaluator=fid_noop, fid_threshold=50.0)
    summary_noop = pipe_noop.run_multi_batch(cfg)

    assert gen_noop.calls == []
    assert fid_noop.calls == []
    assert summary_noop.batches_executed == 0
    assert summary_noop.skipped_seeds == 5
    assert summary_noop.accepted_images == 9
    assert len(pd.read_csv(out_dir / "fid_log.csv")) == 3
    assert len(pd.read_csv(out_dir / "synthetic_metadata.csv")) == 9


def test_multi_batch_resume_discards_uncommitted_metadata_and_guards_split_change(
    tmp_path: Path,
) -> None:
    train_csv, img_dir = _build_training_split_fixture(tmp_path)
    out_dir = tmp_path / "synthetic_edge"

    cfg = SyntheticGenerationConfig(
        train_csv=str(train_csv),
        real_images_dir=str(img_dir),
        output_dir=str(out_dir),
        seed_scope="all",
        batch_size=2,
        num_variations=2,
        seed=42,
        fid_threshold=50.0,
        resume=True,
    )

    # Complete batch 1 then crash on batch 2
    gen_crash = _SeedRecordingInterruptibleGenerator(fail_on_seed_name="00000003_000.png")
    pipe1 = SyntheticBatchPipeline(generator=gen_crash, fid_evaluator=_SequenceFID([20.0]))
    with pytest.raises(RuntimeError):
        pipe1.run_multi_batch(cfg)

    # Simulate an uncommitted phantom row in synthetic_metadata.csv from batch 2
    meta_path = out_dir / "synthetic_metadata.csv"
    meta_df = pd.read_csv(meta_path)
    phantom = pd.DataFrame(
        [
            {
                "Image Index": "synth_phantom_uncommitted.png",
                "Finding Labels": "Pneumothorax",
                "Seed Image": "00000003_000.png",
                "Patient ID": "00000003",
                "Variation": 0,
                "Batch": 2,
            }
        ]
    )
    pd.concat([meta_df, phantom], ignore_index=True).to_csv(meta_path, index=False)

    # Resume with batch 2 rejected -> phantom uncommitted metadata must NOT remain
    pipe2 = SyntheticBatchPipeline(
        generator=_SeedRecordingInterruptibleGenerator(),
        fid_evaluator=_SequenceFID([80.0, 25.0]),
    )
    summary2 = pipe2.run_multi_batch(cfg)
    final_meta = pd.read_csv(meta_path)
    assert "synth_phantom_uncommitted.png" not in set(final_meta["Image Index"])
    assert summary2.accepted_images == 6  # batch 1 (4) + batch 3 (2)

    # Changing training split with resume=True must fail loudly to prevent split mixing
    alt_csv = tmp_path / "train_changed.csv"
    pd.read_csv(train_csv).iloc[:3].to_csv(alt_csv, index=False)
    changed_cfg = SyntheticGenerationConfig(
        train_csv=str(alt_csv),
        real_images_dir=str(img_dir),
        output_dir=str(out_dir),
        batch_size=2,
        num_variations=2,
        resume=True,
    )
    with pytest.raises(ValueError, match="resume_state.json"):
        pipe2.run_multi_batch(changed_cfg)

    # Passing resume=False cleanly resets and regenerates for the new split
    fresh_cfg = SyntheticGenerationConfig(
        train_csv=str(alt_csv),
        real_images_dir=str(img_dir),
        output_dir=str(out_dir),
        batch_size=2,
        num_variations=2,
        resume=False,
    )
    summary_fresh = SyntheticBatchPipeline(
        generator=_SeedRecordingInterruptibleGenerator(),
        fid_evaluator=_SequenceFID([18.0, 19.0]),
    ).run_multi_batch(fresh_cfg)
    assert summary_fresh.total_seeds == 3
    assert summary_fresh.accepted_images == 6
    assert len(pd.read_csv(out_dir / "fid_log.csv")) == 2


def test_generate_synthetics_cli_supports_configurable_multibatch_flags(tmp_path: Path) -> None:
    from scripts import generate_synthetics

    train_csv, img_dir = _build_training_split_fixture(tmp_path)
    out_dir = tmp_path / "cli_synth"
    grid_path = tmp_path / "cli_grid.png"

    fake_pipeline = SyntheticBatchPipeline(
        generator=_SeedRecordingInterruptibleGenerator(),
        fid_evaluator=_SequenceFID([24.0, 58.0, 31.0]),
        fid_threshold=50.0,
    )

    summary = generate_synthetics.main(
        argv=[
            "--train-csv",
            str(train_csv),
            "--image-dir",
            str(img_dir),
            "--output-dir",
            str(out_dir),
            "--seed-scope",
            "all",
            "--batch-size",
            "2",
            "--num-variations",
            "3",
            "--seed",
            "77",
            "--fid-threshold",
            "45.0",
            "--qualitative-grid-path",
            str(grid_path),
            "--resume",
        ],
        pipeline=fake_pipeline,
    )

    assert summary is not None
    assert summary.total_seeds == 5
    assert summary.batches_accepted == 2
    assert summary.batches_rejected == 1
    assert summary.accepted_images == 9
    assert (out_dir / "fid_log.csv").is_file()
    assert (out_dir / "synthetic_metadata.csv").is_file()
    assert grid_path.is_file()

    # Verify default CLI flags process all seeds (not a hardcoded 5-seed sample)
    default_args = generate_synthetics.parse_args([])
    assert default_args.seed_scope == "all"
    assert default_args.max_seeds is None
    assert default_args.num_variations == 3
    assert default_args.resume is True


def test_production_notebook_runs_synthetics_before_tier2_and_keeps_a8_real_only() -> None:
    import json
    import re

    from tcrp.training import AblationRunner

    repo_root = Path(__file__).resolve().parents[2]
    nb_path = repo_root / "notebooks" / "xray_colab_produce_all.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell_sources = [
        "".join(c["source"]) if isinstance(c["source"], list) else str(c["source"])
        for c in nb["cells"]
    ]

    # Locate config, synthetic generation, and model training code cells
    config_src = next(src for src in cell_sources if "TRAIN_HP =" in src and "STEPS =" in src)
    synth_idx = next(
        i for i, src in enumerate(cell_sources) if re.search(r"produce\(\s*'synthetic_generation'", src)
    )
    train_idx = next(
        i
        for i, src in enumerate(cell_sources)
        if re.search(r"produce\(\s*'train_tier2_efficientnet'", src)
    )

    assert synth_idx < train_idx, (
        "Synthetic generation cell must execute BEFORE Tier 2 model training in the production notebook."
    )

    # Synthetic generation enabled in STEPS and listed as an NIH consumer
    assert "'synthetic_generation':     {'run': True" in config_src
    assert "'synthetic_generation'" in config_src.split("NIH_CONSUMERS")[1].split("CHEXPERT_CONSUMERS")[0]

    # Synthetic generation cell passes configurable multi-batch CLI arguments
    synth_src = cell_sources[synth_idx]
    for flag in (
        "scripts/generate_synthetics.py",
        "--seed-scope",
        "--batch-size",
        "--num-variations",
        "--fid-threshold",
        "--seed",
    ):
        assert flag in synth_src, f"Synthetic cell missing configurable CLI flag {flag}"
    assert "data/synthetic/fid_log.csv" in synth_src

    # Tier 2 base training uses accepted synthetics; Tier 1 stays real-only
    train_src = cell_sources[train_idx]
    assert "use_synthetic=False, num_workers=" not in train_src, (
        "Training helper must not hardcode use_synthetic=False for all base models."
    )
    assert "use_synthetic={hp['use_synthetic']" in train_src or "use_synthetic={hp.get('use_synthetic'" in train_src
    assert "'train_tier1'" in config_src and "use_synthetic=False" in config_src
    assert "use_synthetic=True" in config_src
    assert "data/synthetic/synthetic_metadata.csv" in train_src

    # No-synthetic ablation (A8) remains strictly real-only in AblationRunner and notebook
    runner = AblationRunner()
    assert "--no-synthetic" in runner.ablation_configs["A8"]["args"]
    assert runner.ablation_configs["A8"]["overrides"].get("data.synthetic_augmentation") is False
    assert "--no-synthetic" in runner.ablation_configs["A9"]["args"]
    for syn_ab in ("A11", "A12", "A15"):
        assert "--no-synthetic" not in runner.ablation_configs[syn_ab]["args"]
    assert "'A8'" in train_src and "'A9'" in train_src


def test_run_multi_batch_invokes_on_batch_complete_callback(tmp_path: Path) -> None:
    img_dir = tmp_path / "real"
    out_dir = tmp_path / "synth"
    img_dir.mkdir(parents=True)

    rows = []
    for i in range(4):
        name = f"0000010{i}_000.png"
        Image.fromarray(np.full((32, 32, 3), 50 + i * 20, dtype=np.uint8)).save(img_dir / name)
        rows.append(
            {
                "Image Index": name,
                "Finding Labels": "Pneumothorax",
                "Label": 1,
                "Patient ID": f"0000010{i}",
            }
        )
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)

    events: list[tuple[int, int, bool]] = []
    pipeline = SyntheticBatchPipeline(
        generator=_StubGenerator(),
        fid_evaluator=_StubFID(25.0),
        fid_threshold=50.0,
    )
    cfg = SyntheticGenerationConfig(
        train_csv=str(train_csv),
        real_images_dir=str(img_dir),
        output_dir=str(out_dir),
        batch_size=2,
        num_variations=2,
        resume=False,
    )
    summary = pipeline.run_multi_batch(
        cfg,
        on_batch_complete=lambda batch_idx, total, rec: events.append((batch_idx, total, rec.accepted)),
    )
    assert summary.total_batches == 2
    assert events == [(1, 2, True), (2, 2, True)]


def test_pytorch_fid_adapter_caches_real_directory_statistics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    real_dir = tmp_path / "real"
    synth_a = tmp_path / "synth_a"
    synth_b = tmp_path / "synth_b"
    for d in (real_dir, synth_a, synth_b):
        d.mkdir(parents=True)
        for i in range(2):
            Image.fromarray(np.full((32, 32, 3), 40 * (i + 1), dtype=np.uint8)).save(d / f"img_{i}.png")

    compute_calls: list[str] = []

    def fake_compute_stats(path: str, model: object, batch_size: int, dims: int, device: object, num_workers: int = 1) -> tuple[np.ndarray, np.ndarray]:
        compute_calls.append(path)
        return np.zeros(4, dtype=np.float64), np.eye(4, dtype=np.float64)

    def fake_frechet(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
        return 12.5

    fake_inception_mod = types.ModuleType("pytorch_fid.inception")
    fake_inception_mod.InceptionV3 = lambda blocks: types.SimpleNamespace(to=lambda dev: types.SimpleNamespace())  # type: ignore[attr-defined]
    fake_inception_mod.InceptionV3.BLOCK_INDEX_BY_DIM = {2048: 3}  # type: ignore[attr-defined]

    fake_fid_score_mod = types.ModuleType("pytorch_fid.fid_score")
    fake_fid_score_mod.compute_statistics_of_path = fake_compute_stats  # type: ignore[attr-defined]
    fake_fid_score_mod.calculate_frechet_distance = fake_frechet  # type: ignore[attr-defined]

    fake_pkg = types.ModuleType("pytorch_fid")
    fake_pkg.fid_score = fake_fid_score_mod  # type: ignore[attr-defined]
    fake_pkg.inception = fake_inception_mod  # type: ignore[attr-defined]

    import sys

    monkeypatch.setitem(sys.modules, "pytorch_fid", fake_pkg)
    monkeypatch.setitem(sys.modules, "pytorch_fid.fid_score", fake_fid_score_mod)
    monkeypatch.setitem(sys.modules, "pytorch_fid.inception", fake_inception_mod)

    adapter = PyTorchFIDAdapter(device=torch.device("cpu"))
    score_1 = adapter.compute_fid(str(real_dir), str(synth_a))
    score_2 = adapter.compute_fid(str(real_dir), str(synth_b))

    assert score_1 == 12.5
    assert score_2 == 12.5
    # Real directory computed ONCE across both batches; synth_a and synth_b computed once each
    assert compute_calls == [str(real_dir), str(synth_a), str(synth_b)]


def test_makefile_and_notebook_match_synthetic_defaults() -> None:
    import json

    repo_root = Path(__file__).resolve().parents[2]
    makefile_text = (repo_root / "Makefile").read_text(encoding="utf-8")
    assert "--batch-size 200" in makefile_text
    assert "--strength 0.15" in makefile_text

    nb = json.loads((repo_root / "notebooks" / "xray_colab_produce_all.ipynb").read_text(encoding="utf-8"))
    cell_sources = [
        "".join(c["source"]) if isinstance(c["source"], list) else str(c["source"])
        for c in nb["cells"]
    ]
    config_src = next(src for src in cell_sources if "SYNTH_HP =" in src)
    synth_src = next(src for src in cell_sources if "scripts/generate_synthetics.py" in src)
    assert "batch_size=200" in config_src
    assert "strength=0.15" in config_src
    assert "'--strength'" in synth_src


def test_synthetic_audit_reports_recorded_generation_params_and_keeps_stage1_grid(tmp_path):
    import json

    import pandas as pd

    from scripts.revision_experiments import run_synthetic_audit

    syn = tmp_path / "synthetic"
    (syn / "accepted").mkdir(parents=True)
    pd.DataFrame({"batch": [1, 2, 3], "fid": [40.0, 48.0, 60.0], "accepted": [True, True, False]}).to_csv(
        syn / "fid_log.csv", index=False
    )
    gen_summary = tmp_path / "synthetic_generation_summary.json"
    gen_summary.write_text(json.dumps({"strength": 0.15, "batch_size": 200, "num_variations": 3,
                                       "fid_threshold": 50.0, "seed_scope": "all"}))
    stage1_grid = tmp_path / "synthetic_quality_grid.png"
    stage1_grid.write_bytes(b"stage1-grid")

    out = run_synthetic_audit(
        synthetic_dir=str(syn),
        output_json=str(tmp_path / "audit.json"),
        output_fig=str(tmp_path / "synthetic_audit_grid.png"),
        generation_summary=str(gen_summary),
    )
    assert out["strength"] == 0.15
    assert out["batch_size"] == 200
    assert out["fid_accepted_mean"] == 44.0
    assert out["fid_accepted_max"] == 48.0
    assert stage1_grid.read_bytes() == b"stage1-grid"


def test_synthetic_audit_never_invents_generation_params(tmp_path):
    from scripts.revision_experiments import run_synthetic_audit

    out = run_synthetic_audit(
        synthetic_dir=str(tmp_path / "missing"),
        output_json=str(tmp_path / "audit.json"),
        output_fig=str(tmp_path / "grid.png"),
        generation_summary=str(tmp_path / "absent.json"),
    )
    assert "strength" not in out
