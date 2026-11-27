"""Unit tests for the deepened SyntheticBatchPipeline seam with fake adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tcrp.data.synthetic import BatchFIDRecord, PyTorchFIDAdapter, SyntheticBatchPipeline


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
