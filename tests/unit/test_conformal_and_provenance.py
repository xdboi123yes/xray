"""Unit tests for temperature-before-conformal calibration, conformal diagnostics, and provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import structlog
import torch
import torch.nn as nn
from tcrp.uncertainty.calibration import ModelWithTemperature
from tcrp.uncertainty.conformal import ConformalEvaluationReport, ConformalPredictor
from torch.utils.data import DataLoader, TensorDataset


class _FixedLogitModel(nn.Module):
    """Deterministic model returning pre-configured logits per sample index."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = x.long().view(-1)
        return self._logits[idx].to(x.device)


def test_conformal_evaluate_predictions_computes_exact_diagnostics_and_logs() -> None:
    """Verify marginal coverage, class-conditional coverage, set-size rates, and empty fallback."""
    cp = ConformalPredictor(alpha=0.10)
    # threshold = 1 - q_hat = 0.60
    cp.q_hat = 0.40

    # Four worked examples against threshold = 0.60:
    # 0: [0.80, 0.20], label=0 -> raw={0} (singleton), hit=True
    # 1: [0.55, 0.45], label=0 -> raw={} -> fallback={0} (empty fallback + singleton), hit=True
    # 2: [0.25, 0.75], label=1 -> raw={1} (singleton), hit=True
    # 3: [0.30, 0.70], label=0 -> raw={1} (singleton), hit=False
    probs = np.array(
        [
            [0.80, 0.20],
            [0.55, 0.45],
            [0.25, 0.75],
            [0.30, 0.70],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 0], dtype=int)

    with structlog.testing.capture_logs() as cap_logs:
        report = cp.evaluate_predictions(probs, labels, split_name="test")

    assert isinstance(report, ConformalEvaluationReport)
    assert report.n_samples == 4
    assert report.q_hat == pytest.approx(0.40)
    assert report.alpha == pytest.approx(0.10)
    assert report.marginal_coverage == pytest.approx(3 / 4)
    assert report.coverage_class0 == pytest.approx(2 / 3)
    assert report.coverage_class1 == pytest.approx(1.0)
    assert report.singleton_rate == pytest.approx(1.0)
    assert report.two_class_abstention_rate == pytest.approx(0.0)
    assert report.empty_fallback_rate == pytest.approx(1 / 4)
    assert report.avg_set_size == pytest.approx(1.0)

    assert any(
        entry.get("event") == "Conformal evaluation metrics"
        and entry.get("split") == "test"
        and "marginal_coverage" in entry
        and "coverage_class0" in entry
        and "coverage_class1" in entry
        and "singleton_rate" in entry
        and "two_class_abstention_rate" in entry
        and "empty_fallback_rate" in entry
        for entry in cap_logs
    )


def test_conformal_evaluate_predictions_tracks_two_class_abstention_rate() -> None:
    """When q_hat is large (threshold < 0.5), uncertain predictions produce 2-class abstention sets."""
    cp = ConformalPredictor(alpha=0.05)
    # threshold = 1 - 0.70 = 0.30
    cp.q_hat = 0.70

    # Sample 0: [0.50, 0.50] -> both >= 0.30 -> 2-class abstention, hit=True
    # Sample 1: [0.90, 0.10] -> {0} singleton, hit=True
    # Sample 2: [0.40, 0.60] -> both >= 0.30 -> 2-class abstention, hit=True
    probs = np.array([[0.50, 0.50], [0.90, 0.10], [0.40, 0.60]], dtype=float)
    labels = np.array([1, 0, 0], dtype=int)

    report = cp.evaluate_predictions(probs, labels, split_name="test")
    assert report.marginal_coverage == pytest.approx(1.0)
    assert report.singleton_rate == pytest.approx(1 / 3)
    assert report.two_class_abstention_rate == pytest.approx(2 / 3)
    assert report.empty_fallback_rate == pytest.approx(0.0)
    assert report.avg_set_size == pytest.approx(5 / 3)


def test_conformal_calibrate_consumes_temperature_guards_double_scaling_and_logs(
    tmp_path: Path,
) -> None:
    """Calibration applies Validation-Split temperature, logs diagnostics, and persists T_cal."""
    logits = torch.tensor(
        [
            [2.0, -2.0],
            [-2.0, 2.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
    indices = torch.arange(len(labels), dtype=torch.float32)
    cal_loader = DataLoader(TensorDataset(indices, labels), batch_size=5, shuffle=False)
    model = _FixedLogitModel(logits)

    cp_unscaled = ConformalPredictor(alpha=0.2)
    q_unscaled = cp_unscaled.calibrate(model, cal_loader, torch.device("cpu"), temperature=1.0)

    cp_scaled = ConformalPredictor(alpha=0.2)
    with structlog.testing.capture_logs() as cap_logs:
        q_scaled = cp_scaled.calibrate(model, cal_loader, torch.device("cpu"), temperature=2.5)

    # Higher temperature softens softmax probabilities -> larger nonconformity scores -> larger q_hat
    assert q_scaled > q_unscaled
    assert cp_scaled.temperature == pytest.approx(2.5)
    assert cp_scaled.last_calibration_report is not None
    assert cp_scaled.last_calibration_report.n_samples == 5
    assert any(
        entry.get("event") == "Conformal evaluation metrics"
        and entry.get("split") == "calibration"
        and "empty_fallback_rate" in entry
        for entry in cap_logs
    )

    # Save / load preserves q_hat, alpha, and temperature
    ckpt = str(tmp_path / "q_hat.pt")
    cp_scaled.save(ckpt)
    reloaded = ConformalPredictor(alpha=0.05)
    reloaded.load(ckpt)
    assert reloaded.q_hat == pytest.approx(q_scaled)
    assert reloaded.alpha == pytest.approx(0.2)
    assert reloaded.temperature == pytest.approx(2.5)

    # Reject non-positive temperature and accidental double scaling with ModelWithTemperature
    with pytest.raises(ValueError, match="positive"):
        ConformalPredictor(alpha=0.1).calibrate(
            model, cal_loader, torch.device("cpu"), temperature=0.0
        )

    wrapped = ModelWithTemperature(model)
    with torch.no_grad():
        wrapped.temperature.fill_(2.5)
    with pytest.raises(ValueError, match="already temperature-scaled"):
        ConformalPredictor(alpha=0.1).calibrate(
            wrapped, cal_loader, torch.device("cpu"), temperature=2.5
        )

    # Passing wrapped ModelWithTemperature with default temperature=1.0 succeeds and matches manual T=2.5
    cp_wrapped = ConformalPredictor(alpha=0.2)
    q_wrapped = cp_wrapped.calibrate(wrapped, cal_loader, torch.device("cpu"))
    assert q_wrapped == pytest.approx(q_scaled)
    assert cp_wrapped.temperature == pytest.approx(2.5)


def test_conformal_evaluate_uses_calibrated_temperature_on_loader() -> None:
    """ConformalPredictor.evaluate() applies stored temperature scalar and reports override temperature."""
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.arange(4, dtype=torch.float32), labels), batch_size=4)
    model = _FixedLogitModel(logits)

    cp = ConformalPredictor(alpha=0.1)
    cp.q_hat = 0.70  # threshold = 0.30
    cp.temperature = 3.0

    # At T=1.0: softmax([2, 0]) = [0.88, 0.12] (singleton), softmax([1, 0]) = [0.73, 0.27] (singleton)
    #   -> singleton_rate = 1.0, two_class_abstention_rate = 0.0
    # At T=3.0: softmax([2/3, 0]) = [0.66, 0.34] (two-class >= 0.30), softmax([1/3, 0]) = [0.58, 0.42] (two-class)
    #   -> singleton_rate = 0.0, two_class_abstention_rate = 1.0
    expected_scaled_probs = torch.softmax(logits / 3.0, dim=1)
    expected_report = cp.evaluate_predictions(
        expected_scaled_probs, labels, split_name="test", temperature=3.0
    )

    rep_stored = cp.evaluate(model, loader, torch.device("cpu"), split_name="test")
    assert rep_stored == expected_report
    assert rep_stored.temperature == pytest.approx(3.0)
    assert rep_stored.two_class_abstention_rate == pytest.approx(1.0)
    assert rep_stored.singleton_rate == pytest.approx(0.0)

    rep_unscaled = cp.evaluate(
        model, loader, torch.device("cpu"), temperature=1.0, split_name="test"
    )
    assert rep_unscaled.temperature == pytest.approx(1.0)
    assert rep_unscaled.singleton_rate == pytest.approx(1.0)
    assert rep_unscaled.two_class_abstention_rate == pytest.approx(0.0)


def test_conformal_and_provenance_edge_case_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject non-finite/bool temperatures, invalid labels, missing overlaps, corrupt JSON, and CWD drift."""
    import json

    from tcrp.evaluation.provenance import build_provenance_manifest, verify_provenance_manifest

    cp = ConformalPredictor(alpha=0.1)
    cp.q_hat = 0.4
    probs = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    with pytest.raises(ValueError, match="binary"):
        cp.evaluate_predictions(probs, np.array([0, 2]))
    with pytest.raises(ValueError, match="1D"):
        cp.evaluate_predictions(probs, np.array([[0], [1]]))

    # Reject nan/inf/bool temperatures in ConformalPredictor
    dummy_loader = DataLoader(
        TensorDataset(torch.arange(2, dtype=torch.float32), torch.tensor([0, 1])), batch_size=2
    )
    dummy_model = _FixedLogitModel(torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))
    for bad_temp in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite|positive"):
            ConformalPredictor(alpha=0.1).calibrate(
                dummy_model, dummy_loader, torch.device("cpu"), temperature=bad_temp  # type: ignore[arg-type]
            )

    # Split manifest missing pairwise_patient_overlaps must be rejected
    no_overlaps_split = tmp_path / "split_no_overlaps.json"
    no_overlaps_split.write_text(
        json.dumps({"splits": {"train": {}, "val": {}, "calibration": {}, "test": {}}}),
        encoding="utf-8",
    )
    art = tmp_path / "artifact.bin"
    art.write_bytes(b"base-dir-content")
    with pytest.raises(ValueError, match="pairwise_patient_overlaps"):
        build_provenance_manifest(
            output_path=tmp_path / "m_no_ov.json",
            artifact_paths=[art],
            split_manifest_path=no_overlaps_split,
            base_dir=tmp_path,
        )

    # Reject NaN / bool / string temperatures in calibration_metadata
    for bad_cal_temp in (float("nan"), float("inf"), True, "bad"):
        with pytest.raises(ValueError, match="finite positive temperature"):
            build_provenance_manifest(
                output_path=tmp_path / "m_bad_temp.json",
                artifact_paths=[art],
                calibration_metadata={
                    "temperature_split": "val",
                    "conformal_split": "calibration",
                    "temperature_before_conformal": True,
                    "temperature": bad_cal_temp,
                },
                base_dir=tmp_path,
            )

    # Relative path resolution always resolves against base_dir, never CWD
    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir()
    (other_cwd / "artifact.bin").write_bytes(b"cwd-decoy-content")
    monkeypatch.chdir(other_cwd)
    m_rel = tmp_path / "m_rel.json"
    build_provenance_manifest(
        output_path=m_rel,
        artifact_paths=["artifact.bin"],
        base_dir=tmp_path,
    )
    ver_rel = verify_provenance_manifest(m_rel, base_dir=tmp_path, strict=True)
    assert ver_rel.valid is True

    # Corrupt JSON returns valid=False when strict=False instead of raising JSONDecodeError
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    ver_corrupt = verify_provenance_manifest(corrupt, base_dir=tmp_path, strict=False)
    assert ver_corrupt.valid is False
    assert any("parse" in err for err in ver_corrupt.errors)


def test_provenance_manifest_build_verify_and_tamper_detection(tmp_path: Path) -> None:
    """Provenance manifest hashes artifacts, enforces split/calibration contracts, catches tampering."""
    import json

    from tcrp.evaluation.provenance import (
        ProvenanceVerificationResult,
        build_provenance_manifest,
        compute_file_sha256,
        verify_provenance_manifest,
    )

    # Create valid split manifest + critical artifacts under tmp_path
    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text(
        json.dumps(
            {
                "protocol": "patient-disjoint-v1",
                "splits": {
                    "train": {"samples": 20, "patients": 10},
                    "val": {"samples": 6, "patients": 3},
                    "calibration": {"samples": 6, "patients": 3},
                    "test": {"samples": 8, "patients": 4},
                },
                "pairwise_patient_overlaps": {
                    "calibration__test": 0,
                    "calibration__train": 0,
                    "calibration__val": 0,
                    "test__train": 0,
                    "test__val": 0,
                    "train__val": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    temp_art = tmp_path / "temperature.pt"
    qhat_art = tmp_path / "q_hat.pt"
    ckpt_art = tmp_path / "tier2.pth"
    temp_art.write_bytes(b"temp-bytes")
    qhat_art.write_bytes(b"qhat-bytes")
    ckpt_art.write_bytes(b"ckpt-bytes")

    manifest_path = tmp_path / "rerun_manifest.json"
    manifest = build_provenance_manifest(
        output_path=manifest_path,
        artifact_paths=[temp_art, qhat_art, ckpt_art],
        protocol="patient-disjoint-v1",
        split_manifest_path=split_manifest,
        calibration_metadata={
            "temperature_split": "val",
            "conformal_split": "calibration",
            "temperature_before_conformal": True,
            "temperature": 1.35,
            "q_hat": 0.42,
        },
        base_dir=tmp_path,
        git_commit="abc1234",
    )

    assert manifest_path.is_file()
    assert manifest["protocol"] == "patient-disjoint-v1"
    assert manifest["git_commit"] == "abc1234"
    assert manifest["artifacts_sha256"]["temperature.pt"] == compute_file_sha256(temp_art)
    assert manifest["artifacts_sha256"]["q_hat.pt"] == compute_file_sha256(qhat_art)
    assert manifest["artifacts_sha256"]["split_manifest.json"] == compute_file_sha256(split_manifest)

    result = verify_provenance_manifest(
        manifest_path=manifest_path,
        base_dir=tmp_path,
        expected_protocol="patient-disjoint-v1",
        required_artifacts=["temperature.pt", "q_hat.pt", "tier2.pth"],
        strict=True,
    )
    assert isinstance(result, ProvenanceVerificationResult)
    assert result.valid is True
    assert result.verified_artifacts == 4
    assert result.errors == ()

    # Tampering with an artifact must fail verification loudly
    qhat_art.write_bytes(b"tampered-bytes")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_provenance_manifest(
            manifest_path=manifest_path,
            base_dir=tmp_path,
            expected_protocol="patient-disjoint-v1",
            strict=True,
        )

    non_strict = verify_provenance_manifest(
        manifest_path=manifest_path,
        base_dir=tmp_path,
        expected_protocol="patient-disjoint-v1",
        strict=False,
    )
    assert non_strict.valid is False
    assert any("SHA-256 mismatch" in err for err in non_strict.errors)


def test_provenance_manifest_rejects_overlap_or_same_split_calibration(tmp_path: Path) -> None:
    """Builder/verifier rejects leaky splits or invalid calibration split ordering."""
    import json

    from tcrp.evaluation.provenance import build_provenance_manifest

    leaky_split = tmp_path / "split_manifest.json"
    leaky_split.write_text(
        json.dumps(
            {
                "protocol": "patient-disjoint-v1",
                "splits": {"train": {}, "val": {}, "calibration": {}, "test": {}},
                "pairwise_patient_overlaps": {"calibration__val": 2},
            }
        ),
        encoding="utf-8",
    )
    art = tmp_path / "a.bin"
    art.write_bytes(b"data")

    with pytest.raises(ValueError, match="overlap"):
        build_provenance_manifest(
            output_path=tmp_path / "m1.json",
            artifact_paths=[art],
            split_manifest_path=leaky_split,
            base_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="disjoint"):
        build_provenance_manifest(
            output_path=tmp_path / "m2.json",
            artifact_paths=[art],
            calibration_metadata={
                "temperature_split": "calibration",
                "conformal_split": "calibration",
                "temperature_before_conformal": True,
                "temperature": 1.2,
            },
            base_dir=tmp_path,
        )


def test_tiered_system_applies_validation_temperature_and_defers_two_class_conformal() -> None:
    """TieredSystem scales Tier-2 logits by T_cal and flags two-class conformal sets for Human Deferral."""
    from tcrp.models.base import BaseClassifier
    from tcrp.models.cascade import TieredSystem
    from tcrp.routing import StaticThresholdRouter

    class _ConstClassifier(BaseClassifier):
        def __init__(self, logits: list[float]) -> None:
            super().__init__()
            self._logits = torch.tensor([logits], dtype=torch.float32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._logits.expand(x.size(0), -1).to(x.device)

    tier1 = _ConstClassifier([0.1, -0.1])  # low confidence -> escalates
    tier2 = _ConstClassifier([0.0, 2.0])  # raw softmax ~ [0.12, 0.88]; at T=4.0 -> [0.38, 0.62]

    # At threshold = 1 - q_hat = 0.30:
    # - unscaled (T=1.0): [0.12, 0.88] -> singleton {'Pneumothorax'} -> not deferred
    # - scaled (T=4.0):   [0.38, 0.62] -> two-class {'No Finding', 'Pneumothorax'} -> Human Deferral
    cp = ConformalPredictor(alpha=0.05)
    cp.q_hat = 0.70

    x = torch.zeros(1, 3, 16, 16)
    sys_unscaled = TieredSystem(
        tier1_model=tier1,
        tier2_model=tier2,
        config={"model": {"mc_dropout_passes": 2, "tta_passes": 2}},
        conformal_predictor=cp,
        router=StaticThresholdRouter(threshold=0.95),
        temperature=1.0,
    )
    res_unscaled = sys_unscaled.route(x)
    assert res_unscaled.tier_used == 2
    assert res_unscaled.conformal_set == ["Pneumothorax"]
    assert res_unscaled.flagged_for_review is False

    sys_scaled = TieredSystem(
        tier1_model=tier1,
        tier2_model=tier2,
        config={"model": {"mc_dropout_passes": 2, "tta_passes": 2}},
        conformal_predictor=cp,
        router=StaticThresholdRouter(threshold=0.95),
        temperature=4.0,
    )
    res_scaled = sys_scaled.route(x)
    assert res_scaled.tier_used == 2
    assert res_scaled.conformal_set == ["No Finding", "Pneumothorax"]
    assert res_scaled.flagged_for_review is True
    assert res_scaled.confidence < res_unscaled.confidence


def test_generate_report_figures_accepts_validation_split_prediction_columns(tmp_path: Path) -> None:
    """plot_roc_curve works directly on export_predictions.py val_predictions.csv schema."""
    import pandas as pd

    from scripts.generate_report_figures import plot_roc_curve

    df = pd.DataFrame(
        {
            "image_id": [f"0000000{i}_000.png" for i in range(6)],
            "y_true": [0, 0, 0, 1, 1, 1],
            "tiered_prob": [0.1, 0.2, 0.3, 0.7, 0.85, 0.9],
        }
    )
    out_png = tmp_path / "roc_curve.png"
    plot_roc_curve(df, str(out_png))
    assert out_png.is_file() and out_png.stat().st_size > 0


def test_production_notebook_enforces_temperature_before_conformal_and_provenance_gate() -> None:
    """Notebook runs Val temperature scaling BEFORE disjoint Cal conformal q_hat and verifies provenance."""
    import json
    import re

    repo_root = Path(__file__).resolve().parents[2]
    nb_path = repo_root / "notebooks" / "xray_colab_produce_all.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    sources = [
        "".join(c["source"]) if isinstance(c["source"], list) else str(c["source"])
        for c in cells
    ]

    config_src = next(src for src in sources if "STEPS =" in src and "MARKERS =" in src)
    temp_idx = next(
        i for i, src in enumerate(sources) if re.search(r"produce\(\s*'calibration_temperature'", src)
    )
    conf_idx = next(
        i for i, src in enumerate(sources) if re.search(r"produce\(\s*'calibration_conformal'", src)
    )
    fig_idx = next(
        i for i, src in enumerate(sources) if re.search(r"produce\(\s*'figure_input_csvs'", src)
    )
    gate_idx = next(
        i for i, src in enumerate(sources) if re.search(r"produce\(\s*'integrity_gate'", src)
    )

    # 1. Temperature scaling strictly precedes split-conformal calibration
    assert temp_idx < conf_idx, (
        "Temperature scaling must execute BEFORE split-conformal calibration in the notebook."
    )

    temp_src = sources[temp_idx]
    conf_src = sources[conf_idx]

    # 2. Temperature scaling fits exclusively on Validation Split (val.csv)
    assert "data/processed/val.csv" in temp_src
    assert "data/processed/calibration.csv" not in temp_src
    assert "outputs/results/temperature.pt" in temp_src

    # 3. Split-conformal consumes temperature.pt, verifies val vs calibration disjointness,
    #    calibrates strictly on calibration.csv, and logs/persists conformal diagnostics
    assert "outputs/results/temperature.pt" in conf_src
    assert "data/processed/calibration.csv" in conf_src
    assert "verify_patient_disjointness" in conf_src
    assert "temperature=" in conf_src
    assert "conformal_calibration_summary.json" in conf_src
    assert "last_calibration_report" in conf_src
    assert "'val_auc': 0.82" not in conf_src, "Notebook must not persist fabricated fallback val_auc"

    # 4. Config registers calibration_conformal and eval_A5b in STEPS, MARKERS, and NIH_CONSUMERS
    assert "'calibration_conformal'" in config_src
    assert "'eval_A5b'" in config_src
    assert "'outputs/results/q_hat.pt'" in config_src

    # 5. Figure-input cell must NOT overwrite Validation-Split val_predictions.csv with test predictions
    fig_src = sources[fig_idx]
    assert "'outputs/results/val_predictions.csv'" not in fig_src
    assert '"outputs/results/val_predictions.csv"' not in fig_src

    # 6. Provenance gate uses build_provenance_manifest + verify_provenance_manifest and hashes calibration artifacts
    gate_src = sources[gate_idx]
    assert "build_provenance_manifest" in gate_src
    assert "verify_provenance_manifest" in gate_src
    assert "calibration_temperature" in gate_src
    assert "calibration_conformal" in gate_src
    assert "synthetic_generation" in gate_src


def _split_fixture(tmp_path: Path) -> Path:
    import hashlib
    import json

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    splits = {}
    for name in ("train", "val", "calibration", "test"):
        raw = f"Image Index,Label\n{name}_001.png,0\n".encode()
        (processed / f"{name}.csv").write_bytes(raw)
        splits[name] = {"rows": 1, "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(json.dumps({"splits": splits}))
    return manifest


def test_verify_split_csv_hashes_accepts_the_recorded_splits(tmp_path: Path) -> None:
    from tcrp.evaluation.provenance import verify_split_csv_hashes

    manifest = _split_fixture(tmp_path)
    assert verify_split_csv_hashes(manifest, tmp_path / "data" / "processed") == []


def test_verify_split_csv_hashes_flags_regenerated_or_missing_splits(tmp_path: Path) -> None:
    from tcrp.evaluation.provenance import verify_split_csv_hashes

    manifest = _split_fixture(tmp_path)
    processed = tmp_path / "data" / "processed"
    (processed / "test.csv").write_text("Image Index,Label\nother.png,1\n")
    (processed / "calibration.csv").unlink()
    errors = verify_split_csv_hashes(manifest, processed)
    assert any("test" in e and "mismatch" in e.lower() for e in errors)
    assert any("calibration" in e and "missing" in e.lower() for e in errors)
    assert len(errors) == 2
