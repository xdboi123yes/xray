"""Track B scientific integrity and provenance gate (final Stage 2 step).

Freezes the exact library versions (``pip freeze``) next to the git commit, re-checks that the split
CSVs and Stage 1 checkpoints are byte-identical to what ticket 06 recorded, confirms the disjoint
Validation -> Calibration calibration order and the paired prediction row counts, then writes and
strictly re-verifies ``outputs/provenance/rerun_manifest.json`` with SHA-256 digests of every
Stage 1 input and Stage 2 artifact.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tcrp.evaluation.provenance import (
    build_provenance_manifest,
    verify_provenance_manifest,
    verify_split_csv_hashes,
)

from scripts.run_track_b_stage2 import (
    PROTOCOL_VERSION,
    STAGE1_INPUTS,
    STAGE1_SUMMARY_PATH,
    build_stage2_plan,
    verify_stage1_inputs,
)

SPLIT_MANIFEST = "outputs/provenance/split_manifest.json"
LOCKFILE = "outputs/provenance/requirements-lock.txt"
MANIFEST = "outputs/provenance/rerun_manifest.json"
PREDICTIONS = ("outputs/results/tiered_predictions.csv", "outputs/results/tiered_predictions_efficientnet.csv")


def _write_lockfile(root: Path) -> None:
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    header = (
        f"# python=={platform.python_version()}\n# torch=={torch.__version__} cuda=={torch.version.cuda}\n"
        f"# gpu=={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}\n"
    )
    (root / LOCKFILE).parent.mkdir(parents=True, exist_ok=True)
    (root / LOCKFILE).write_text(header + frozen.stdout, encoding="utf-8")


def _calibration_metadata(root: Path) -> dict:
    temp = torch.load(root / "outputs/results/temperature.pt", map_location="cpu", weights_only=False)
    qhat = torch.load(root / "outputs/results/q_hat.pt", map_location="cpu", weights_only=False)
    summary = json.loads((root / "outputs/results/conformal_calibration_summary.json").read_text())
    if temp.get("backbone") != qhat.get("backbone"):
        raise ValueError(f"calibration backbone mismatch: {temp.get('backbone')!r} vs {qhat.get('backbone')!r}")
    if not math.isclose(float(temp["temperature"]), float(qhat["temperature"]), rel_tol=1e-3, abs_tol=1e-4):
        raise ValueError("temperature.pt and q_hat.pt disagree on the Validation-Split temperature")
    return {
        "backbone": str(temp["backbone"]),
        "temperature_split": str(temp.get("split")),
        "conformal_split": str(qhat.get("conformal_split")),
        "temperature_before_conformal": summary.get("temperature_before_conformal") is True,
        "temperature": float(temp["temperature"]),
        "q_hat": float(qhat["q_hat"]),
    }


def main() -> None:
    root = Path.cwd()
    errors = verify_split_csv_hashes(root / SPLIT_MANIFEST, root / "data/processed")
    errors += verify_stage1_inputs(root, root / STAGE1_SUMMARY_PATH)
    test_rows = json.loads((root / SPLIT_MANIFEST).read_text())["splits"]["test"]["rows"]
    for pred in PREDICTIONS:
        n = len(pd.read_csv(root / pred))
        if n != test_rows:
            errors.append(f"{pred} has {n} rows; patient-disjoint test split has {test_rows}")
    if errors:
        sys.exit("[provenance] INTEGRITY GATE FAILED:\n  " + "\n  ".join(errors))

    _write_lockfile(root)
    stage2_outputs = [o for s in build_stage2_plan() if s.name != "provenance_gate" for o in s.outputs]
    artifacts = sorted({*STAGE1_INPUTS, *stage2_outputs, LOCKFILE} - {SPLIT_MANIFEST})
    manifest = build_provenance_manifest(
        output_path=root / MANIFEST,
        artifact_paths=artifacts,
        protocol=PROTOCOL_VERSION,
        split_manifest_path=SPLIT_MANIFEST,
        calibration_metadata=_calibration_metadata(root),
        base_dir=root,
    )
    if manifest["git_commit"] == "unknown":
        sys.exit("[provenance] INTEGRITY GATE FAILED: run from a git checkout so the exact commit is recorded")
    result = verify_provenance_manifest(
        manifest_path=root / MANIFEST,
        base_dir=root,
        expected_protocol=PROTOCOL_VERSION,
        required_artifacts=["outputs/results/temperature.pt", "outputs/results/q_hat.pt", LOCKFILE],
        strict=True,
    )
    print(f"[provenance] SCIENTIFIC INTEGRITY GATE PASSED: {PROTOCOL_VERSION}, "
          f"{result.verified_artifacts} artifacts SHA-256 verified -> {MANIFEST}")


if __name__ == "__main__":
    main()
