"""Tests for the Track B Stage 2 runner: step plan, resume, and Stage 1 input verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_track_b_stage2 import (
    STAGE1_INPUTS,
    build_stage2_plan,
    pending_steps,
    verify_stage1_inputs,
)


def _names(plan):
    return [s.name for s in plan]


def test_plan_calibrates_temperature_before_conformal_before_any_evaluation():
    names = _names(build_stage2_plan(python="python"))
    t, c = names.index("calibrate_temperature"), names.index("calibrate_conformal")
    evals = [i for i, n in enumerate(names) if n.startswith("eval_")]
    assert t < c < min(evals)


def test_plan_covers_every_ticket_07_deliverable():
    names = set(_names(build_stage2_plan(python="python")))
    expected_evals = {
        f"eval_{a}"
        for a in ("A1", "A2", "A3", "A4", "A5", "A5b", "A6", "A7", "A8", "A9", "A10",
                  "A11", "A12", "A13", "A14", "A14b", "A15")
    }
    assert expected_evals <= names
    assert {
        "prepare_chexpert",
        "predictions_arkplus",
        "predictions_efficientnet",
        "ablation_json",
        "statistical_tests",
        "compare_models",
        "revision_experiments",
        "thesis_figures",
        "benchmark_latency",
        "provenance_gate",
    } <= names
    assert _names(build_stage2_plan(python="python"))[-1] == "provenance_gate"


def test_every_step_input_is_a_stage1_input_or_an_earlier_step_output():
    produced: set[str] = set(STAGE1_INPUTS)
    for step in build_stage2_plan(python="python"):
        missing = [p for p in step.inputs if p not in produced]
        assert not missing, f"{step.name} consumes {missing} before any step produces them"
        produced.update(step.outputs)


def test_bootstrap_and_mc_tta_knobs_reach_the_commands():
    plan = {s.name: s for s in build_stage2_plan(python="py", n_boot=2000, mc_passes=20, tta_passes=10)}
    stats = plan["statistical_tests"].argv
    assert stats[stats.index("--n-iterations") + 1] == "2000"
    a6 = plan["eval_A6"].argv
    assert a6[a6.index("--mc-passes") + 1] == "20"
    assert a6[a6.index("--tta-passes") + 1] == "10"
    assert plan["revision_experiments"].argv[-2:] == ("--n-permutations", "10")


def test_pending_steps_resumes_after_completed_steps(tmp_path: Path):
    plan = build_stage2_plan(python="python")
    first = plan[0]
    for out in first.outputs:
        (tmp_path / out).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / out).write_text("x")
    state = {"completed": {first.name: {"seconds": 1.0}}}
    remaining = pending_steps(plan, root=tmp_path, state=state)
    assert remaining[0].name == plan[1].name
    assert first.name not in _names(remaining)


def test_pending_steps_reruns_a_completed_step_whose_outputs_vanished(tmp_path: Path):
    plan = build_stage2_plan(python="python")
    state = {"completed": {plan[0].name: {"seconds": 1.0}}}
    assert pending_steps(plan, root=tmp_path, state=state)[0].name == plan[0].name


def test_pending_steps_never_trusts_outputs_without_a_completion_record(tmp_path: Path):
    plan = build_stage2_plan(python="python")
    for out in plan[0].outputs:
        (tmp_path / out).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / out).write_text("stale")
    assert pending_steps(plan, root=tmp_path, state={})[0].name == plan[0].name


def _write(root: Path, rel: str, data: bytes) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_verify_stage1_inputs_accepts_matching_hashes(tmp_path: Path):
    hashes = {"outputs/models/best_tier1_mobilenet.pth": _write(tmp_path, "outputs/models/best_tier1_mobilenet.pth", b"w")}
    summary = tmp_path / "stage1.json"
    summary.write_text(json.dumps({"protocol": "patient-disjoint-v1", "checkpoint_sha256": hashes}))
    assert verify_stage1_inputs(tmp_path, summary) == []


def test_verify_stage1_inputs_reports_tampered_and_missing_checkpoints(tmp_path: Path):
    good = _write(tmp_path, "outputs/models/a.pth", b"original")
    (tmp_path / "outputs/models/a.pth").write_bytes(b"retrained")
    summary = tmp_path / "stage1.json"
    summary.write_text(
        json.dumps(
            {
                "protocol": "patient-disjoint-v1",
                "checkpoint_sha256": {"outputs/models/a.pth": good, "outputs/models/b.pth": "0" * 64},
            }
        )
    )
    errors = verify_stage1_inputs(tmp_path, summary)
    assert any("a.pth" in e and "mismatch" in e.lower() for e in errors)
    assert any("b.pth" in e and "missing" in e.lower() for e in errors)


def test_verify_stage1_inputs_rejects_wrong_protocol(tmp_path: Path):
    summary = tmp_path / "stage1.json"
    summary.write_text(json.dumps({"protocol": "image-level-v0", "checkpoint_sha256": {}}))
    errors = verify_stage1_inputs(tmp_path, summary)
    assert errors and "protocol" in errors[0].lower()


@pytest.mark.parametrize("missing", ["data/processed/calibration.csv", "outputs/provenance/split_manifest.json"])
def test_stage1_inputs_include_splits_and_manifest(missing: str):
    assert missing in STAGE1_INPUTS
