"""Resumable Track B Stage 2 runner (calibration, evaluations, analyses, benchmarks, provenance).

Runs on the Colab L4 after the Stage 1 artifacts (splits, synthetics, checkpoints) are restored
from the Track B Drive tree. Each step is a subprocess with its own log; completed steps are
recorded in a state file so a detached run can be relaunched and resume where it stopped.
Stage 1 checkpoints are verified against ``stage1_training_summary.json`` SHA-256 digests first,
so Stage 2 can never silently evaluate a retrained or corrupted model.

Usage (from the repo root)::

    python -m scripts.run_track_b_stage2 [--drive-sync-dir DIR] [--only STEP ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "patient-disjoint-v1"
STATE_PATH = "outputs/provenance/stage2_state.json"
STAGE1_SUMMARY_PATH = "outputs/provenance/stage1_training_summary.json"
LOG_DIR = "outputs/logs/stage2"

T1 = "outputs/models/best_tier1_mobilenet.pth"
EFF = "outputs/models/best_tier2_efficientnet.pth"
ARK = "outputs/models/best_tier2_ark_plus.pth"
THRESHOLD = "outputs/models/tier1_mobilenet_threshold.json"
TEMPERATURE = "outputs/results/temperature.pt"
Q_HAT = "outputs/results/q_hat.pt"
Q_HAT_ARK = "outputs/results/q_hat_ark_plus.pt"
TEST_CSV = "data/processed/test.csv"
CHEXPERT_CSV = "data/processed/chexpert_test.csv"
CHEXPERT_LARGE_CSV = "data/processed/chexpert_large_test.csv"
ARK_PRED = "outputs/results/tiered_predictions.csv"
ARK_VAL_PRED = "outputs/results/val_predictions.csv"
EFF_PRED = "outputs/results/tiered_predictions_efficientnet.csv"

# Artifacts that Stage 1 (ticket 06) must have restored before Stage 2 starts.
STAGE1_INPUTS: tuple[str, ...] = (
    "data/raw/Data_Entry_2017.csv",
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/calibration.csv",
    TEST_CSV,
    "outputs/provenance/split_manifest.json",
    STAGE1_SUMMARY_PATH,
    "data/synthetic/fid_log.csv",
    "data/synthetic/synthetic_metadata.csv",
    "data/synthetic/generation_complete.json",
    T1,
    EFF,
    ARK,
    THRESHOLD,
    "outputs/models/A8_NoSynthetic/best_model.pth",
    "outputs/models/A9_NoAugmentation/best_model.pth",
    "outputs/models/A11_ArkPlus_Only_NoMCTTA/best_model.pth",
    "outputs/models/A12_ArkPlus_Only_MC_TTA/best_model.pth",
    "outputs/models/A15_Mixup_Cutmix/best_model.pth",
)

# (ablation id, run name, extra weight inputs) evaluated by scripts/evaluate_ablation.py.
_ABLATION_EVALS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("A1", "A1_Tier1_Only", (T1,)),
    ("A2", "A2_Tier2_EfficientNet_Only", (EFF,)),
    ("A3", "A3_Tiered_Static", (T1, EFF)),
    ("A4", "A4_Tiered_Dynamic", (T1, EFF)),
    ("A5", "A5_Tiered_MC_Only", (T1, EFF)),
    ("A5b", "A5b_Tiered_TTA_Only", (T1, EFF)),
    ("A6", "A6_Tiered_MC_TTA", (T1, EFF)),
    ("A7", "A7_Tiered_Conformal", (T1, EFF)),
    ("A8", "A8_NoSynthetic", (T1, "outputs/models/A8_NoSynthetic/best_model.pth")),
    ("A9", "A9_NoAugmentation", (T1, "outputs/models/A9_NoAugmentation/best_model.pth")),
    ("A10", "A10_Always_Tier2", (EFF,)),
    ("A11", "A11_ArkPlus_Only_NoMCTTA", ("outputs/models/A11_ArkPlus_Only_NoMCTTA/best_model.pth",)),
    ("A12", "A12_ArkPlus_Only_MC_TTA", ("outputs/models/A12_ArkPlus_Only_MC_TTA/best_model.pth",)),
    ("A15", "A15_Mixup_Cutmix", (T1, "outputs/models/A15_Mixup_Cutmix/best_model.pth")),
)


@dataclass(frozen=True)
class Step:
    """One Stage 2 subprocess with the files it consumes and must produce."""

    name: str
    argv: tuple[str, ...]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = field(default_factory=tuple)


def build_stage2_plan(
    python: str = sys.executable,
    n_boot: int = 2000,
    mc_passes: int = 20,
    tta_passes: int = 10,
    micro_batch: int = 512,
    batch_size: int = 16,
    num_workers: int = 8,
    n_permutations: int = 10,
    chexpert_cache_dir: str | None = None,
) -> list[Step]:
    """Return the ordered Stage 2 plan: calibration first, provenance gate last."""
    py = (python, "-u")
    calibrated = (TEMPERATURE, Q_HAT)
    steps: list[Step] = [
        Step(
            "calibrate_temperature",
            (*py, "scripts/calibrate_track_b.py", "temperature"),
            inputs=(EFF, "data/processed/val.csv"),
            outputs=(TEMPERATURE, "outputs/figures/reliability_tier2_eff.png"),
        ),
        Step(
            "calibrate_conformal",
            (*py, "scripts/calibrate_track_b.py", "conformal"),
            inputs=(TEMPERATURE, "data/processed/calibration.csv"),
            outputs=(Q_HAT, "outputs/results/conformal_calibration_summary.json"),
        ),
    ]
    for ablation_id, run_name, weights in _ABLATION_EVALS:
        steps.append(
            Step(
                f"eval_{ablation_id}",
                (
                    *py, "scripts/evaluate_ablation.py", "--ablation", ablation_id,
                    "--batch-size", str(batch_size), "--micro-batch", str(micro_batch),
                    "--num-workers", str(num_workers), "--amp",
                    "--mc-passes", str(mc_passes), "--tta-passes", str(tta_passes),
                ),
                inputs=(TEST_CSV, *calibrated, *weights),
                outputs=(f"outputs/results/{run_name}.json",),
            )
        )
    steps += [
        Step(
            "eval_A13",
            (*py, "scripts/evaluate_tiered.py", "--run-name", "A13_Tiered_ArkPlus",
             "--tier2-backbone", "ark_plus", "--dynamic-threshold", "true"),
            inputs=(TEST_CSV, T1, ARK, THRESHOLD, *calibrated),
            outputs=("outputs/results/A13_Tiered_ArkPlus.json", Q_HAT_ARK),
        ),
        Step(
            "prepare_chexpert",
            (*py, "scripts/prepare_chexpert.py",
             *(("--drive-cache-dir", chexpert_cache_dir) if chexpert_cache_dir else ())),
            outputs=(CHEXPERT_CSV, CHEXPERT_LARGE_CSV),
        ),
        Step(
            "eval_A14",
            (*py, "scripts/evaluate_chexpert.py", "--run-name", "A14_CheXpert_ZeroShot",
             "--tier2-backbone", "ark_plus", "--csv", CHEXPERT_CSV, "--n-bootstrap", str(n_boot)),
            inputs=(CHEXPERT_CSV, T1, ARK, Q_HAT_ARK),
            outputs=("outputs/results/A14_CheXpert_ZeroShot.json",),
        ),
        Step(
            "eval_A14b",
            (*py, "scripts/evaluate_chexpert.py", "--run-name", "A14b_CheXpert_Large_ZeroShot",
             "--tier2-backbone", "ark_plus", "--csv", CHEXPERT_LARGE_CSV, "--n-bootstrap", str(n_boot),
             "--recalibrate-from", "outputs/results/A14_CheXpert_ZeroShot.json"),
            inputs=(CHEXPERT_LARGE_CSV, T1, ARK, Q_HAT_ARK, "outputs/results/A14_CheXpert_ZeroShot.json"),
            outputs=("outputs/results/A14b_CheXpert_Large_ZeroShot.json",),
        ),
        Step(
            "predictions_arkplus",
            (*py, "scripts/export_predictions.py", "--tier2-backbone", "ark_plus",
             "--mc-passes", str(mc_passes), "--output", ARK_PRED, "--val-output", ARK_VAL_PRED),
            inputs=(TEST_CSV, T1, ARK, THRESHOLD),
            outputs=(ARK_PRED, ARK_VAL_PRED),
        ),
        Step(
            "predictions_efficientnet",
            (*py, "scripts/export_predictions.py", "--tier2-backbone", "efficientnet_b4",
             "--mc-passes", str(mc_passes), "--output", EFF_PRED,
             "--val-output", "outputs/results/val_predictions_efficientnet.csv"),
            inputs=(TEST_CSV, T1, EFF, THRESHOLD),
            outputs=(EFF_PRED, "outputs/results/val_predictions_efficientnet.csv"),
        ),
    ]
    eval_jsons = tuple(o for s in steps if s.name.startswith("eval_") for o in s.outputs if o.endswith(".json"))
    steps += [
        Step(
            "ablation_json",
            (*py, "scripts/build_ablation_json.py"),
            inputs=eval_jsons,
            outputs=("outputs/results/ablation.json",),
        ),
        Step(
            "statistical_tests",
            (*py, "scripts/statistical_tests.py", "--n-iterations", str(n_boot)),
            inputs=(ARK_PRED, EFF_PRED),
            outputs=("outputs/results/statistical_comparison.csv",),
        ),
        Step(
            "compare_models",
            (*py, "scripts/compare_models.py", "--n-iterations", str(n_boot)),
            inputs=(ARK_PRED,),
            outputs=("outputs/results/model_comparison_bootstrap.csv",),
        ),
        Step(
            "revision_experiments",
            (*py, "scripts/revision_experiments.py", "--all", "--require-inputs",
             "--test-predictions", ARK_PRED, "--val-predictions", ARK_VAL_PRED,
             "--n-permutations", str(n_permutations)),
            inputs=(ARK_PRED, ARK_VAL_PRED, "data/synthetic/fid_log.csv"),
            outputs=(
                "outputs/results/dataset_audit.json",
                "outputs/results/routing_grid_val.csv",
                "outputs/results/order_sensitivity.json",
                "outputs/results/clinical_threshold_sweep.csv",
                "outputs/results/calibration_summary.json",
                "outputs/figures/reliability_diagram_15bin.png",
                "outputs/results/synthetic_audit.json",
            ),
        ),
        Step(
            "thesis_figures",
            (*py, "scripts/build_thesis_figures.py"),
            inputs=("outputs/results/ablation.json", "outputs/results/A13_Tiered_ArkPlus.json",
                    "outputs/results/A14_CheXpert_ZeroShot.json"),
            outputs=(),
        ),
        Step(
            "benchmark_latency",
            (*py, "scripts/benchmark_latency.py", "--mc-passes", str(mc_passes),
             "--tta-passes", str(tta_passes), "--output", "outputs/results/benchmark_latency.json"),
            inputs=(T1, EFF, ARK),
            outputs=("outputs/results/benchmark_latency.json",),
        ),
    ]
    produced = tuple(dict.fromkeys(o for s in steps for o in s.outputs))
    steps.append(
        Step(
            "provenance_gate",
            (*py, "scripts/verify_track_b_provenance.py"),
            inputs=produced,
            outputs=("outputs/provenance/requirements-lock.txt", "outputs/provenance/rerun_manifest.json"),
        )
    )
    return steps


def _outputs_present(step: Step, root: Path) -> bool:
    return all((root / p).is_file() for p in step.outputs)


def pending_steps(plan: list[Step], root: Path, state: dict[str, Any]) -> list[Step]:
    """Return the steps still to run, from the first incomplete one to the end.

    A step counts as complete only with a completion record AND all outputs on disk;
    once one step must rerun, everything downstream reruns too.
    """
    completed = state.get("completed", {})
    for i, step in enumerate(plan):
        if step.name not in completed or not _outputs_present(step, root):
            return plan[i:]
    return []


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_stage1_inputs(root: Path, summary_path: Path) -> list[str]:
    """Check Stage 1 checkpoints against the SHA-256 digests recorded at the end of ticket 06."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if summary.get("protocol") != PROTOCOL_VERSION:
        return [f"Stage 1 protocol mismatch: expected {PROTOCOL_VERSION!r}, got {summary.get('protocol')!r}"]
    errors: list[str] = []
    for rel, expected in summary.get("checkpoint_sha256", {}).items():
        path = root / rel
        if not path.is_file():
            errors.append(f"Stage 1 artifact missing: {rel}")
        elif _sha256(path) != expected:
            errors.append(f"Stage 1 SHA-256 mismatch: {rel}")
    return errors


def _load_state(root: Path) -> dict[str, Any]:
    path = root / STATE_PATH
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"protocol": PROTOCOL_VERSION, "completed": {}}


def _save_state(root: Path, state: dict[str, Any]) -> None:
    path = root / STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sync_to_drive(root: Path, drive_dir: Path, rel_paths: list[str]) -> None:
    for rel in rel_paths:
        src = root / rel
        if src.is_file():
            dst = drive_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _sync_tree(root: Path, drive_dir: Path, rel_dir: str) -> None:
    src_dir = root / rel_dir
    if src_dir.is_dir():
        files = [str(p.relative_to(root)) for p in src_dir.rglob("*") if p.is_file()]
        _sync_to_drive(root, drive_dir, files)


def run(plan: list[Step], root: Path, drive_dir: Path | None) -> int:
    """Execute pending steps in order, stopping at the first failure."""
    state = _load_state(root)
    (root / LOG_DIR).mkdir(parents=True, exist_ok=True)
    for step in pending_steps(plan, root, state):
        missing = [p for p in step.inputs if not (root / p).is_file()]
        if missing:
            print(f"[stage2] BLOCKED {step.name}: missing inputs {missing}", flush=True)
            return 2
        state["completed"].pop(step.name, None)
        for out in step.outputs:  # never let a stale artifact pass for a fresh one
            (root / out).unlink(missing_ok=True)
        log_path = root / LOG_DIR / f"{step.name}.log"
        print(f"[stage2] START {step.name}: {' '.join(step.argv)}", flush=True)
        t0 = time.time()
        with log_path.open("w") as log:
            rc = subprocess.call(list(step.argv), cwd=root, stdout=log, stderr=subprocess.STDOUT)
        seconds = round(time.time() - t0, 2)
        missing_out = [p for p in step.outputs if not (root / p).is_file()]
        if rc != 0 or missing_out:
            print(f"[stage2] FAILED {step.name} rc={rc} missing_outputs={missing_out} ({seconds}s); "
                  f"see {log_path}", flush=True)
            return 1
        state["completed"][step.name] = {"seconds": seconds, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _save_state(root, state)
        print(f"[stage2] DONE {step.name} ({seconds}s)", flush=True)
        if drive_dir is not None:
            _sync_to_drive(root, drive_dir, [*step.outputs, STATE_PATH, str(log_path.relative_to(root))])
    if drive_dir is not None:
        for rel_dir in ("outputs/results", "outputs/figures", "outputs/provenance", "paper", "thesis/figures"):
            _sync_tree(root, drive_dir, rel_dir)
    print("[stage2] ALL STEPS COMPLETE", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drive-sync-dir", type=Path, default=None,
                        help="Copy each step's outputs here when it finishes (Track B Drive tree).")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--tta-passes", type=int, default=10)
    parser.add_argument("--micro-batch", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chexpert-cache-dir", default=None,
                        help="Drive folder caching the CheXpert valid/ and large_sample/ images.")
    parser.add_argument("--skip-stage1-check", action="store_true",
                        help="Do not verify Stage 1 checkpoint hashes (never for paper runs).")
    args = parser.parse_args()

    root = Path.cwd()
    missing = [p for p in STAGE1_INPUTS if not (root / p).is_file()]
    if missing:
        sys.exit(f"[stage2] Stage 1 artifacts missing; restore them from Drive first: {missing}")
    if not args.skip_stage1_check:
        errors = verify_stage1_inputs(root, root / STAGE1_SUMMARY_PATH)
        if errors:
            sys.exit("[stage2] Stage 1 verification failed: " + "; ".join(errors))
        print("[stage2] Stage 1 checkpoints match stage1_training_summary.json SHA-256 digests.", flush=True)

    plan = build_stage2_plan(
        n_boot=args.n_boot, mc_passes=args.mc_passes, tta_passes=args.tta_passes,
        micro_batch=args.micro_batch, num_workers=args.num_workers,
        chexpert_cache_dir=args.chexpert_cache_dir,
    )
    sys.exit(run(plan, root, args.drive_sync_dir))


if __name__ == "__main__":
    main()
