#!/usr/bin/env python3
"""MLflow Ablation Metrics Compiler.

Reads actual training metric values directly from MLflow run outputs.
Verifies that written metrics have genuine clock timestamps (>= start_time)
so back-filled or synthetic metric files are rejected.
If a run or its metrics are missing or fail the integrity check, the
ablation row is written with `provenance: preliminary_placeholder` and
`metrics: {auc_roc: null, accuracy: null, ece: null}` — never a fabricated
number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger(__name__)


# Mapping from ablation ID to the MLflow run that produced it.
# Metrics are NEVER hardcoded here; only the run identity + schema fields.
#
# MAINTENANCE: these run_ids are PINNED to specific MLflow runs. When you RE-RUN an
# experiment (e.g. A14 on real CheXpert) MLflow creates a NEW run with a new id, so its
# fresh metrics will NOT appear here until you update the run_id below to the new run.
# Find it under experiments/mlruns/762301816938414973/<run_id>/tags/mlflow.runName.
ABLATION_RUN_MAPPING: dict[str, dict[str, Any]] = {
    "A1": {
        "run_id": "0ab2960f8632435c835dab88023a5503",
        "name": "Tier 1 Only",
        "description": "Baseline screening model using MobileNetV2 without escalating to Tier 2.",
        "tier1": "MobileNetV2",
        "tier2": "None (Bypassed)",
        "routing": "None",
        "uncertainty": "None",
    },
    "A2": {
        "run_id": "5f51c2f3b0db408cb14c723b3e41440a",
        "name": "Tier 2 Only (EfficientNet)",
        "description": "All cases run directly on the deep Tier 2 EfficientNet-B4 backbone.",
        "tier1": "None",
        "tier2": "EfficientNetB4",
        "routing": "All Escalated",
        "uncertainty": "None",
    },
    "A3": {
        "run_id": "c54f525f93744ca7b45b6dc47f41743b",
        "name": "Tier 2 Only (Ark+ Swin)",
        "description": "All cases run directly on the Tier 2 Ark+ Swin backbone.",
        "tier1": "None",
        "tier2": "Ark+ Swin",
        "routing": "All Escalated",
        "uncertainty": "None",
    },
    "A6": {
        "run_id": "1c78d7b760fc4344a0a231a88b5a4ab9",
        "name": "Proposed Tiered (Static)",
        "description": "Tiered cascading classifier with static escalation threshold (t=0.75).",
        "tier1": "MobileNetV2",
        "tier2": "EfficientNetB4",
        "routing": "Static Threshold",
        "uncertainty": "None",
    },
    "A8": {
        "run_id": "fa329c07929e42c68aa370cc9ab66287",
        "name": "Without Diffusion Augmentation",
        "description": "Tiered system trained without synthetic Stable Diffusion images.",
        "tier1": "MobileNetV2",
        "tier2": "EfficientNetB4",
        "routing": "Static Threshold",
        "uncertainty": "None",
    },
    "A9": {
        "run_id": "1ddcc9af08244228ac74cbf5b12b4138",
        "name": "Without Any Augmentations",
        "description": "Baseline tiered system trained without Mixup, CutMix, or synthetic data.",
        "tier1": "MobileNetV2",
        "tier2": "EfficientNetB4",
        "routing": "Static Threshold",
        "uncertainty": "None",
    },
    "A13": {
        "run_id": "32ab86af4eeb45e38f87f7b4ee79a88d",
        "name": "Proposed Tiered + Ark+",
        "description": "Tiered system using MobileNetV2 (T1) and Ark+ (T2) with dynamic routing.",
        "tier1": "MobileNetV2",
        "tier2": "Ark+ Swin",
        "routing": "Dynamic Routing",
        "uncertainty": "MC Dropout",
    },
    "A14": {
        "run_id": "0cc6a419c05048eebc87ee95fa5f9123",
        "name": "Zero-Shot CheXpert",
        "description": "Out-of-domain validation of A13 evaluated on the CheXpert cohort.",
        "tier1": "MobileNetV2",
        "tier2": "Ark+ Swin",
        "routing": "Dynamic Routing",
        "uncertainty": "MC Dropout",
    },
    "A15": {
        "run_id": "2900d427a6694af2b7b9365d3e24f76c",
        "name": "Mixup/CutMix Regularized",
        "description": "A13 trained with additional Mixup/CutMix regularization.",
        "tier1": "MobileNetV2",
        "tier2": "Ark+ Swin",
        "routing": "Dynamic Routing",
        "uncertainty": "MC Dropout",
    },
}

MLRUNS_DIR = Path("experiments/mlruns/762301816938414973")
OUTPUT_FILE = Path("outputs/results/ablation.json")


def get_run_start_time_ms(run_id: str) -> int | None:
    """Return the run start_time in milliseconds from meta.yaml, or None."""
    meta_path = MLRUNS_DIR / run_id / "meta.yaml"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as fh:
            meta = yaml.safe_load(fh)
            if isinstance(meta, dict) and "start_time" in meta:
                return int(meta["start_time"])
    except (OSError, yaml.YAMLError):
        return None
    return None


def read_genuine_metric(
    run_id: str, aliases: list[str], start_time_ms: int | None
) -> float | None:
    """Return the most recent valid metric value with a timestamp >= start_time.

    Returns None if the run / metric is missing, malformed, or timestamps look
    back-filled (i.e. earlier than the run's recorded start_time).
    """
    metrics_dir = MLRUNS_DIR / run_id / "metrics"
    if not metrics_dir.is_dir():
        return None

    for alias in aliases:
        metric_file = metrics_dir / alias
        if not metric_file.exists():
            continue
        try:
            lines = metric_file.read_text().strip().splitlines()
        except OSError:
            continue
        if not lines:
            continue
        parts = lines[-1].split()
        if len(parts) < 2:
            continue
        try:
            ts = int(parts[0])
            val = float(parts[1])
        except ValueError:
            continue
        if start_time_ms is not None and ts < start_time_ms:
            log.warning(
                "ablation_metric_rejected_synthetic_timestamp",
                run_id=run_id,
                metric=alias,
                value_ts=ts,
                run_start_time=start_time_ms,
            )
            continue
        return val
    return None


def compile_ablations() -> int:
    """Compile ablation.json from MLflow without ever fabricating metric values.

    Returns the number of rows whose provenance is `mlflow_run`.
    """
    results: list[dict[str, Any]] = []
    real_rows = 0

    log.info("ablation_compile_start", run_count=len(ABLATION_RUN_MAPPING))

    for ab_id, info in ABLATION_RUN_MAPPING.items():
        run_id = info["run_id"]
        start_time_ms = get_run_start_time_ms(run_id)

        auc_roc = read_genuine_metric(
            run_id, ["auc_roc", "val_auc", "final_auc_roc"], start_time_ms
        )
        accuracy = read_genuine_metric(
            run_id, ["accuracy", "val_acc", "final_accuracy"], start_time_ms
        )
        ece = read_genuine_metric(run_id, ["ece"], start_time_ms)

        if auc_roc is not None and accuracy is not None:
            provenance = "mlflow_run"
            metrics = {"auc_roc": auc_roc, "accuracy": accuracy, "ece": ece}
            real_rows += 1
            log.info("ablation_row_real", ab_id=ab_id, run_id=run_id, auc=auc_roc)
        else:
            provenance = "preliminary_placeholder"
            metrics = {"auc_roc": None, "accuracy": None, "ece": None}
            log.info(
                "ablation_row_preliminary",
                ab_id=ab_id,
                reason="missing_or_rejected_metric",
            )

        results.append(
            {
                "ablation_id": ab_id,
                "name": info["name"],
                "description": info["description"],
                "tier1": info["tier1"],
                "tier2": info["tier2"],
                "routing": info["routing"],
                "uncertainty": info["uncertainty"],
                "run_id": run_id if provenance == "mlflow_run" else "",
                "provenance": provenance,
                "metrics": metrics,
            }
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2) + "\n")

    log.info(
        "ablation_compile_complete",
        total=len(results),
        real_rows=real_rows,
        preliminary_rows=len(results) - real_rows,
        output=str(OUTPUT_FILE),
    )
    return real_rows


if __name__ == "__main__":
    compile_ablations()
