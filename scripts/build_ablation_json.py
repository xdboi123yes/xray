#!/usr/bin/env python3
"""Compile ``outputs/results/ablation.json`` from REAL evaluation markers only.

The full A1-A15 table is defined once in :mod:`scripts.ablation_spec`. For each
row this compiler reads the genuine, computed metrics that the evaluation
scripts persist to ``outputs/results/<run_name>.json``:

    * A1-A12, A15 -> written by ``evaluate_ablation.py``
    * A13         -> written by ``evaluate_tiered.py``
    * A14         -> written by ``evaluate_chexpert.py``

A row is marked ``provenance: evaluation_json`` only when both ``auc_roc`` and
``accuracy`` are present as real numbers; otherwise it stays an honest
``preliminary_placeholder`` with null metrics. No number is ever fabricated and
no metric is hardcoded here -- only the run identity and display schema.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import structlog

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Imported after the sys.path injection above so the sibling spec module resolves.
from ablation_spec import ABLATION_SPECS, AblationSpec

log = structlog.get_logger(__name__)

OUTPUT_FILE = Path("outputs/results/ablation.json")


def _as_float(value: object) -> float | None:
    """Coerce a JSON metric value to float, or None when it is not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def read_result_metrics(spec: AblationSpec) -> tuple[float | None, float | None, float | None]:
    """Return (auc_roc, accuracy, ece) from a row's results JSON, or all None."""
    path = Path(spec.result_json)
    if not path.exists():
        return None, None, None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None, None
    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
    auc = metrics.get("auc_roc", metrics.get("auc"))
    acc = metrics.get("accuracy")
    ece = metrics.get("ece")
    return _as_float(auc), _as_float(acc), _as_float(ece)


def compile_ablations() -> int:
    """Compile ablation.json from the per-row evaluation markers.

    Returns the number of rows backed by a genuine evaluation JSON.
    """
    results: list[dict[str, Any]] = []
    real_rows = 0

    log.info("ablation_compile_start", run_count=len(ABLATION_SPECS))

    for spec in ABLATION_SPECS:
        auc_roc, accuracy, ece = read_result_metrics(spec)
        if auc_roc is not None and accuracy is not None:
            provenance = "evaluation_json"
            metrics = {"auc_roc": auc_roc, "accuracy": accuracy, "ece": ece}
            real_rows += 1
            log.info("ablation_row_real", ab_id=spec.ablation_id, auc=auc_roc)
        else:
            provenance = "preliminary_placeholder"
            metrics = {"auc_roc": None, "accuracy": None, "ece": None}
            log.info("ablation_row_preliminary", ab_id=spec.ablation_id)

        results.append(
            {
                "ablation_id": spec.ablation_id,
                "name": spec.name,
                "description": spec.description,
                "tier1": spec.tier1,
                "tier2": spec.tier2,
                "routing": spec.routing,
                "uncertainty": spec.uncertainty,
                "run_id": "",
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
