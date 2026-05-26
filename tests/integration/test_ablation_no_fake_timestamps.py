"""Integration test verifying that MLflow run metrics carry genuine clock timestamps.

Checks that the unix timestamp written inside MLflow metric files is strictly
greater than or equal to the run's start_time or creation metadata.
"""

from __future__ import annotations

import json
import os
import yaml

MLRUNS_DIR = "experiments/mlruns/762301816938414973"

def test_ablation_no_fake_timestamps() -> None:
    """Verify that all mlflow_run metrics in ablation.json have valid timestamps."""
    ablation_json = "outputs/results/ablation.json"
    if not os.path.exists(ablation_json):
        return

    with open(ablation_json) as f:
        data = json.load(f)

    for row in data:
        # We only check provenance for actual mlflow_runs
        if row.get("provenance") != "mlflow_run":
            continue

        run_id = row.get("run_id")
        assert run_id, f"Ablation row {row['ablation_id']} mapped to 'mlflow_run' has empty run_id"

        run_path = os.path.join(MLRUNS_DIR, run_id)
        meta_yaml_path = os.path.join(run_path, "meta.yaml")
        assert os.path.exists(meta_yaml_path), f"meta.yaml not found for MLflow run {run_id}"

        # 1. Read start_time from meta.yaml or use directory mtime
        start_time_ms = 0
        with open(meta_yaml_path) as yf:
            meta_data = yaml.safe_load(yf)
            # MLflow meta.yaml usually has start_time or end_time (in milliseconds)
            if "start_time" in meta_data:
                start_time_ms = int(meta_data["start_time"])
            else:
                # Fallback to filesystem mtime in ms
                start_time_ms = int(os.path.getmtime(meta_yaml_path) * 1000)

        # 2. Verify metrics files have timestamps strictly >= start_time_ms
        metrics_dir = os.path.join(run_path, "metrics")
        assert os.path.exists(metrics_dir), f"metrics folder missing for run {run_id}"

        for metric_name in ["auc_roc", "val_auc", "final_auc_roc"]:
            metric_file = os.path.join(metrics_dir, metric_name)
            if not os.path.exists(metric_file):
                continue

            with open(metric_file) as mf:
                lines = mf.readlines()
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # Format is usually: timestamp value [step]
                        # MLflow writes timestamp first
                        timestamp_ms = int(parts[0])
                        # If the timestamp is fake (e.g. 1716650000000 invented date)
                        # but the run was created in 2026, then start_time_ms will be around 1779737458000
                        # which is much larger than 1716650000000, causing a failure!
                        assert timestamp_ms >= start_time_ms, (
                            f"Fake timestamp detected in run {run_id}, metric {metric_name}, line {idx + 1}: "
                            f"metric timestamp {timestamp_ms} is earlier than run start_time {start_time_ms}"
                        )
