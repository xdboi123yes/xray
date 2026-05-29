"""Unit tests for the Ablation Study orchestration runner.

Verifies configuration mapping, dynamic config overrides, and mock dry-run capabilities.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

from application.orchestration.ablation_runner import AblationRunner


def test_ablation_runner_configs() -> None:
    """Test that AblationRunner defines correct Week 4 experiments."""
    runner = AblationRunner()
    configs = runner.ablation_configs

    assert "A11" in configs
    assert "A12" in configs
    assert "A13" in configs

    # Verify A11 details
    assert configs["A11"]["script"] == "scripts/train_tier2.py"
    assert "--backbone" in configs["A11"]["args"]
    assert "ark_plus" in configs["A11"]["args"]
    assert "--no-mc-tta" in configs["A11"]["args"]

    # Verify A12 details
    assert configs["A12"]["script"] == "scripts/train_tier2.py"
    assert "ark_plus" in configs["A12"]["args"]
    assert "--no-mc-tta" not in configs["A12"]["args"]

    # Verify A13 details
    assert configs["A13"]["script"] == "scripts/evaluate_tiered.py"
    assert "--tier2-backbone" in configs["A13"]["args"]
    assert "ark_plus" in configs["A13"]["args"]


@patch("subprocess.Popen")
def test_ablation_runner_dry_run(mock_popen: MagicMock) -> None:
    """Test that AblationRunner applies dry-run override and executes successfully."""
    # Setup mock subprocess return
    mock_proc = MagicMock()
    mock_proc.wait.return_value = 0
    mock_proc.stdout = None
    mock_popen.return_value = mock_proc

    # Dummy config path that exists for override testing
    dummy_config_path = "tests/mock_config.yaml"
    with open(dummy_config_path, "w") as f:
        f.write(
            """model:
  mc_dropout_passes: 20
  tta_passes: 10
training:
  epochs: 50
"""
        )

    try:
        runner = AblationRunner(config_path=dummy_config_path)
        result = runner.run_experiment("A11", dry_run=True)

        assert result["status"] == "SUCCESS"
        assert result["returncode"] == 0
        assert mock_popen.called

        # Confirm command executed contained targeted parameters
        args, _ = mock_popen.call_args
        cmd = args[0]
        assert "scripts/train_tier2.py" in cmd[1]
        assert "ark_plus" in cmd

        # Confirm config file was correctly restored after experiment
        with open(dummy_config_path) as f:
            restored_content = f.read()
        assert "epochs: 50" in restored_content

    finally:
        # Clean up temporary test file
        if os.path.exists(dummy_config_path):
            os.remove(dummy_config_path)


def test_ablation_json_mlflow_validation() -> None:
    """Verify that every row in ablation.json carries a valid provenance and metrics."""
    ablation_json_path = "outputs/results/ablation.json"
    assert os.path.exists(ablation_json_path), f"{ablation_json_path} must exist"

    with open(ablation_json_path) as f:
        data = json.load(f)

    assert len(data) > 0, "ablation.json should not be empty"

    for row in data:
        # 1. Verify required keys exist
        assert "ablation_id" in row
        assert "provenance" in row
        assert "metrics" in row

        provenance = row["provenance"]
        assert provenance in ["mlflow_run", "preliminary_placeholder"]

        if provenance == "mlflow_run":
            run_id = row["run_id"]
            assert run_id, (
                f"ablation_id {row['ablation_id']} has empty run_id with mlflow_run provenance"
            )

            # Verify MLflow run metrics directory exists
            metrics_dir = os.path.join(
                "experiments", "mlruns", "762301816938414973", run_id, "metrics"
            )
            assert os.path.isdir(metrics_dir), (
                f"MLflow metrics directory not found for run {run_id}"
            )

            # Verify at least one version of the auc_roc metric is present in MLflow metrics
            auc_roc_found = False
            for metric_name in ["auc_roc", "val_auc", "final_auc_roc"]:
                metric_file = os.path.join(metrics_dir, metric_name)
                if os.path.exists(metric_file):
                    auc_roc_found = True
                    break
            assert auc_roc_found, f"auc_roc metric not found in MLflow metrics for run {run_id}"
