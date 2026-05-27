"""Orchestration service for Week 4 Ablation experiments.

Coordinates configurations A11, A12, and A13, dynamically overrides configurations,
executes training/evaluation scripts, compiles results, and logs to MLflow.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from typing import Any

import yaml

import structlog

log = structlog.get_logger(__name__)


class AblationRunner:
    """Orchestration runner coordinating advanced Ark+ ablation experiments.

    Supports dynamic YAML overrides, automated execution of training and
    evaluation scripts, and results compilation.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initialize the AblationRunner.

        Args:
            config_path: Path to the root project config.yaml file.
        """
        self.config_path = config_path
        self.ablation_configs: dict[str, dict[str, Any]] = {
            "A8": {
                "name": "A8_NoSynthetic",
                "description": "Full system without synthetic augmentation (classical only)",
                "script": "scripts/train_tier2.py",
                "args": [
                    "--run-name",
                    "A8_NoSynthetic",
                    "--backbone",
                    "efficientnet_b4",
                ],
                "overrides": {
                    "data.synthetic_augmentation": False,
                },
            },
            "A9": {
                "name": "A9_NoAugmentation",
                "description": "Full system without any augmentation",
                "script": "scripts/train_tier2.py",
                "args": [
                    "--run-name",
                    "A9_NoAugmentation",
                    "--backbone",
                    "efficientnet_b4",
                ],
                "overrides": {
                    "data.synthetic_augmentation": False,
                },
            },
            "A11": {
                "name": "A11_ArkPlus_Only_NoMCTTA",
                "description": "Tier 2 = Ark+ (no MC/TTA)",
                "script": "scripts/train_tier2.py",
                "args": [
                    "--run-name",
                    "A11_ArkPlus_Only_NoMCTTA",
                    "--backbone",
                    "ark_plus",
                    "--no-mc-tta",
                ],
                "overrides": {
                    "model.mc_dropout_passes": 1,
                    "model.tta_passes": 1,
                },
            },
            "A12": {
                "name": "A12_ArkPlus_Only_MC_TTA",
                "description": "Tier 2 = Ark+ + MC + TTA",
                "script": "scripts/train_tier2.py",
                "args": [
                    "--run-name",
                    "A12_ArkPlus_Only_MC_TTA",
                    "--backbone",
                    "ark_plus",
                ],
                "overrides": {},
            },
            "A13": {
                "name": "A13_Tiered_ArkPlus",
                "description": "Tiered + Ark+ Tier2 (MC + TTA + Conformal)",
                "script": "scripts/evaluate_tiered.py",
                "args": [
                    "--run-name",
                    "A13_Tiered_ArkPlus",
                    "--dynamic-threshold",
                    "true",
                    "--tier2-backbone",
                    "ark_plus",
                ],
                "overrides": {},
            },
            "A14": {
                "name": "A14_CheXpert_ZeroShot",
                "description": "A13 Tiered + Ark+ Tier2 evaluated Zero-Shot on CheXpert",
                "script": "scripts/evaluate_chexpert.py",
                "args": [
                    "--run-name",
                    "A14_CheXpert_ZeroShot",
                    "--tier2-backbone",
                    "ark_plus",
                ],
                "overrides": {},
            },
            "A15": {
                "name": "A15_Mixup_Cutmix",
                "description": "A13 Tiered + Ark+ Tier2 with Mixup/Cutmix batch regularizations",
                "script": "scripts/train_tier2.py",
                "args": [
                    "--run-name",
                    "A15_Mixup_Cutmix",
                    "--backbone",
                    "ark_plus",
                ],
                "overrides": {},
            },
        }

    def run_experiment(
        self, ablation_id: str, dry_run: bool = False
    ) -> dict[str, Any]:
        """Execute a single ablation experiment.

        Args:
            ablation_id: String ID matching A11, A12, or A13.
            dry_run: If True, restricts training epochs to 1 for quick validation.

        Returns:
            Dictionary containing execution statistics and status.

        Raises:
            ValueError: If the ablation_id is not recognized.
        """
        if ablation_id not in self.ablation_configs:
            raise ValueError(
                f"Unknown ablation ID: '{ablation_id}'. "
                f"Available configurations: {list(self.ablation_configs.keys())}"
            )

        config = self.ablation_configs[ablation_id]
        log.info(f"\n{'='*60}")
        log.info(f"STARTING ABLATION {ablation_id}: {config['description']}")
        log.info(f"{'='*60}")

        result: dict[str, Any] = {
            "ablation_id": ablation_id,
            "name": config["name"],
            "description": config["description"],
            "started_at": datetime.now().isoformat(),
            "returncode": None,
            "status": "FAILED",
        }

        # Apply config overrides
        original_yaml_content: str | None = None
        overrides = dict(config.get("overrides", {}))

        if dry_run:
            log.info("[AblationRunner] Dry-run active. Overriding epochs to 1.")
            overrides["training.epochs"] = 1

        if overrides:
            with open(self.config_path) as f:
                original_yaml_content = f.read()

            try:
                yaml_data = yaml.safe_load(original_yaml_content)
                for key_path, val in overrides.items():
                    keys = key_path.split(".")
                    d = yaml_data
                    for k in keys[:-1]:
                        d = d[k]
                    d[keys[-1]] = val

                with open(self.config_path, "w") as f:
                    yaml.dump(yaml_data, f, default_flow_style=False)
                log.info(f"[AblationRunner] Applied overrides: {overrides}")
            except Exception as ex:
                log.error(f"[AblationRunner] Error applying overrides: {ex}")
                result["error"] = f"Override Error: {ex}"
                return result

        try:
            # Execute targeted python script command
            cmd = [sys.executable, config["script"], *config.get("args", [])]
            log.info(f"[AblationRunner] Executing command: {' '.join(cmd)}")
            
            # Use Popen to stream standard output and standard error in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line-buffered
            )
            
            # Read and print output in real-time
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
            
            returncode = process.wait()
            result["returncode"] = returncode

            if returncode == 0:
                result["status"] = "SUCCESS"
                log.info(f"[AblationRunner] Ablation {ablation_id} completed successfully.")
            else:
                log.error(
                    f"[AblationRunner] Warning: Ablation {ablation_id} failed with exit code {returncode}"
                )

        except Exception as ex:
            log.error(f"[AblationRunner] Exception during execution: {ex}")
            result["error"] = str(ex)

        finally:
            # Restore original YAML config if overridden
            if original_yaml_content is not None:
                with open(self.config_path, "w") as f:
                    f.write(original_yaml_content)
                log.info("[AblationRunner] Restored original config file.")

        result["completed_at"] = datetime.now().isoformat()
        return result

    def run_all(self, dry_run: bool = False) -> list[dict[str, Any]]:
        """Run all registered ablation experiments (A11, A12, A13).

        Args:
            dry_run: If True, runs 1-epoch dry-runs for validation.

        Returns:
            List of results dictionaries for all executions.
        """
        results = []
        for ab_id in sorted(self.ablation_configs.keys()):
            res = self.run_experiment(ab_id, dry_run=dry_run)
            results.append(res)
        return results
