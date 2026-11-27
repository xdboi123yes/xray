"""Ablation study orchestrator for training and evaluation configurations."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def _override_to_env(key_path: str, value: Any) -> tuple[str, str]:
    """Translate dotted config key ``section.field`` into ``XRAY_SECTION__FIELD`` env var."""
    env_key = "XRAY_" + "__".join(part.upper() for part in key_path.split("."))
    env_val = str(value).lower() if isinstance(value, bool) else str(value)
    return env_key, env_val


class AblationRunner:
    """Executes registered ablation experiments with temporary YAML overrides."""

    def __init__(self, config_path: str = "config.yaml") -> None:
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
                    "--no-synthetic",
                ],
                "overrides": {"data.synthetic_augmentation": False},
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
                    "--no-synthetic",
                    "--no-classical-augmentation",
                ],
                "overrides": {"data.synthetic_augmentation": False},
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
                "overrides": {"model.mc_dropout_passes": 1, "model.tta_passes": 1},
            },
            "A12": {
                "name": "A12_ArkPlus_Only_MC_TTA",
                "description": "Tier 2 = Ark+ + MC + TTA",
                "script": "scripts/train_tier2.py",
                "args": ["--run-name", "A12_ArkPlus_Only_MC_TTA", "--backbone", "ark_plus"],
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
                "args": ["--run-name", "A14_CheXpert_ZeroShot", "--tier2-backbone", "ark_plus"],
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
                    "--mixup-cutmix",
                ],
                "overrides": {},
            },
        }

    def run_experiment(self, ablation_id: str, dry_run: bool = False) -> dict[str, Any]:
        """Run one registered ablation configuration, restoring YAML overrides afterward."""
        if ablation_id not in self.ablation_configs:
            raise ValueError(
                f"Unknown ablation ID: '{ablation_id}'. "
                f"Available configurations: {list(self.ablation_configs.keys())}"
            )

        config = self.ablation_configs[ablation_id]
        result: dict[str, Any] = {
            "ablation_id": ablation_id,
            "name": config["name"],
            "description": config["description"],
            "started_at": datetime.now().isoformat(),
            "returncode": None,
            "status": "FAILED",
        }

        overrides = dict(config.get("overrides", {}))
        if dry_run:
            overrides["training.epochs"] = 1

        child_env = os.environ.copy()
        for key_path, val in overrides.items():
            env_key, env_val = _override_to_env(key_path, val)
            child_env[env_key] = env_val

        try:
            cmd = [sys.executable, "-u", config["script"], *config.get("args", [])]
            log.info("Executing ablation", ablation_id=ablation_id, cmd=" ".join(cmd))
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=child_env,
            )
            if process.stdout is not None:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()

            returncode = process.wait()
            result["returncode"] = returncode
            if returncode == 0:
                result["status"] = "SUCCESS"
        except Exception as exc:
            log.error("Ablation execution failed", ablation_id=ablation_id, error=str(exc))
            result["error"] = str(exc)

        result["completed_at"] = datetime.now().isoformat()
        return result

    def run_all(self, dry_run: bool = False) -> list[dict[str, Any]]:
        """Execute every registered ablation configuration in sorted order."""
        return [
            self.run_experiment(ab_id, dry_run=dry_run)
            for ab_id in sorted(self.ablation_configs.keys())
        ]
