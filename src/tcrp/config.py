"""Type-safe configuration loader and mock-data integrity guard for TCR-P.

Loads YAML configuration from ``config/base.yaml`` with environment-specific
overrides (``APP_ENV``: ``dev``, ``prod``, ``test``) and ``XRAY_`` env vars.
Also hosts :func:`guard_mock` so placeholder data can never silently enter a
real research run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOW_MOCK_ENV = "XRAY_ALLOW_MOCK"


def mock_allowed() -> bool:
    """Return True only when ``XRAY_ALLOW_MOCK=1`` is set in the environment."""
    return os.getenv(ALLOW_MOCK_ENV) == "1"


def guard_mock(context: str) -> None:
    """Raise unless placeholder/mock data is explicitly permitted.

    Args:
        context: Human-readable description of the missing real input.

    Raises:
        RuntimeError: If ``XRAY_ALLOW_MOCK`` is not ``"1"``.
    """
    if not mock_allowed():
        raise RuntimeError(
            f"{context}: refusing to fabricate placeholder data in a real run "
            f"(a real input is probably missing). Set {ALLOW_MOCK_ENV}=1 to "
            f"explicitly allow mock/dry-run data."
        )


class ModelSettings(BaseModel):
    """Configuration parameters for Tier 1, Tier 2, routing, and uncertainty passes."""

    tier1_backbone: str = "mobilenet_v2"
    tier2_backbone: str = "efficientnet_b4"
    confidence_threshold: float = 0.75
    threshold_window_size: int = 50
    threshold_delta: float = 0.05
    mc_dropout_passes: int = 20
    tta_passes: int = 10


class TrainingSettings(BaseModel):
    """Hyperparameters for model training loops."""

    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    epochs: int = 50
    early_stopping_patience: int = 7
    seed: int = 42


class DataSettings(BaseModel):
    """Dataset split ratios, image resolution, and augmentation policy."""

    image_size: int = 224
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    synthetic_augmentation: bool = True


class AugmentationSettings(BaseModel):
    """Stable Diffusion and FID gate settings for Synthetic Augmentation Batches."""

    fid_threshold: float = 50.0
    sd_num_inference_steps: int = 50
    sd_guidance_scale: float = 7.5


class EvaluationSettings(BaseModel):
    """Settings for split-conformal calibration and temperature scaling."""

    conformal_coverage: float = 0.95
    ece_bins: int = 15
    calibration_temperature: float = 1.0


class PathSettings(BaseModel):
    """Filesystem paths for datasets, checkpoints, results, and paper figures."""

    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_synthetic: str = "data/synthetic"
    image_dir: str = "data/raw/images"
    models: str = "outputs/models"
    results: str = "outputs/results"
    figures: str = "outputs/figures"
    thesis_figures: str = "thesis/figures"
    logs: str = "logs"


def _resolve_config_dir() -> Path:
    """Locate the repository ``config/`` directory containing base and env YAMLs."""
    repo_candidate = Path(__file__).resolve().parents[2] / "config"
    if repo_candidate.is_dir():
        return repo_candidate
    return Path.cwd() / "config"


class Settings(BaseSettings):
    """Top-level settings loading hierarchical YAML configs with env overrides."""

    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    data: DataSettings = DataSettings()
    augmentation: AugmentationSettings = AugmentationSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    paths: PathSettings = PathSettings()

    model_config = SettingsConfigDict(
        env_prefix="XRAY_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **values: Any) -> None:
        """Load ``base.yaml``, merge environment overrides, then explicit kwargs."""
        env = os.getenv("APP_ENV", "dev").lower()
        config_dir = _resolve_config_dir()
        base_path = config_dir / "base.yaml"
        env_path = config_dir / f"{env}.yaml"

        config_dict: dict[str, Any] = {}

        if base_path.exists():
            with open(base_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_dict = loaded

        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                env_loaded = yaml.safe_load(f)
                if isinstance(env_loaded, dict):
                    for section, values_dict in env_loaded.items():
                        if (
                            section in config_dict
                            and isinstance(config_dict[section], dict)
                            and isinstance(values_dict, dict)
                        ):
                            config_dict[section].update(values_dict)
                        else:
                            config_dict[section] = values_dict

        for section, values_dict in values.items():
            if (
                section in config_dict
                and isinstance(config_dict[section], dict)
                and isinstance(values_dict, dict)
            ):
                config_dict[section].update(values_dict)
            else:
                config_dict[section] = values_dict

        super().__init__(**config_dict)


_settings = Settings()


def get_settings() -> Settings:
    """Return the global Settings singleton."""
    return _settings
