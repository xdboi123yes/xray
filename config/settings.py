"""Type-safe configuration loader for the Chest X-Ray classification pipeline.

Utilizes pydantic-settings to load YAML configurations with environment-specific
overrides and environment variable injection support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseModel):
    """Configuration parameters for Tier 1 and Tier 2 deep learning models."""
    tier1_backbone: str = "mobilenet_v2"
    tier2_backbone: str = "efficientnet_b4"
    confidence_threshold: float = 0.75
    threshold_window_size: int = 50
    threshold_delta: float = 0.05
    mc_dropout_passes: int = 20
    tta_passes: int = 10


class TrainingSettings(BaseModel):
    """Hyperparameters and configuration for model training loops."""
    batch_size: int = 32
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    epochs: int = 50
    early_stopping_patience: int = 7
    seed: int = 42


class DataSettings(BaseModel):
    """Dataset configurations including splits and augmentation policies."""
    image_size: int = 224
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    synthetic_augmentation: bool = True


class AugmentationSettings(BaseModel):
    """Stable Diffusion and FID thresholds for synthetic data generation."""
    fid_threshold: float = 50.0
    sd_num_inference_steps: int = 50
    sd_guidance_scale: float = 7.5


class EvaluationSettings(BaseModel):
    """Settings for conformal prediction, calibration, and ECE bins."""
    conformal_coverage: float = 0.95
    ece_bins: int = 10
    calibration_temperature: float = 1.0


class PathSettings(BaseModel):
    """System paths for datasets, models, logs, results, and thesis assets."""
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_synthetic: str = "data/synthetic"
    image_dir: str = "data/raw/images"
    models: str = "outputs/models"
    results: str = "outputs/results"
    figures: str = "outputs/figures"
    thesis_figures: str = "thesis/figures"
    logs: str = "logs"


class SecuritySettings(BaseModel):
    """Security configurations including CORS origins."""
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


class Settings(BaseSettings):
    """Top-level settings manager loading hierarchal yaml configs with env overrides."""
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    data: DataSettings = DataSettings()
    augmentation: AugmentationSettings = AugmentationSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    paths: PathSettings = PathSettings()
    security: SecuritySettings = SecuritySettings()

    model_config = SettingsConfigDict(
        env_prefix="XRAY_",
        env_nested_delimiter="__",
        case_sensitive=False
    )

    def __init__(self, **values: Any) -> None:
        """Initializes settings by loading base and environment YAML configurations.

        Args:
            **values: Explicit override values for initialization.
        """
        # Determine environment: dev (default), prod, test
        env = os.getenv("APP_ENV", "dev").lower()

        # Locate config directory relative to this file
        config_dir = Path(__file__).parent
        base_path = config_dir / "base.yaml"
        env_path = config_dir / f"{env}.yaml"

        config_dict: dict[str, Any] = {}

        # 1. Load base.yaml
        if base_path.exists():
            with open(base_path, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_dict = loaded

        # 2. Merge environment overrides (e.g. dev.yaml, test.yaml)
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

        # 3. Merge manual overrides passed via keyword arguments
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


# Global singleton instance of Settings
_settings = Settings()


def get_settings() -> Settings:
    """Returns the global Settings singleton instance."""
    return _settings
