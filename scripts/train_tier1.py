"""CLI script to train Tier 1 Model (MobileNetV2).

Leverages type-safe configurations and orchestrates the training run 
through the clean application service layer.
"""

from __future__ import annotations

import argparse

from application.dto.training_config_dto import TrainingConfigDTO
from application.services.training_service import TrainingService


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tier 1 Model")
    parser.add_argument(
        "--run-name",
        type=str,
        default="Tier1_MobileNetV2",
        help="MLflow run name",
    )
    args = parser.parse_args()

    # DTO mapping for MobileNetV2 Tier 1 backbone
    dto = TrainingConfigDTO(
        backbone="mobilenet_v2",
        run_name=args.run_name,
        batch_size=32,
        lr_backbone=1e-4,
        lr_head=1e-3,
        epochs=50,
        early_stopping_patience=7,
        seed=42,
        use_synthetic=False,
    )

    # Delegate model configuration and training execution to application layer
    service = TrainingService()
    service.train_model(dto)


if __name__ == "__main__":
    main()
