"""CLI script to train Tier 2 Models (EfficientNet / Ark+).

Leverages type-safe configurations and orchestrates the training run 
through the clean application service layer.
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to sys.path to support standalone execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from application.dto.training_config_dto import TrainingConfigDTO
from application.services.training_service import TrainingService


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tier 2 Model")
    parser.add_argument(
        "--run-name",
        type=str,
        default="Tier2_EfficientNetB4",
        help="MLflow run name",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="efficientnet_b4",
        choices=["efficientnet_b4", "ark_plus"],
        help="Tier 2 backbone network model type",
    )
    parser.add_argument(
        "--no-mc-tta",
        action="store_true",
        help="Disable MC dropout and TTA (inference-only compatibility flag).",
    )
    args = parser.parse_args()

    # DTO mapping for Tier 2 backbone network
    dto = TrainingConfigDTO(
        backbone=args.backbone,
        run_name=args.run_name,
        batch_size=32,
        lr_backbone=1e-4,
        lr_head=1e-3,
        epochs=50,
        early_stopping_patience=7,
        seed=42,
        use_synthetic=True,
    )

    # Delegate model configuration and training execution to application layer
    service = TrainingService()
    service.train_model(dto)


if __name__ == "__main__":
    main()
