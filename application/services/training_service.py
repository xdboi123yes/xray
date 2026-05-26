"""Application Service orchestrating the training cycles of deep learning classifiers.

Wires together PyTorch datasets, optimizers, learning rate schedulers,
hardware accelerators, and infrastructure-level logging observers.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from application.dto.training_config_dto import TrainingConfigDTO
from config.settings import get_settings
from core.models.factory import ModelFactory

# Explicitly import all classifiers to trigger registration decorators in ModelFactory
import core.models.tier1_mobilenet
import core.models.tier2_efficientnet
import core.models.tier2_ark
from infrastructure.data.dataset import NIHChestXrayDataset
from infrastructure.training.losses import get_class_weighted_loss
from infrastructure.training.observers.carbon_tracker import CarbonTrackerObserver
from infrastructure.training.observers.checkpoint_observer import CheckpointObserver
from infrastructure.training.observers.early_stopping import EarlyStoppingObserver
from infrastructure.training.observers.lr_logger import LRLoggerObserver
from infrastructure.training.observers.mlflow_observer import MLflowObserver
from infrastructure.training.scheduler import get_scheduler
from infrastructure.training.trainer import Trainer

import structlog

log = structlog.get_logger(__name__)


class TrainingService:
    """Orchestrates model configuration, hyperparameter setup, and hardware training loop."""

    def __init__(self) -> None:
        """Initializes the TrainingService and ensures model folders exist."""
        self.settings = get_settings()
        os.makedirs(self.settings.paths.models, exist_ok=True)

    def train_model(
        self,
        config_dto: TrainingConfigDTO,
        train_csv: str = "data/processed/train.csv",
        val_csv: str = "data/processed/val.csv",
        resume_checkpoint: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Orchestrates hyperparameter setups and starts the trainer cycle.

        Args:
            config_dto: Training hyperparameter config parameters.
            train_csv: CSV path containing training X-ray indices and labels.
            val_csv: CSV path containing validation X-ray indices and labels.
            resume_checkpoint: Optional checkpoint weight file to restore training.
            dry_run: If True, forces training to complete in exactly 1 epoch.

        Returns:
            A metadata dict containing final metrics, weight path, and MLflow tags.
        """
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        log.info(
            f"[TrainingService] Initializing training for '{config_dto.backbone}' "
            f"on device '{device}'..."
        )

        # 1. Instantiate the Model Backbone using ModelFactory
        model = ModelFactory.create(config_dto.backbone)
        model.to(device)

        # 2. Build training and validation datasets with proper transforms (resizing, augmentations, normalization)
        from core.augmentation.classical import ClassicalAugmentation

        train_transform = ClassicalAugmentation(
            image_size=self.settings.data.image_size, is_training=True
        )._pipeline
        val_transform = ClassicalAugmentation(
            image_size=self.settings.data.image_size, is_training=False
        )._pipeline

        train_dataset = NIHChestXrayDataset(
            csv_file=train_csv,
            transform=train_transform,
            use_synthetic=config_dto.use_synthetic,
            synthetic_dir=self.settings.paths.data_synthetic,
            synthetic_csv=os.path.join(self.settings.paths.data_synthetic, "synthetic.csv"),
        )
        val_dataset = NIHChestXrayDataset(
            csv_file=val_csv,
            transform=val_transform,
        )

        # Dynamic number of workers to prevent CPU/IO bottleneck on high-performance GPUs
        num_workers = 4
        if device.type == "cuda":
            num_workers = min(8, os.cpu_count() or 4)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config_dto.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config_dto.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # 3. Create Optimizer and Scheduler
        # Dynamic disjoint separation of backbone and classifier parameters by ID
        classifier_layer = getattr(model, "classifier", getattr(model.backbone, "classifier", None))
        if classifier_layer is None:
            raise AttributeError("Could not locate classifier head on the model.")

        classifier_param_ids = set(id(p) for p in classifier_layer.parameters())
        backbone_params = [p for p in model.parameters() if id(p) not in classifier_param_ids]
        classifier_params = list(classifier_layer.parameters())

        params = [
            {"params": backbone_params, "lr": config_dto.lr_backbone},
            {"params": classifier_params, "lr": config_dto.lr_head},
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=1e-4)

        # Custom ReduceLROnPlateau scheduler
        scheduler = get_scheduler(optimizer, self.settings)

        # 4. Class-weighted Loss Criterion
        criterion = get_class_weighted_loss(train_csv)
        criterion = criterion.to(device)

        # 5. Initialize training runtime settings dictionary
        epochs = 1 if dry_run else config_dto.epochs
        patience = 1 if dry_run else config_dto.early_stopping_patience

        run_config = {
            "training": {
                "epochs": epochs,
                "batch_size": config_dto.batch_size,
                "seed": config_dto.seed,
            },
            "model": {
                "backbone": config_dto.backbone,
            },
            "paths": {
                "models": self.settings.paths.models,
            },
        }

        # 6. Instantiate PyTorch Trainer Engine
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=run_config,
            scheduler=scheduler,
        )

        # 7. Wire Infrastructure Observers
        # MLflow logger
        mlflow_obs = MLflowObserver(
            experiment_name="Chest_XRay_Classification",
            run_name=config_dto.run_name,
        )
        trainer.add_observer(mlflow_obs)

        # Early Stopping observer
        early_stopping = EarlyStoppingObserver(
            patience=patience,
            monitor="val_auc",
            mode="max",
        )
        trainer.add_observer(early_stopping)

        # LR Logger observer
        lr_logger = LRLoggerObserver()
        trainer.add_observer(lr_logger)

        # Carbon footprint observer
        carbon_tracker = CarbonTrackerObserver()
        trainer.add_observer(carbon_tracker)

        # Model Checkpoint saving observer
        checkpoint_dir = os.path.join(self.settings.paths.models, config_dto.run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")

        checkpoint_obs = CheckpointObserver(
            checkpoint_path=checkpoint_path,
            monitor="val_auc",
            mode="max",
        )

        trainer.add_observer(checkpoint_obs)

        # 8. Start training cycle
        log.info(f"[TrainingService] Executing training loop for {epochs} epochs...")
        final_metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from=resume_checkpoint,
        )

        # Retrieve MLflow Run information if logged successfully
        mlflow_run_id = getattr(mlflow_obs, "run_id", "local_run")

        results = {
            "backbone": config_dto.backbone,
            "run_name": config_dto.run_name,
            "mlflow_run_id": mlflow_run_id,
            "best_val_auc": trainer.best_metric,
            "final_epoch": trainer.current_epoch,
            "model_weight_path": checkpoint_path,
            "final_metrics": final_metrics,
        }

        log.info(
            f"[TrainingService] Training completed successfully. "
            f"Best Val AUC: {trainer.best_metric:.4f}. Weights saved at {checkpoint_path}."
        )

        return results
