"""Training engine, checkpoint serialization, weighted loss, scheduler, and runner."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd
import structlog
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from tcrp.config import Settings, get_settings
from tcrp.data.augmentation import ClassicalAugmentation, MixupCutmixAugmentation
from tcrp.data.datasets import NIHChestXrayDataset
from tcrp.models.backbones import ModelFactory
from tcrp.models.base import BaseClassifier
from tcrp.training.observers import (
    CarbonTrackerObserver,
    CheckpointObserver,
    EarlyStoppingObserver,
    HistoryObserver,
    LRLoggerObserver,
    MLflowObserver,
    TrainingObserver,
)

log = structlog.get_logger(__name__)


class TrainingConfig(BaseModel):
    """Hyperparameters for a Tier 1 or Tier 2 training run."""

    backbone: str = Field(..., description="Registered ModelFactory backbone key.")
    run_name: str = Field(..., description="Unique run name for checkpoints and MLflow.")
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(50, ge=1)
    lr_backbone: float = Field(1e-4, gt=0.0)
    lr_head: float = Field(1e-3, gt=0.0)
    early_stopping_patience: int = Field(7, ge=1)
    seed: int = Field(42)
    use_synthetic: bool = Field(True)
    use_classical_augmentation: bool = Field(True)
    use_mixup_cutmix: bool = Field(False)
    num_workers: int = Field(0, ge=0)


class CheckpointManager:
    """Saves and restores complete training snapshots (model, optimizer, scheduler, epoch)."""

    @classmethod
    def save_state(cls, path: str, trainer: Any) -> None:
        """Serialize model, optimizer, scheduler, and metadata to ``path``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            "epoch": trainer.current_epoch,
            "best_metric": getattr(trainer, "best_metric", 0.0),
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "config": trainer.config,
        }
        torch.save(state, path)

    @classmethod
    def load_state(cls, path: str, trainer: Any) -> int:
        """Restore training state into ``trainer`` and return the saved epoch index."""
        log.info("Restoring training checkpoint", path=path)
        state = torch.load(path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(state["model_state_dict"])
        trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        if trainer.scheduler and state.get("scheduler_state_dict"):
            trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
        trainer.best_metric = state.get("best_metric", 0.0)
        return int(state.get("epoch", 0))


def get_class_weighted_loss(
    csv_path: str,
    target_classes: Sequence[str] = ("No Finding", "Pneumothorax"),
) -> nn.CrossEntropyLoss:
    """Compute inverse-frequency binary class weights from ``csv_path``."""
    df = pd.read_csv(csv_path)
    if "Label" in df.columns:
        count_0 = int((df["Label"] == 0).sum()) or 1
        count_1 = int((df["Label"] == 1).sum()) or 1
    else:
        df_filtered = df[df["Finding Labels"].isin(target_classes)]
        class_counts = df_filtered["Finding Labels"].value_counts()
        count_0 = int(class_counts.get("No Finding", 1))
        count_1 = int(class_counts.get("Pneumothorax", 1))

    total = count_0 + count_1
    w0 = total / count_0
    w1 = total / count_1
    denom = w0 + w1
    weights = torch.tensor([(w0 / denom) * 2.0, (w1 / denom) * 2.0], dtype=torch.float32)
    log.info("Computed class weights", no_finding=float(weights[0]), pneumothorax=float(weights[1]))
    return nn.CrossEntropyLoss(weight=weights)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any] | Settings | None = None,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Create a ``ReduceLROnPlateau`` scheduler monitoring validation AUC."""
    if config is None:
        settings = get_settings()
    elif isinstance(config, dict):
        settings = Settings(**config)
    else:
        settings = config

    patience = getattr(settings.training, "early_stopping_patience", 7) // 2
    if patience < 1:
        patience = 3

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience, min_lr=1e-6
    )


class Trainer:
    """Observer-aware PyTorch training engine with AMP, gradient clipping, and resume."""

    def __init__(
        self,
        model: BaseClassifier,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: dict[str, Any],
        scheduler: Any = None,
        use_amp: bool = True,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float | None = 1.0,
        batch_augmentation: Any = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.use_amp = use_amp and (device.type == "cuda")
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val
        self.batch_augmentation = batch_augmentation

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)  # type: ignore[attr-defined]
        self._observers: list[TrainingObserver] = []
        self.current_epoch = 0
        self.best_metric = 0.0
        self.stop_training = False

    def add_observer(self, obs: TrainingObserver) -> None:
        """Attach a :class:`TrainingObserver` to receive lifecycle events."""
        self._observers.append(obs)

    def train(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        resume_from: str | None = None,
    ) -> dict[str, float]:
        """Run training and validation epochs, returning the final validation metrics."""
        self.stop_training = False
        start_epoch = CheckpointManager.load_state(resume_from, self) if resume_from else 0

        for obs in self._observers:
            obs.on_train_start(self)

        epochs = int(self.config.get("training", {}).get("epochs", 50))
        final_metrics: dict[str, float] = {}

        for epoch in range(start_epoch, epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch + 1
            if hasattr(self.model, "unfreeze_at_epoch"):
                self.model.unfreeze_at_epoch(self.current_epoch)  # type: ignore[operator]

            for obs in self._observers:
                obs.on_epoch_start(self.current_epoch, self)

            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._val_epoch(val_loader)
            epoch_metrics = {**train_metrics, **val_metrics}
            final_metrics = val_metrics

            log.info("Epoch complete", epoch=self.current_epoch, total_epochs=epochs, **epoch_metrics)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_auc", 0.0))
                else:
                    self.scheduler.step()

            for obs in self._observers:
                obs.on_validation_end(self.current_epoch, val_metrics, self)
            for obs in self._observers:
                obs.on_epoch_end(self.current_epoch, epoch_metrics, self)

        for obs in self._observers:
            obs.on_train_end(self)

        return final_metrics

    def _train_epoch(self, loader: DataLoader[Any]) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        preds_list: list[float] = []
        targets_list: list[float] = []
        self.optimizer.zero_grad()

        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {self.current_epoch:02d} [Train]",
            leave=True,
            disable=not sys.stdout.isatty(),
        )
        for batch_idx, batch in pbar:
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            labels_a, labels_b, mix_lam = labels, labels, 1.0
            if self.batch_augmentation is not None:
                images, labels_a, labels_b, mix_lam = self.batch_augmentation.apply_batch(
                    images, labels
                )

            with torch.amp.autocast("cuda", enabled=self.use_amp):  # type: ignore[attr-defined]
                outputs = self.model(images)
                loss = (
                    mix_lam * self.criterion(outputs, labels_a)
                    + (1.0 - mix_lam) * self.criterion(outputs, labels_b)
                ) / self.accumulate_grad_batches

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulate_grad_batches == 0 or (batch_idx + 1) == len(loader):
                if self.gradient_clip_val is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss = loss.item() * self.accumulate_grad_batches
            batch_n = len(images)
            total_loss += running_loss * batch_n
            n_samples += batch_n
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds_list.extend(probs.detach().cpu().numpy().tolist())
            targets_list.extend(labels.detach().cpu().numpy().tolist())

        epoch_loss = total_loss / n_samples if n_samples > 0 else 0.0
        try:
            auc = float(roc_auc_score(targets_list, preds_list))
        except ValueError:
            auc = 0.5
        acc = (
            float(accuracy_score(targets_list, [1 if p > 0.5 else 0 for p in preds_list]))
            if targets_list
            else 0.0
        )
        return {"train_loss": epoch_loss, "train_auc": auc, "train_acc": acc}

    def _val_epoch(self, loader: DataLoader[Any]) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        preds_list: list[float] = []
        targets_list: list[float] = []

        pbar = tqdm(
            loader,
            desc=f"Epoch {self.current_epoch:02d} [Val]",
            leave=True,
            disable=not sys.stdout.isatty(),
        )
        with torch.no_grad():
            for batch in pbar:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                with torch.amp.autocast("cuda", enabled=self.use_amp):  # type: ignore[attr-defined]
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                batch_n = len(images)
                total_loss += loss.item() * batch_n
                n_samples += batch_n
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds_list.extend(probs.cpu().numpy().tolist())
                targets_list.extend(labels.cpu().numpy().tolist())

        epoch_loss = total_loss / n_samples if n_samples > 0 else 0.0
        try:
            auc = float(roc_auc_score(targets_list, preds_list))
        except ValueError:
            auc = 0.5
        acc = (
            float(accuracy_score(targets_list, [1 if p > 0.5 else 0 for p in preds_list]))
            if targets_list
            else 0.0
        )
        return {"val_loss": epoch_loss, "val_auc": auc, "val_acc": acc}


def train_model(
    config: TrainingConfig,
    train_csv: str = "data/processed/train.csv",
    val_csv: str = "data/processed/val.csv",
    resume_checkpoint: str | None = None,
    dry_run: bool = False,
    settings: Settings | None = None,
    device: torch.device | None = None,
    pretrained: bool = True,
    image_dir: str | None = None,
    enable_external_observers: bool = True,
) -> dict[str, Any]:
    """High-level training orchestrator wiring datasets, optimizer, observers, and Trainer."""
    active_settings = settings or get_settings()
    os.makedirs(active_settings.paths.models, exist_ok=True)

    active_device = device or torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    if active_device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = ModelFactory.create(config.backbone, pretrained=pretrained).to(active_device)

    train_transform = ClassicalAugmentation(
        image_size=active_settings.data.image_size,
        is_training=config.use_classical_augmentation,
    ).pipeline
    val_transform = ClassicalAugmentation(
        image_size=active_settings.data.image_size,
        is_training=False,
    ).pipeline

    train_dataset = NIHChestXrayDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        transform=train_transform,
        use_synthetic=config.use_synthetic,
        synthetic_dir=os.path.join(active_settings.paths.data_synthetic, "accepted"),
        synthetic_csv=os.path.join(active_settings.paths.data_synthetic, "synthetic_metadata.csv"),
    )
    val_dataset = NIHChestXrayDataset(
        csv_file=val_csv,
        image_dir=image_dir,
        transform=val_transform,
    )

    num_workers = config.num_workers
    if num_workers == 0 and active_device.type == "cuda":
        num_workers = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(active_device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(active_device.type == "cuda"),
    )

    classifier_layer = getattr(model, "classifier", getattr(model.backbone, "classifier", None))
    if classifier_layer is None:
        raise AttributeError("Could not locate classifier head on the model.")

    classifier_param_ids = {id(p) for p in classifier_layer.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in classifier_param_ids]
    classifier_params = list(classifier_layer.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.lr_backbone},
            {"params": classifier_params, "lr": config.lr_head},
        ],
        weight_decay=1e-4,
    )
    scheduler = get_scheduler(optimizer, active_settings)
    criterion = get_class_weighted_loss(train_csv).to(active_device)

    epochs = 1 if dry_run else config.epochs
    patience = 1 if dry_run else config.early_stopping_patience
    run_config = {
        "training": {"epochs": epochs, "batch_size": config.batch_size, "seed": config.seed},
        "model": {"backbone": config.backbone},
        "paths": {"models": active_settings.paths.models},
    }

    batch_aug = MixupCutmixAugmentation() if config.use_mixup_cutmix else None
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=active_device,
        config=run_config,
        scheduler=scheduler,
        batch_augmentation=batch_aug,
    )

    mlflow_obs: MLflowObserver | None = None
    if enable_external_observers:
        mlflow_obs = MLflowObserver(
            experiment_name="Chest_XRay_Classification",
            run_name=config.run_name,
        )
        trainer.add_observer(mlflow_obs)
        trainer.add_observer(CarbonTrackerObserver())

    trainer.add_observer(EarlyStoppingObserver(patience=patience, monitor="val_auc", mode="max"))
    trainer.add_observer(LRLoggerObserver())

    checkpoint_dir = os.path.join(active_settings.paths.models, config.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    history_obs = HistoryObserver(
        history_path=history_path,
        metadata={
            "run_name": config.run_name,
            "backbone": config.backbone,
            "protocol": "patient-disjoint-v1",
            "device": str(active_device),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "use_synthetic": config.use_synthetic,
            "use_classical_augmentation": config.use_classical_augmentation,
            "use_mixup_cutmix": config.use_mixup_cutmix,
            "batch_size": config.batch_size,
            "lr_backbone": config.lr_backbone,
            "lr_head": config.lr_head,
            "epochs_configured": epochs,
            "early_stopping_patience": patience,
            "seed": config.seed,
        },
        monitor="val_auc",
        mode="max",
    )
    trainer.add_observer(
        CheckpointObserver(checkpoint_path=checkpoint_path, monitor="val_auc", mode="max")
    )
    trainer.add_observer(history_obs)

    log.info(
        "Starting training run",
        run_name=config.run_name,
        backbone=config.backbone,
        device=str(active_device),
        epochs=epochs,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
    )
    final_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from=resume_checkpoint,
    )

    return {
        "backbone": config.backbone,
        "run_name": config.run_name,
        "mlflow_run_id": getattr(mlflow_obs, "run_id", None) or "local_run",
        "best_val_auc": trainer.best_metric,
        "final_epoch": trainer.current_epoch,
        "model_weight_path": checkpoint_path,
        "history_path": history_path,
        "wall_clock_seconds": history_obs.wall_clock_seconds,
        "peak_vram_mb": history_obs.peak_vram_mb,
        "final_metrics": final_metrics,
    }
