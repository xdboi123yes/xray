"""Production-grade training engine for tiered Chest X-Ray models.

Implements the observer-aware training loop featuring AMP (mixed precision),
gradient accumulation, gradient clipping, and robust lifecycle hooks.
"""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.interfaces.base_model import BaseClassifier
from core.interfaces.base_observer import TrainingObserver


class Trainer:
    """Production training engine for Chest X-Ray classifiers.

    Orchestrates the training and validation epochs, processes batches with
    mixed precision (AMP), clips gradients, accumulates gradients, and publishes
    lifecycle events to registered observers.
    """

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
    ) -> None:
        """Initialize the Trainer.

        Args:
            model: Classifier to train.
            optimizer: Optimizer instance.
            criterion: Loss function module.
            device: Training hardware device.
            config: General parameter configuration.
            scheduler: LR scheduler (optional).
            use_amp: If True, use mixed precision training (AMP).
            accumulate_grad_batches: Number of batches to aggregate gradients.
            gradient_clip_val: Threshold value for gradient norm clipping.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.use_amp = use_amp and (device.type == "cuda")
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_val = gradient_clip_val

        # Setup PyTorch AMP scaler if GPU supports it
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self._observers: list[TrainingObserver] = []
        self.current_epoch = 0
        self.best_metric = 0.0
        self.stop_training = False  # Set dynamically by early stopping observer

    def add_observer(self, obs: TrainingObserver) -> None:
        """Register a training observer.

        Args:
            obs: An observer implementing TrainingObserver hooks.
        """
        self._observers.append(obs)

    def train(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        resume_from: str | None = None,
    ) -> dict[str, float]:
        """Execute the complete training process over configured epochs.

        Args:
            train_loader: Dataloader containing training instances.
            val_loader: Dataloader containing validation instances.
            resume_from: Path to checkpoint file to restore state (optional).

        Returns:
            A dictionary containing final validation epoch metrics.
        """
        self.stop_training = False

        # If resume checkpoint path is provided, restore complete state
        start_epoch = 0
        if resume_from:
            start_epoch = self._load_checkpoint(resume_from)

        # Notify observers: on_train_start
        for obs in self._observers:
            obs.on_train_start(self)

        epochs = self.config.get("training", {}).get("epochs", 50)
        final_metrics: dict[str, float] = {}

        for epoch in range(start_epoch, epochs):
            if self.stop_training:
                break

            self.current_epoch = epoch + 1

            # Support progressive backbone unfreezing if implemented by the model
            if hasattr(self.model, "unfreeze_at_epoch"):
                self.model.unfreeze_at_epoch(self.current_epoch)  # type: ignore[operator]

            # Notify observers: on_epoch_start
            for obs in self._observers:
                obs.on_epoch_start(self.current_epoch, self)

            # 1. Train epoch
            train_metrics = self._train_epoch(train_loader)

            # 2. Validation epoch
            val_metrics = self._val_epoch(val_loader)

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            final_metrics = val_metrics

            # Print epoch summary
            print(
                f"\n✨ [Epoch {self.current_epoch:02d}/{epochs}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | Train AUC: {train_metrics['train_auc']:.4f} | Train Acc: {train_metrics['train_acc']:.4f} || "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Val AUC: {val_metrics['val_auc']:.4f} | Val Acc: {val_metrics['val_acc']:.4f}",
                flush=True
            )

            # Adjust learning rate scheduler if present
            if self.scheduler is not None:
                # Handle ReduceLROnPlateau which requires validation metric
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    val_auc = val_metrics.get("val_auc", 0.0)
                    self.scheduler.step(val_auc)
                else:
                    self.scheduler.step()

            # Notify observers: on_validation_end
            for obs in self._observers:
                obs.on_validation_end(self.current_epoch, val_metrics, self)

            # Notify observers: on_epoch_end
            for obs in self._observers:
                obs.on_epoch_end(self.current_epoch, epoch_metrics, self)

        # Notify observers: on_train_end
        for obs in self._observers:
            obs.on_train_end(self)

        return final_metrics

    def _train_epoch(self, loader: DataLoader[Any]) -> dict[str, float]:
        """Perform one training epoch over batches.

        Args:
            loader: Training Dataloader.

        Returns:
            Dictionary of epoch training metrics.
        """
        self.model.train()
        total_loss = 0.0
        preds_list: list[float] = []
        targets_list: list[float] = []

        self.optimizer.zero_grad()

        from tqdm import tqdm
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {self.current_epoch:02d} [Train]", leave=True)
        for batch_idx, batch in pbar:
            # Safe unpack: loader might yield (images, labels) or (images, labels, metadata)
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            # Auto-cast context for float16 mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                # Scale loss according to gradient accumulation steps
                loss = loss / self.accumulate_grad_batches

            # Backward pass with optional scaling
            self.scaler.scale(loss).backward()

            # Step optimizer every accumulate_grad_batches steps
            if (batch_idx + 1) % self.accumulate_grad_batches == 0 or (
                batch_idx + 1
            ) == len(loader):
                if self.gradient_clip_val is not None:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss = loss.item() * self.accumulate_grad_batches
            total_loss += running_loss * len(images)
            pbar.set_postfix(loss=f"{running_loss:.4f}")

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds_list.extend(probs.detach().cpu().numpy().tolist())
            targets_list.extend(labels.detach().cpu().numpy().tolist())

        from sklearn.metrics import accuracy_score, roc_auc_score

        epoch_loss = total_loss / len(loader.dataset)  # type: ignore
        try:
            auc = float(roc_auc_score(targets_list, preds_list))
        except ValueError:
            auc = 0.5  # Fallback if batch contains only one class

        binary_preds = [1 if p > 0.5 else 0 for p in preds_list]
        acc = float(accuracy_score(targets_list, binary_preds))

        return {"train_loss": epoch_loss, "train_auc": auc, "train_acc": acc}

    def _val_epoch(self, loader: DataLoader[Any]) -> dict[str, float]:
        """Perform one validation epoch over batches.

        Args:
            loader: Validation Dataloader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        preds_list: list[float] = []
        targets_list: list[float] = []

        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch:02d} [Val]", leave=True)
        with torch.no_grad():
            for batch in pbar:
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * len(images)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds_list.extend(probs.cpu().numpy().tolist())
                targets_list.extend(labels.cpu().numpy().tolist())

        from sklearn.metrics import accuracy_score, roc_auc_score

        epoch_loss = total_loss / len(loader.dataset)  # type: ignore
        try:
            auc = float(roc_auc_score(targets_list, preds_list))
        except ValueError:
            auc = 0.5

        binary_preds = [1 if p > 0.5 else 0 for p in preds_list]
        acc = float(accuracy_score(targets_list, binary_preds))

        return {"val_loss": epoch_loss, "val_auc": auc, "val_acc": acc}

    def _load_checkpoint(self, path: str) -> int:
        """Internal helper to delegate loading checkpoints.

        Args:
            path: Checkpoint file path.

        Returns:
            The starting epoch number (0-indexed).
        """
        from infrastructure.persistence.checkpoint import CheckpointManager

        return CheckpointManager.load_state(path, self)
