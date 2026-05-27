"""Tier 2 deep Swin-based model with Ark+ pre-trained weights support.

This module implements the Ark+ model based on a Swin Transformer backbone,
fully complying with the BaseClassifier interface.
"""

from pathlib import Path
from typing import Literal, cast

import structlog
import timm
import torch
import torch.nn as nn

from core.interfaces.base_model import BaseClassifier
from core.models.factory import ModelFactory

log = structlog.get_logger(__name__)

ArkVariant = Literal["base", "small", "tiny"]


@ModelFactory.register("ark_plus")
class Tier2ArkPlus(BaseClassifier):
    """Tier 2 Ark+ Swin Transformer classifier implementation.

    Enables loading official Ark+ pre-trained weights or standard timm Swin
    weights, with progressive unfreezing and full MC/TTA support.
    """

    def __init__(
        self,
        num_classes: int = 2,
        variant: ArkVariant = "base",
        pretrained: bool = True,
        freeze_epochs: int = 5,
        ark_checkpoint_path: str | None = None,
    ) -> None:
        """Initialize the Tier 2 Ark+ model.

        Args:
            num_classes: Number of output prediction classes.
            variant: Swin variant name (e.g. 'base', 'small', 'tiny').
            pretrained: If True, load pre-trained ImageNet/Ark+ weights.
            freeze_epochs: Number of epochs to freeze the backbone for gradual unfreezing.
            ark_checkpoint_path: Optional path to the downloaded Ark+ .pth file.
        """
        super().__init__()
        self._variant = variant
        self._freeze_epochs = freeze_epochs
        self.backbone, feat_dim = self._build_backbone(variant, pretrained, ark_checkpoint_path)

        # Replace standard classifier head with thesis architecture
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, num_classes),
        )

        self._frozen = False
        if freeze_epochs > 0:
            self._freeze_backbone()

    def _build_backbone(
        self, variant: str, pretrained: bool, ark_path: str | None
    ) -> tuple[nn.Module, int]:
        """Construct the Swin backbone and load pre-trained weights.

        Args:
            variant: Swin variant string.
            pretrained: If True, attempt to load pre-trained weights.
            ark_path: Optional local path to the Ark+ pre-trained model.

        Returns:
            A tuple of (backbone_module, feature_dimension).
        """
        model_name = f"swin_{variant}_patch4_window7_224"
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if pretrained:
            if ark_path and Path(ark_path).exists():
                try:
                    state = torch.load(ark_path, map_location="cpu")
                    # If weights are wrapped in a dict
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    backbone.load_state_dict(state, strict=False)
                    log.info(f"[Tier2ArkPlus] Loaded Ark+ weights from {ark_path}")
                except Exception as e:
                    log.error(
                        f"[Tier2ArkPlus] Error loading Ark+ weights: {e}. "
                        "Falling back to ImageNet Swin."
                    )
            else:
                log.info(f"[Tier2ArkPlus] Fallback: ImageNet-pretrained Swin-{variant}")
        else:
            log.info(f"[Tier2ArkPlus] Initialized random-weight Swin-{variant}")

        feat_dim = getattr(backbone, "num_features", 1024)
        return backbone, feat_dim

    def _freeze_backbone(self) -> None:
        """Freeze all parameters in the backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze_at_epoch(self, current_epoch: int) -> None:
        """Dynamically unfreeze the backbone once target epoch is reached.

        Args:
            current_epoch: The active epoch number (1-indexed).
        """
        if self._frozen and current_epoch >= self._freeze_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self._frozen = False
            log.info(f"[Tier2ArkPlus] Backbone unfrozen at epoch {current_epoch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits.

        Args:
            x: Input image tensor.

        Returns:
            Logits tensor of shape [batch_size, num_classes].
        """
        feats = self.backbone(x)
        return cast(torch.Tensor, self.classifier(feats))

    def get_confidence(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute predicted class indices and corresponding confidence scores.

        Args:
            logits: Raw logits tensor of shape [batch_size, num_classes].

        Returns:
            A tuple of (confidences, predictions) tensors.
        """
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        return confidences, predictions

    def mc_forward(self, x: torch.Tensor, T: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform T forward passes with active dropout to estimate epistemic uncertainty.

        Args:
            x: Input image tensor.
            T: Number of stochastic passes.

        Returns:
            A tuple of (mean_probabilities, variances) tensors.
        """
        was_training = self.training
        self.train()

        outputs = []
        with torch.no_grad():
            for _ in range(T):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                outputs.append(probs.unsqueeze(0))

        outputs_tensor = torch.cat(outputs, dim=0)
        mean_probs = outputs_tensor.mean(dim=0)
        variance = outputs_tensor.var(dim=0)

        if not was_training:
            self.eval()

        return mean_probs, variance

    def tta_forward(self, x: torch.Tensor, n_augments: int = 10) -> torch.Tensor:
        """Perform Test-Time Augmentation (TTA) forward pass over multiple variations.

        Args:
            x: Input image tensor.
            n_augments: Number of augmented variations.

        Returns:
            Mean probabilities tensor of shape [batch_size, num_classes].
        """
        import torchvision.transforms as t_aug

        tta_transforms = t_aug.Compose(
            [
                t_aug.RandomHorizontalFlip(p=0.5),
                t_aug.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ]
        )

        was_training = self.training
        self.eval()

        outputs = []
        with torch.no_grad():
            for i in range(n_augments):
                curr_x = x if i == 0 else tta_transforms(x)
                logits = self.forward(curr_x)
                probs = torch.softmax(logits, dim=1)
                outputs.append(probs.unsqueeze(0))

        outputs_tensor = torch.cat(outputs, dim=0)
        mean_probs = outputs_tensor.mean(dim=0)

        if was_training:
            self.train()

        return mean_probs

    def mc_tta_forward(
        self, x: torch.Tensor, T: int = 20, n_augments: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine Monte Carlo Dropout and Test-Time Augmentation.

        Args:
            x: Input image tensor.
            T: Number of stochastic dropout runs per augmented image.
            n_augments: Number of augmented variations.

        Returns:
            A tuple of (mean_probabilities, variances) tensors.
        """
        import torchvision.transforms as t_aug

        tta_transforms = t_aug.Compose(
            [
                t_aug.RandomHorizontalFlip(p=0.5),
                t_aug.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ]
        )

        was_training = self.training
        self.train()

        outputs = []
        with torch.no_grad():
            for i in range(n_augments):
                curr_x = x if i == 0 else tta_transforms(x)
                for _ in range(T):
                    logits = self.forward(curr_x)
                    probs = torch.softmax(logits, dim=1)
                    outputs.append(probs.unsqueeze(0))

        outputs_tensor = torch.cat(outputs, dim=0)
        mean_probs = outputs_tensor.mean(dim=0)
        variance = outputs_tensor.var(dim=0)

        if not was_training:
            self.eval()

        return mean_probs, variance
