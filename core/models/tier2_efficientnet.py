"""Tier 2 deep and robust classification model (EfficientNetB4).

This module implements the escalated classifier path using EfficientNetB4,
featuring MC Dropout and Test-Time Augmentation (TTA).
"""

from typing import cast

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from core.interfaces.base_model import BaseClassifier
from core.models.factory import ModelFactory


@ModelFactory.register("efficientnet_b4")
class Tier2EfficientNet(BaseClassifier):
    """Tier 2 EfficientNetB4 classifier implementation.

    Designed for high-accuracy predictions. Implements Monte Carlo Dropout and
    Test-Time Augmentation (TTA) to calculate epistemic and aleatoric uncertainties.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        """Initialize the Tier 2 EfficientNetB4 model.

        Args:
            num_classes: Number of output prediction classes.
            pretrained: If True, load ImageNet pre-trained weights.
        """
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b4(weights=weights)

        # Replace standard classifier head with thesis architecture
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits.

        Args:
            x: Input image tensor.

        Returns:
            Logits tensor of shape [batch_size, num_classes].
        """
        return cast(torch.Tensor, self.backbone(x))

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

    def _enable_dropout_only(self) -> None:
        """Keep normalization layers in eval mode while enabling Dropout layers for MC sampling."""
        self.eval()
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    def mc_forward(self, x: torch.Tensor, T: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform T forward passes with active dropout to estimate epistemic uncertainty.

        Keeps BatchNorm layers in evaluation mode so running statistics are preserved
        even at batch size 1.

        Args:
            x: Input image tensor.
            T: Number of stochastic passes.

        Returns:
            A tuple of (mean_probabilities, variances) tensors.
        """
        was_training = self.training
        self._enable_dropout_only()

        outputs = []
        with torch.no_grad():
            for _ in range(T):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                outputs.append(probs.unsqueeze(0))

        # Shape: [T, batch_size, num_classes]
        outputs_tensor = torch.cat(outputs, dim=0)
        mean_probs = outputs_tensor.mean(dim=0)
        variance = outputs_tensor.var(dim=0, unbiased=False)

        if was_training:
            self.train()
        else:
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

    def mc_tta_forward_decomposed(
        self, x: torch.Tensor, T: int = 20, n_augments: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine MC Dropout and TTA with Law-of-Total-Variance decomposition.

        For K=n_augments test-time views and T stochastic dropout passes per view:
            Var_total(p) = E_k[Var_t(p | a_k)] + Var_k[E_t(p | a_k)]
                         = epistemic_var        + aleatoric_tta_var

        Args:
            x: Input image tensor.
            T: Number of stochastic dropout runs per augmented image.
            n_augments: Number of augmented variations.

        Returns:
            Tuple of (mean_probs, total_var, epistemic_var, aleatoric_tta_var).
        """
        import torchvision.transforms as t_aug

        tta_transforms = t_aug.Compose(
            [
                t_aug.RandomHorizontalFlip(p=0.5),
                t_aug.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ]
        )

        was_training = self.training
        self._enable_dropout_only()

        # Shape across views: [K, T, batch_size, num_classes]
        view_outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(n_augments):
                curr_x = x if i == 0 else tta_transforms(x)
                mc_runs = []
                for _ in range(T):
                    logits = self.forward(curr_x)
                    probs = torch.softmax(logits, dim=1)
                    mc_runs.append(probs.unsqueeze(0))
                view_outputs.append(torch.cat(mc_runs, dim=0).unsqueeze(0))

        all_probs = torch.cat(view_outputs, dim=0)  # [K, T, B, C]
        per_view_mean = all_probs.mean(dim=1)  # [K, B, C]
        per_view_var = all_probs.var(dim=1, unbiased=False)  # [K, B, C]

        mean_probs = per_view_mean.mean(dim=0)  # [B, C]
        epistemic_var = per_view_var.mean(dim=0)  # E_k[Var_t(p | a_k)]
        aleatoric_tta_var = per_view_mean.var(dim=0, unbiased=False)  # Var_k[E_t(p | a_k)]
        total_var = epistemic_var + aleatoric_tta_var

        if was_training:
            self.train()
        else:
            self.eval()

        return mean_probs, total_var, epistemic_var, aleatoric_tta_var

    def mc_tta_forward(
        self, x: torch.Tensor, T: int = 20, n_augments: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine Monte Carlo Dropout and Test-Time Augmentation.

        Args:
            x: Input image tensor.
            T: Number of stochastic dropout runs per augmented image.
            n_augments: Number of augmented variations.

        Returns:
            A tuple of (mean_probabilities, total_variances) tensors.
        """
        mean_probs, total_var, _, _ = self.mc_tta_forward_decomposed(
            x, T=T, n_augments=n_augments
        )
        return mean_probs, total_var
