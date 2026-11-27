"""Deep BaseClassifier module centralizing MC Dropout, TTA, and Uncertainty Decomposition.

Subclasses only define architecture initialization and ``forward(x)``. All
classifiers inherit identical BatchNorm-safe Monte Carlo Dropout, Test-Time
Augmentation, and Law-of-Total-Variance decomposition:

    Var_total(p) = E_k[Var_t(p | a_k)] + Var_k[E_t(p | a_k)]
                 = epistemic_var        + aleatoric_tta_var
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torchvision.transforms as t_aug


def _build_tta_transforms() -> t_aug.Compose:
    """Return clinically safe spatial perturbations for Test-Time Augmentation."""
    return t_aug.Compose(
        [
            t_aug.RandomHorizontalFlip(p=0.5),
            t_aug.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
    )


class BaseClassifier(nn.Module, ABC):  # type: ignore[misc]
    """Base interface and shared uncertainty engine for Tier 1 and Tier 2 classifiers."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape ``[batch_size, num_classes]``."""

    def get_confidence(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(confidences, predicted_class_indices)`` from raw logits."""
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        return confidences, predictions

    def _enable_dropout_only(self) -> None:
        """Keep normalization layers in eval mode while enabling Dropout layers."""
        self.eval()
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    def mc_forward(self, x: torch.Tensor, T: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``T`` stochastic Dropout passes while preserving BatchNorm eval mode."""
        was_training = self.training
        self._enable_dropout_only()

        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(T):
                probs = torch.softmax(self.forward(x), dim=1)
                outputs.append(probs.unsqueeze(0))

        outputs_tensor = torch.cat(outputs, dim=0)
        mean_probs = outputs_tensor.mean(dim=0)
        variance = outputs_tensor.var(dim=0, unbiased=False)

        self.train() if was_training else self.eval()
        return mean_probs, variance

    def tta_forward(self, x: torch.Tensor, n_augments: int = 10) -> torch.Tensor:
        """Average softmax probabilities over ``n_augments`` views (view 0 unperturbed)."""
        tta_transforms = _build_tta_transforms()
        was_training = self.training
        self.eval()

        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(n_augments):
                curr_x = x if i == 0 else tta_transforms(x)
                probs = torch.softmax(self.forward(curr_x), dim=1)
                outputs.append(probs.unsqueeze(0))

        mean_probs = torch.cat(outputs, dim=0).mean(dim=0)
        if was_training:
            self.train()
        return mean_probs

    def mc_tta_forward_decomposed(
        self, x: torch.Tensor, T: int = 20, n_augments: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine MC Dropout and TTA with Law-of-Total-Variance Uncertainty Decomposition.

        Returns:
            ``(mean_probs, total_var, epistemic_var, aleatoric_tta_var)`` each shaped
            ``[batch_size, num_classes]``, satisfying ``total_var = epistemic_var + aleatoric_tta_var``.
        """
        tta_transforms = _build_tta_transforms()
        was_training = self.training
        self._enable_dropout_only()

        view_outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(n_augments):
                curr_x = x if i == 0 else tta_transforms(x)
                mc_runs = [
                    torch.softmax(self.forward(curr_x), dim=1).unsqueeze(0) for _ in range(T)
                ]
                view_outputs.append(torch.cat(mc_runs, dim=0).unsqueeze(0))

        all_probs = torch.cat(view_outputs, dim=0)  # [K, T, B, C]
        per_view_mean = all_probs.mean(dim=1)  # [K, B, C]
        per_view_var = all_probs.var(dim=1, unbiased=False)  # [K, B, C]

        mean_probs = per_view_mean.mean(dim=0)
        epistemic_var = per_view_var.mean(dim=0)
        aleatoric_tta_var = per_view_mean.var(dim=0, unbiased=False)
        total_var = epistemic_var + aleatoric_tta_var

        self.train() if was_training else self.eval()
        return mean_probs, total_var, epistemic_var, aleatoric_tta_var

    def mc_tta_forward(
        self, x: torch.Tensor, T: int = 20, n_augments: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean_probs, total_var)`` from combined MC Dropout + TTA."""
        mean_probs, total_var, _, _ = self.mc_tta_forward_decomposed(
            x, T=T, n_augments=n_augments
        )
        return mean_probs, total_var
