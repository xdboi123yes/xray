"""Calibration and Temperature Scaling Utilities.

Implements Expected Calibration Error (ECE), parametric temperature scaling optimization
via L-BFGS, and reliability diagram plots.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import structlog

log = structlog.get_logger(__name__)


def compute_ece(probs: Any, labels: Any, n_bins: int = 10) -> float:
    """Computes the Expected Calibration Error (ECE) for binary classification.

    Args:
        probs: Array-like of probabilities for the positive class (0 to 1).
        labels: Array-like of ground truth binary labels (0 or 1).
        n_bins: Number of confidence bins to divide ECE calculation.

    Returns:
        The expected calibration error score.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = float(in_bin.mean())
        if prop_in_bin > 0:
            accuracy_in_bin = float(labels[in_bin].mean())
            avg_confidence_in_bin = float(probs[in_bin].mean())
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


class ModelWithTemperature(nn.Module):
    """Wrapper decorator for temperature-scaling calibrators."""

    def __init__(self, model: nn.Module) -> None:
        """Initializes the model wrapper.

        Args:
            model: Underline PyTorch module returning raw logits.
        """
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass forwarding temperature scaled logits."""
        logits = self.model(input_tensor)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Divides raw logits by learnable temperature scalar safely."""
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader: Any, device: torch.device) -> ModelWithTemperature:
        """Tunes the model temperature using a validation dataset via L-BFGS.

        Args:
            valid_loader: Validation split DataLoader.
            device: Active hardware device.
        """
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input_val, label, _ in valid_loader:
                input_val = input_val.to(device)
                logits = self.model(input_val)
                logits_list.append(logits)
                labels_list.append(label)

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        before_temperature_nll = nll_criterion(logits, labels).item()
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        labels_np = labels.cpu().numpy()
        before_temperature_ece = compute_ece(probs, labels_np)
        log.info(
            f"Before Temp Scaling - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}"
        )

        # Optimize temperature w.r.t. negative log likelihood using L-BFGS
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        probs_after = (
            torch.softmax(self.temperature_scale(logits), dim=1)[:, 1].detach().cpu().numpy()
        )
        after_temperature_ece = compute_ece(probs_after, labels_np)
        log.info(f"Optimal Temperature (T): {self.temperature.item():.3f}")
        log.info(
            f"After Temp Scaling - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}"
        )

        return self


def plot_reliability_diagram(
    probs: Any, labels: Any, n_bins: int = 10, save_path: str | None = None
) -> None:
    """Plots a calibration reliability diagram.

    Args:
        probs: Positive predictions probabilities list.
        labels: Ground truth binary labels.
        n_bins: Bin discretization.
        save_path: Optional output path to save diagram figure.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = []
    confidences = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        if in_bin.any():
            accuracies.append(float(labels[in_bin].mean()))
            confidences.append(float(probs[in_bin].mean()))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration", color="gray")
    plt.plot(confidences, accuracies, marker="o", label="Model Calibration", color="blue")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.close()
