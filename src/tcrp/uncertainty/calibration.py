"""Post-hoc temperature scaling, ECE computation, and reliability diagram plotting.

Temperature scaling (``T_cal``) must be fitted strictly on the Validation Split
before split-conformal quantile estimation runs on the disjoint Calibration Split.
"""

from __future__ import annotations

import os
from itertools import pairwise
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim

log = structlog.get_logger(__name__)


def compute_ece(probs: Any, labels: Any, n_bins: int = 15) -> float:
    """Compute binary Expected Calibration Error (ECE) over equal-width bins."""
    probs_arr = np.asarray(probs)
    labels_arr = np.asarray(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in pairwise(bin_boundaries):
        in_bin = (probs_arr > bin_lower) & (probs_arr <= bin_upper)
        prop_in_bin = float(in_bin.mean())
        if prop_in_bin > 0:
            accuracy_in_bin = float(labels_arr[in_bin].mean())
            avg_confidence_in_bin = float(probs_arr[in_bin].mean())
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


class ModelWithTemperature(nn.Module):  # type: ignore[misc]
    """Post-hoc temperature-scaling wrapper optimizing scalar ``T_cal`` via L-BFGS."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits."""
        return self.temperature_scale(self.model(input_tensor))

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide raw logits by the learned positive temperature scalar."""
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader: Any, device: torch.device) -> ModelWithTemperature:
        """Fit ``T_cal`` on the Validation Split via NLL minimization with L-BFGS."""
        self.to(device)
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)

        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in valid_loader:
                input_val = batch[0].to(device)
                label = batch[1]
                logits_list.append(self.model(input_val))
                labels_list.append(label)

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        before_nll = nll_criterion(logits, labels).item()
        before_ece = compute_ece(
            torch.softmax(logits, dim=1)[:, 1].cpu().numpy(), labels.cpu().numpy()
        )
        log.info("Before temperature scaling", nll=round(before_nll, 4), ece=round(before_ece, 4))

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        after_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_ece = compute_ece(
            torch.softmax(self.temperature_scale(logits), dim=1)[:, 1].detach().cpu().numpy(),
            labels.cpu().numpy(),
        )
        log.info(
            "After temperature scaling",
            temperature=round(float(self.temperature.item()), 4),
            nll=round(after_nll, 4),
            ece=round(after_ece, 4),
        )
        return self


def calibrate_temperature(
    model: nn.Module,
    validation_loader: Any,
    device: torch.device | None = None,
) -> tuple[ModelWithTemperature, float]:
    """Fit post-hoc temperature scaling on ``validation_loader`` and return ``(wrapper, T_cal)``."""
    active_device = device or torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    calibrator = ModelWithTemperature(model)
    calibrator.set_temperature(validation_loader, active_device)
    return calibrator, float(calibrator.temperature.item())


def plot_reliability_diagram(
    probs: Any, labels: Any, n_bins: int = 15, save_path: str | None = None
) -> str | None:
    """Plot a calibration reliability diagram and optionally save it to ``save_path``."""
    probs_arr = np.asarray(probs)
    labels_arr = np.asarray(labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    accuracies: list[float] = []
    confidences: list[float] = []
    for bin_lower, bin_upper in pairwise(bin_boundaries):
        in_bin = (probs_arr > bin_lower) & (probs_arr <= bin_upper)
        if in_bin.any():
            accuracies.append(float(labels_arr[in_bin].mean()))
            confidences.append(float(probs_arr[in_bin].mean()))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration", color="gray")
    plt.plot(confidences, accuracies, marker="o", label="Model Calibration", color="blue")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)

    abs_saved: str | None = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        abs_saved = os.path.abspath(save_path)
    plt.close()
    return abs_saved
