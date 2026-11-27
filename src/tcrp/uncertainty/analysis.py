"""Monte Carlo Dropout entropy/mutual-information analysis and Test-Time Augmentation."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import structlog
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

log = structlog.get_logger(__name__)


def compute_predictive_entropy(mean_probs: Any) -> float:
    """Compute predictive entropy ``H[y|x] = -sum(p * log(p))``."""
    probs = np.clip(np.asarray(mean_probs), 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_mutual_information(mc_probs: Any) -> tuple[float, float, float]:
    """Compute ``(predictive_entropy, expected_entropy, mutual_information)`` from ``[T, C]``."""
    mc_arr = np.asarray(mc_probs)
    mean_probs = mc_arr.mean(axis=0)
    predictive_entropy = compute_predictive_entropy(mean_probs)
    expected_entropy = float(
        np.mean([compute_predictive_entropy(mc_arr[t]) for t in range(mc_arr.shape[0])])
    )
    return predictive_entropy, expected_entropy, predictive_entropy - expected_entropy


def _enable_dropout_only(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def analyze_uncertainty_batch(
    model: Any, dataloader: Any, device: torch.device, T: int = 20
) -> dict[str, np.ndarray[Any, Any]]:
    """Run BatchNorm-safe MC Dropout over ``dataloader`` and collect uncertainty arrays."""
    results: dict[str, list[Any]] = {
        "labels": [],
        "predictions": [],
        "confidences": [],
        "variances": [],
        "predictive_entropies": [],
        "expected_entropies": [],
        "mutual_informations": [],
        "correct": [],
    }

    _enable_dropout_only(model)
    for batch in tqdm(dataloader, desc="Uncertainty Analysis", disable=True):
        images = batch[0].to(device)
        labels = batch[1]

        mc_outputs = []
        with torch.no_grad():
            for _ in range(T):
                probs = torch.softmax(model(images), dim=1)
                mc_outputs.append(probs.cpu().numpy())

        mc_outputs_arr = np.array(mc_outputs)  # [T, B, C]
        for i in range(images.size(0)):
            mc_probs_i = mc_outputs_arr[:, i, :]
            mean_probs = mc_probs_i.mean(axis=0)
            variance = float(mc_probs_i.var(axis=0).mean())
            pred_class = int(np.argmax(mean_probs))
            confidence = float(mean_probs[pred_class])
            pred_ent, exp_ent, mi = compute_mutual_information(mc_probs_i)

            label = int(labels[i].item())
            results["labels"].append(label)
            results["predictions"].append(pred_class)
            results["confidences"].append(confidence)
            results["variances"].append(variance)
            results["predictive_entropies"].append(pred_ent)
            results["expected_entropies"].append(exp_ent)
            results["mutual_informations"].append(mi)
            results["correct"].append(int(pred_class == label))

    model.eval()
    return {key: np.array(val) for key, val in results.items()}


def analyze_uncertainty_vs_correctness(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Compare uncertainty metric means and stds across correct vs. incorrect predictions."""
    correct_mask = results["correct"] == 1
    incorrect_mask = results["correct"] == 0

    summary: dict[str, dict[str, float]] = {}
    for metric in ["variances", "predictive_entropies", "mutual_informations"]:
        correct_vals = results[metric][correct_mask]
        incorrect_vals = results[metric][incorrect_mask]
        summary[metric] = {
            "correct_mean": float(correct_vals.mean() if len(correct_vals) > 0 else 0.0),
            "correct_std": float(correct_vals.std() if len(correct_vals) > 0 else 0.0),
            "incorrect_mean": float(incorrect_vals.mean() if len(incorrect_vals) > 0 else 0.0),
            "incorrect_std": float(incorrect_vals.std() if len(incorrect_vals) > 0 else 0.0),
        }
    return summary


def plot_uncertainty_distribution(results: dict[str, Any], save_dir: str | None = None) -> None:
    """Plot histograms comparing correct vs. incorrect prediction uncertainty."""
    correct_mask = results["correct"] == 1
    incorrect_mask = results["correct"] == 0

    metrics = {
        "MC Dropout Variance": "variances",
        "Predictive Entropy": "predictive_entropies",
        "Mutual Information (Epistemic)": "mutual_informations",
    }

    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, key) in zip(axes, metrics.items(), strict=False):
        correct_vals = results[key][correct_mask]
        incorrect_vals = results[key][incorrect_mask]
        ax.hist(correct_vals, bins=30, alpha=0.6, label="Correct", color="#2ecc71", density=True)
        if len(incorrect_vals) > 0:
            ax.hist(
                incorrect_vals,
                bins=30,
                alpha=0.6,
                label="Incorrect",
                color="#e74c3c",
                density=True,
            )
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.set_title(f"{title} Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, "uncertainty_distributions.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def print_uncertainty_summary(results: dict[str, Any]) -> None:
    """Log a formatted summary comparing uncertainty across correct and incorrect predictions."""
    summary = analyze_uncertainty_vs_correctness(results)
    total = len(results["correct"])
    n_correct = int(results["correct"].sum())
    accuracy = n_correct / total if total > 0 else 0.0
    log.info("Uncertainty analysis summary", total=total, correct=n_correct, accuracy=accuracy)
    for metric_name, metric_key in [
        ("MC Dropout Variance", "variances"),
        ("Predictive Entropy", "predictive_entropies"),
        ("Mutual Information", "mutual_informations"),
    ]:
        s = summary[metric_key]
        ratio = s["incorrect_mean"] / s["correct_mean"] if s["correct_mean"] > 0 else 0.0
        log.info(metric_name, ratio=round(ratio, 2), **s)


class TestTimeAugmenter:
    """Generates clinically safe TTA views and aggregates softmax predictions."""

    def __init__(self, n_augments: int = 10) -> None:
        self.n_augments = n_augments
        self.transforms = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )

    def generate_augments(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return ``n_augments`` views starting with the unmodified input ``x``."""
        return [x, *(self.transforms(x) for _ in range(self.n_augments - 1))]

    def predict_with_tta(
        self, model: torch.nn.Module, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``model`` over TTA views and return ``(mean_probs, mean_variance)``."""
        model.eval()
        aug_inputs = self.generate_augments(x)
        with torch.no_grad():
            probs_stack = torch.stack([torch.softmax(model(aug_x), dim=1) for aug_x in aug_inputs])
        return probs_stack.mean(dim=0), probs_stack.var(dim=0).mean(dim=-1)
