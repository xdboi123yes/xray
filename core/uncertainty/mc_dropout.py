"""Monte Carlo (MC) Dropout Uncertainty Quantification Module.

Computes predictive entropy and mutual information to decompose epistemic and aleatoric
uncertainties, and analyzes their correlation with classification correctness.
"""

from __future__ import annotations

import os
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import structlog

log = structlog.get_logger(__name__)

def compute_predictive_entropy(mean_probs: Any) -> float:
    """Computes predictive entropy from mean softmax probabilities.

    H[y|x] = -sum(p * log(p))

    Args:
        mean_probs: Array of class probabilities.

    Returns:
        The predictive entropy score.
    """
    probs = np.clip(np.asarray(mean_probs), 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))

def compute_mutual_information(mc_probs: Any) -> tuple[float, float, float]:
    """Computes mutual information (epistemic uncertainty) from MC Dropout runs.

    MI = H[y|x] - E[H[y|x, w]]

    Args:
        mc_probs: Array of shape [T, num_classes] containing probabilities from T passes.

    Returns:
        A tuple of (predictive_entropy, expected_entropy, mutual_information).
    """
    mc_probs = np.asarray(mc_probs)
    mean_probs = mc_probs.mean(axis=0)

    predictive_entropy = compute_predictive_entropy(mean_probs)

    expected_entropy = 0.0
    for t in range(mc_probs.shape[0]):
        expected_entropy += compute_predictive_entropy(mc_probs[t])
    expected_entropy /= mc_probs.shape[0]

    mutual_information = predictive_entropy - expected_entropy

    return predictive_entropy, expected_entropy, mutual_information

def analyze_uncertainty_batch(model: Any, dataloader: Any, device: torch.device, T: int = 20) -> dict[str, Any]:
    """Runs MC Dropout in a batch over a dataloader to collect uncertainty logs.

    Args:
        model: Specialist classifier supporting mc_forward() operations.
        dataloader: Target dataset evaluation loader.
        device: Computing accelerator device.
        T: Number of MC dropout iterations.

    Returns:
        A dictionary containing arrays of predictions, labels, variances, and entropies.
    """
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

    model.train()  # Enable Dropout layers during inference

    for images, labels, _ in tqdm(dataloader, desc="Uncertainty Analysis"):
        images = images.to(device)

        mc_outputs = []
        with torch.no_grad():
            for _ in range(T):
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                mc_outputs.append(probs.cpu().numpy())

        mc_outputs_arr = np.array(mc_outputs)  # [T, batch, num_classes]

        for i in range(images.size(0)):
            mc_probs_i = mc_outputs_arr[:, i, :]  # [T, num_classes]
            mean_probs = mc_probs_i.mean(axis=0)
            variance = float(mc_probs_i.var(axis=0).mean())

            pred_class = int(np.argmax(mean_probs))
            confidence = float(mean_probs[pred_class])

            pred_ent, exp_ent, mi = compute_mutual_information(mc_probs_i)

            label = int(labels[i].item())
            correct = int(pred_class == label)

            results["labels"].append(label)
            results["predictions"].append(pred_class)
            results["confidences"].append(confidence)
            results["variances"].append(variance)
            results["predictive_entropies"].append(pred_ent)
            results["expected_entropies"].append(exp_ent)
            results["mutual_informations"].append(mi)
            results["correct"].append(correct)

    model.eval()

    final_results = {}
    for key, val in results.items():
        final_results[key] = np.array(val)

    return final_results

def analyze_uncertainty_vs_correctness(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Compares uncertainty levels between correct and incorrect diagnostic decisions.

    Args:
        results: Results dict from analyze_uncertainty_batch.

    Returns:
        Summary statistics dict.
    """
    correct_mask = results["correct"] == 1
    incorrect_mask = results["correct"] == 0

    summary = {}
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
    """Plots uncertainty distributions comparing correct vs incorrect predictions.

    Args:
        results: Batch statistics dict.
        save_dir: Optional directory path to save resulting figure.
    """
    correct_mask = results["correct"] == 1
    incorrect_mask = results["correct"] == 0

    metrics = {
        "MC Dropout Variance": "variances",
        "Predictive Entropy": "predictive_entropies",
        "Mutual Information (Epistemic)": "mutual_informations",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (title, key) in zip(axes, metrics.items(), strict=False):
        correct_vals = results[key][correct_mask]
        incorrect_vals = results[key][incorrect_mask]

        ax.hist(
            correct_vals, bins=30, alpha=0.6, label="Correct", color="#2ecc71", density=True
        )
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
        log.info(f"Uncertainty distribution plot saved to {save_dir}/uncertainty_distributions.png")

    plt.close()

def print_uncertainty_summary(results: dict[str, Any]) -> None:
    """Prints a formatted terminal summary report of uncertainty analysis.

    Args:
        results: Stats dict.
    """
    summary = analyze_uncertainty_vs_correctness(results)

    total = len(results["correct"])
    n_correct = int(results["correct"].sum())
    accuracy = n_correct / total if total > 0 else 0.0

    log.info("\n" + "=" * 60)
    log.info("UNCERTAINTY ANALYSIS SUMMARY")
    log.info("=" * 60)
    log.info(f"Total samples: {total}")
    log.info(f"Accuracy: {accuracy:.4f} ({n_correct}/{total})")
    log.info()

    for metric_name, metric_key in [
        ("MC Dropout Variance", "variances"),
        ("Predictive Entropy", "predictive_entropies"),
        ("Mutual Information", "mutual_informations"),
    ]:
        s = summary[metric_key]
        log.info(f"--- {metric_name} ---")
        log.info(f"  Correct predictions:   mean={s['correct_mean']:.4f} ± {s['correct_std']:.4f}")
        log.info(f"  Incorrect predictions: mean={s['incorrect_mean']:.4f} ± {s['incorrect_std']:.4f}")

        if s["correct_mean"] > 0:
            ratio = s["incorrect_mean"] / s["correct_mean"]
            log.info(f"  Ratio (incorrect/correct): {ratio:.2f}x")
        log.info()
