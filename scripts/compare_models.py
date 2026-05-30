"""Paired bootstrap statistical comparison script.

Computes paired bootstrap confidence intervals and statistical significance (p-values)
comparing AUC and accuracy between Tier2EfficientNet and Tier2ArkPlus.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.integrity import guard_mock


def calculate_paired_bootstrap(
    targets: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_iterations: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Calculate paired bootstrap statistical differences between two models.

    Args:
        targets: Binary ground truth labels (0 or 1).
        preds_a: Probability predictions from Model A (EfficientNet).
        preds_b: Probability predictions from Model B (Ark+).
        n_iterations: Number of bootstrap iterations.
        alpha: Significance level (default: 0.05 for 95% CI).

    Returns:
        Dictionary compiling statistical metrics, differences, CIs, and p-values.
    """
    n_samples = len(targets)
    auc_a_list = []
    auc_b_list = []
    auc_diff_list = []
    acc_a_list = []
    acc_b_list = []
    acc_diff_list = []

    # Calculate actual/empirical metrics
    auc_a_base = float(roc_auc_score(targets, preds_a))
    auc_b_base = float(roc_auc_score(targets, preds_b))
    auc_diff_base = auc_b_base - auc_a_base

    binary_a_base = (preds_a > 0.5).astype(int)
    binary_b_base = (preds_b > 0.5).astype(int)
    acc_a_base = float(accuracy_score(targets, binary_a_base))
    acc_b_base = float(accuracy_score(targets, binary_b_base))
    acc_diff_base = acc_b_base - acc_a_base

    # Perform bootstrap iterations with replacement
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_targets = targets[indices]
        boot_preds_a = preds_a[indices]
        boot_preds_b = preds_b[indices]

        # Calculate AUC for iteration (fall back to 0.5 if target contains single class)
        try:
            auc_a = roc_auc_score(boot_targets, boot_preds_a)
            auc_b = roc_auc_score(boot_targets, boot_preds_b)
        except ValueError:
            auc_a, auc_b = 0.5, 0.5

        # Calculate accuracy for iteration
        boot_bin_a = (boot_preds_a > 0.5).astype(int)
        boot_bin_b = (boot_preds_b > 0.5).astype(int)
        acc_a = accuracy_score(boot_targets, boot_bin_a)
        acc_b = accuracy_score(boot_targets, boot_bin_b)

        auc_a_list.append(auc_a)
        auc_b_list.append(auc_b)
        auc_diff_list.append(auc_b - auc_a)

        acc_a_list.append(acc_a)
        acc_b_list.append(acc_b)
        acc_diff_list.append(acc_b - acc_a)

    # Convert to arrays for stats
    auc_diffs = np.array(auc_diff_list)
    acc_diffs = np.array(acc_diff_list)

    # Compute percentiles for two-sided confidence intervals
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    auc_ci = (
        float(np.percentile(auc_diffs, lower_pct)),
        float(np.percentile(auc_diffs, upper_pct)),
    )
    acc_ci = (
        float(np.percentile(acc_diffs, lower_pct)),
        float(np.percentile(acc_diffs, upper_pct)),
    )

    # Calculate two-sided bootstrap p-value
    # p-value is twice the proportion of bootstrap differences that cross zero
    auc_p = 2 * min(np.mean(auc_diffs <= 0), np.mean(auc_diffs >= 0))
    acc_p = 2 * min(np.mean(acc_diffs <= 0), np.mean(acc_diffs >= 0))

    # Avoid exact zero p-value reporting in case of non-overlapping distributions
    auc_p = max(auc_p, 1.0 / n_iterations)
    acc_p = max(acc_p, 1.0 / n_iterations)

    return {
        "model_a": {
            "auc": auc_a_base,
            "auc_ci": (
                float(np.percentile(auc_a_list, lower_pct)),
                float(np.percentile(auc_a_list, upper_pct)),
            ),
            "acc": acc_a_base,
            "acc_ci": (
                float(np.percentile(acc_a_list, lower_pct)),
                float(np.percentile(acc_a_list, upper_pct)),
            ),
        },
        "model_b": {
            "auc": auc_b_base,
            "auc_ci": (
                float(np.percentile(auc_b_list, lower_pct)),
                float(np.percentile(auc_b_list, upper_pct)),
            ),
            "acc": acc_b_base,
            "acc_ci": (
                float(np.percentile(acc_b_list, lower_pct)),
                float(np.percentile(acc_b_list, upper_pct)),
            ),
        },
        "difference": {
            "auc_diff": auc_diff_base,
            "auc_ci": auc_ci,
            "auc_p_value": auc_p,
            "acc_diff": acc_diff_base,
            "acc_ci": acc_ci,
            "acc_p_value": acc_p,
        },
    }


def generate_high_fidelity_simulation(n_samples: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate high-fidelity simulation predictions for testing.

    Models realistic Pneumothorax prediction probabilities.
    EfficientNet reaches ~0.78 AUC, while Ark+ reaches ~0.84 AUC.

    Args:
        n_samples: Number of samples to simulate.

    Returns:
        A tuple of (targets, efficientnet_predictions, ark_plus_predictions).
    """
    guard_mock("[compare_models] simulated predictions requested")
    np.random.seed(42)
    # 35% positive prevalence for chest X-ray subset
    targets = (np.random.rand(n_samples) < 0.35).astype(int)

    # EfficientNet prediction probabilities
    preds_a = targets * 0.55 + (1 - targets) * 0.25 + np.random.normal(0, 0.22, n_samples)
    preds_a = np.clip(preds_a, 0.01, 0.99)

    # Ark+ Swin prediction probabilities (statistically superior, lower noise)
    preds_b = targets * 0.65 + (1 - targets) * 0.18 + np.random.normal(0, 0.17, n_samples)
    preds_b = np.clip(preds_b, 0.01, 0.99)

    return targets, preds_a, preds_b


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired Model Bootstrap Comparison")
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1000,
        help="Number of bootstrap iterations.",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/results/tiered_predictions.csv",
        help="Real per-image predictions CSV from scripts/export_predictions.py.",
    )
    parser.add_argument("--model-a", default="tier2_prob", help="Column for model A probabilities.")
    parser.add_argument("--model-b", default="tiered_prob", help="Column for model B probabilities.")
    args = parser.parse_args()

    print(
        f"Starting paired bootstrap comparison (iterations: {args.n_iterations})..."
    )

    # Compare the REAL per-image predictions exported by scripts/export_predictions.py.
    if os.path.exists(args.predictions):
        df = pd.read_csv(args.predictions)
        targets = df["y_true"].to_numpy()
        preds_a = df[args.model_a].to_numpy()
        preds_b = df[args.model_b].to_numpy()
        print(
            f"Loaded REAL predictions from {args.predictions} "
            f"(A={args.model_a}, B={args.model_b}, n={len(targets)})."
        )
    else:
        # No real predictions: fail loudly rather than fabricate (unless explicitly allowed).
        guard_mock(f"[compare_models] {args.predictions} not found")
        print("MOCK MODE — generating placeholder predictions (XRAY_ALLOW_MOCK=1).")
        targets, preds_a, preds_b = generate_high_fidelity_simulation(500)

    # Compute paired bootstrap statistics
    stats = calculate_paired_bootstrap(
        targets, preds_a, preds_b, n_iterations=args.n_iterations
    )

    # Display results nicely
    print("\n" + "=" * 60)
    print("PAIRED BOOTSTRAP MODEL COMPARISON RESULTS")
    print("=" * 60)
    print("MODEL A: EfficientNetB4")
    print(
        f"  AUC:      {stats['model_a']['auc']:.4f} [95% CI: {stats['model_a']['auc_ci'][0]:.4f} - {stats['model_a']['auc_ci'][1]:.4f}]"
    )
    print(
        f"  Accuracy: {stats['model_a']['acc']:.4f} [95% CI: {stats['model_a']['acc_ci'][0]:.4f} - {stats['model_a']['acc_ci'][1]:.4f}]"
    )
    print("-" * 60)
    print("MODEL B: Ark+ (Swin-Base)")
    print(
        f"  AUC:      {stats['model_b']['auc']:.4f} [95% CI: {stats['model_b']['auc_ci'][0]:.4f} - {stats['model_b']['auc_ci'][1]:.4f}]"
    )
    print(
        f"  Accuracy: {stats['model_b']['acc']:.4f} [95% CI: {stats['model_b']['acc_ci'][0]:.4f} - {stats['model_b']['acc_ci'][1]:.4f}]"
    )
    print("-" * 60)
    print("DIFFERENCE (Model B - Model A)")
    print(
        f"  AUC Difference: {stats['difference']['auc_diff']:.4f} [95% CI: {stats['difference']['auc_ci'][0]:.4f} - {stats['difference']['auc_ci'][1]:.4f}]"
    )
    print(f"  AUC p-value:    {stats['difference']['auc_p_value']:.4f}")
    print(
        f"  Acc Difference: {stats['difference']['acc_diff']:.4f} [95% CI: {stats['difference']['acc_ci'][0]:.4f} - {stats['difference']['acc_ci'][1]:.4f}]"
    )
    print(f"  Acc p-value:    {stats['difference']['acc_p_value']:.4f}")
    print("=" * 60)

    # Save to CSV format
    os.makedirs("outputs/results", exist_ok=True)
    csv_path = "outputs/results/model_comparison_bootstrap.csv"
    summary_data = {
        "Metric": ["AUC", "Accuracy"],
        "ModelA_EfficientNetB4": [
            f"{stats['model_a']['auc']:.4f} ({stats['model_a']['auc_ci'][0]:.4f}-{stats['model_a']['auc_ci'][1]:.4f})",
            f"{stats['model_a']['acc']:.4f} ({stats['model_a']['acc_ci'][0]:.4f}-{stats['model_a']['acc_ci'][1]:.4f})",
        ],
        "ModelB_ArkPlus_Swin": [
            f"{stats['model_b']['auc']:.4f} ({stats['model_b']['auc_ci'][0]:.4f}-{stats['model_b']['auc_ci'][1]:.4f})",
            f"{stats['model_b']['acc']:.4f} ({stats['model_b']['acc_ci'][0]:.4f}-{stats['model_b']['acc_ci'][1]:.4f})",
        ],
        "Difference": [
            f"{stats['difference']['auc_diff']:.4f} ({stats['difference']['auc_ci'][0]:.4f}-{stats['difference']['auc_ci'][1]:.4f})",
            f"{stats['difference']['acc_diff']:.4f} ({stats['difference']['acc_ci'][0]:.4f}-{stats['difference']['acc_ci'][1]:.4f})",
        ],
        "p_value": [
            stats["difference"]["auc_p_value"],
            stats["difference"]["acc_p_value"],
        ],
    }
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    print(f"Saved CSV results to: {csv_path}")

    # Save a gorgeous LaTeX table for the thesis document
    latex_path = "outputs/results/model_comparison_table.tex"
    latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Statistical Comparison between EfficientNetB4 and Ark+ Swin-Base under Paired Bootstrap}}
\\label{{tab:model_comparison_bootstrap}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Metric}} & \\textbf{{EfficientNetB4}} & \\textbf{{Ark+ Swin-Base}} & \\textbf{{Difference}} & \\textbf{{p-value}} \\\\ \\hline
AUC & {stats['model_a']['auc']:.4f} ({stats['model_a']['auc_ci'][0]:.4f}, {stats['model_a']['auc_ci'][1]:.4f}) & {stats['model_b']['auc']:.4f} ({stats['model_b']['auc_ci'][0]:.4f}, {stats['model_b']['auc_ci'][1]:.4f}) & {stats['difference']['auc_diff']:.4f} ({stats['difference']['auc_ci'][0]:.4f}, {stats['difference']['auc_ci'][1]:.4f}) & {stats['difference']['auc_p_value']:.4f} \\\\
Accuracy & {stats['model_a']['acc']:.4f} ({stats['model_a']['acc_ci'][0]:.4f}, {stats['model_a']['acc_ci'][1]:.4f}) & {stats['model_b']['acc']:.4f} ({stats['model_b']['acc_ci'][0]:.4f}, {stats['model_b']['acc_ci'][1]:.4f}) & {stats['difference']['acc_diff']:.4f} ({stats['difference']['acc_ci'][0]:.4f}, {stats['difference']['acc_ci'][1]:.4f}) & {stats['difference']['acc_p_value']:.4f} \\\\ \\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"Saved LaTeX thesis table to: {latex_path}")


if __name__ == "__main__":
    main()
