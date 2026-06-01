"""Comparative statistical testing CLI script for chest X-ray classifiers.

Loads REAL per-image predictions exported by ``scripts/export_predictions.py``
(``outputs/results/tiered_predictions.csv``) and performs:
- DeLong tests for AUC differences
- McNemar tests for paired classification accuracy
- Paired permutation tests for distribution-free comparisons
- Bootstrap confidence intervals for all major clinical metrics

It NEVER fabricates results. If the real predictions file is missing the script
refuses to run unless ``XRAY_ALLOW_MOCK=1`` is set (tests / offline dry-runs),
in which case a clearly-labelled simulation is used instead.

Generates thesis-ready LaTeX tables and CSV reports.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.evaluation.stats import (
    bootstrap_ci,
    delong_test,
    mcnemar_test,
    permutation_test,
)
from core.integrity import guard_mock


def generate_tiered_simulation(n_samples: int = 500) -> dict[str, np.ndarray]:
    """Generate clearly-labelled SIMULATED predictions (mock path only).

    This is used solely when no real predictions exist AND the caller opted in
    via ``XRAY_ALLOW_MOCK=1``. It must never feed thesis numbers.

    Args:
        n_samples: Number of evaluation samples.

    Returns:
        A dictionary containing y_true, t1_mobilenet, t2_effnet, t2_arkplus, and
        tiered_system predictions.
    """
    np.random.seed(42)
    y_true = (np.random.rand(n_samples) < 0.35).astype(int)

    # Tier 1 (MobileNetV2): moderate performance, higher noise
    t1 = y_true * 0.42 + (1 - y_true) * 0.32 + np.random.normal(0, 0.26, n_samples)
    t1 = np.clip(t1, 0.01, 0.99)

    # Tier 2 (EfficientNetB4): strong performance, standard baseline
    t2_eff = y_true * 0.58 + (1 - y_true) * 0.22 + np.random.normal(0, 0.20, n_samples)
    t2_eff = np.clip(t2_eff, 0.01, 0.99)

    # Tier 2 (Ark+ Swin-Base): SOTA, outstanding diagnostic accuracy
    t2_ark = y_true * 0.68 + (1 - y_true) * 0.16 + np.random.normal(0, 0.14, n_samples)
    t2_ark = np.clip(t2_ark, 0.01, 0.99)

    # Tiered System (Routed prediction): accept confident Tier 1, else escalate.
    tiered = np.where((t1 < 0.15) | (t1 > 0.85), t1, t2_ark)

    return {
        "y_true": y_true,
        "t1_mobilenet": t1,
        "t2_effnet": t2_eff,
        "t2_arkplus": t2_ark,
        "tiered_system": tiered,
    }


def load_real_predictions(
    ark_csv: str, eff_csv: str | None
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """Load REAL per-image predictions written by scripts/export_predictions.py.

    The Ark+ export (``tiered_predictions.csv``) carries the tiered system and its
    Tier 1 / Tier 2 component probabilities. The optional EfficientNet export
    (``tiered_predictions_efficientnet.csv``) is merged by ``image_id`` so the
    EfficientNet-vs-Ark+ comparison uses genuinely paired predictions.

    Args:
        ark_csv: Path to the Ark+ per-image predictions CSV (required).
        eff_csv: Path to the EfficientNet per-image predictions CSV (optional).

    Returns:
        Dict with y_true, t1_mobilenet, t2_arkplus, tiered_system arrays, plus a
        nested ``cmp_eff_ark`` dict (y_true/eff/ark) when the EfficientNet CSV
        is present and shares images with the Ark+ export.

    Raises:
        FileNotFoundError: If the Ark+ predictions CSV does not exist.
    """
    if not os.path.exists(ark_csv):
        raise FileNotFoundError(ark_csv)

    ark = pd.read_csv(ark_csv)
    data: dict[str, np.ndarray | dict[str, np.ndarray]] = {
        "y_true": ark["y_true"].to_numpy().astype(int),
        "t1_mobilenet": ark["tier1_prob"].to_numpy(dtype=float),
        "t2_arkplus": ark["tier2_prob"].to_numpy(dtype=float),
        "tiered_system": ark["tiered_prob"].to_numpy(dtype=float),
    }

    if eff_csv and os.path.exists(eff_csv):
        eff = pd.read_csv(eff_csv)[["image_id", "tier2_prob"]].rename(
            columns={"tier2_prob": "eff"}
        )
        a = ark[["image_id", "y_true", "tier2_prob"]].rename(columns={"tier2_prob": "ark"})
        merged = a.merge(eff, on="image_id", how="inner")
        if len(merged) > 0:
            data["cmp_eff_ark"] = {
                "y_true": merged["y_true"].to_numpy().astype(int),
                "eff": merged["eff"].to_numpy(dtype=float),
                "ark": merged["ark"].to_numpy(dtype=float),
            }
    return data


def format_ci(val: float, ci: tuple[float, float]) -> str:
    """Formats value and confidence interval for printing."""
    return f"{val:.4f} [95% CI: {ci[0]:.4f} - {ci[1]:.4f}]"


def run_comparison(
    y_true: np.ndarray,
    y_probs1: np.ndarray,
    y_probs2: np.ndarray,
    model1_name: str,
    model2_name: str,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict:
    """Runs a complete suite of comparative statistical tests between two models."""
    print(f"\nComparing {model1_name} vs {model2_name}...")

    y_true = np.asarray(y_true)
    y_probs1 = np.asarray(y_probs1, dtype=float)
    y_probs2 = np.asarray(y_probs2, dtype=float)

    # 1. Bootstrap CI for key metrics
    metrics = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    bootstrap_results = {}
    for metric in metrics:
        bootstrap_results[metric] = bootstrap_ci(
            y_true,
            y_probs1,
            y_probs2,
            metric=metric,
            n_iterations=n_iterations,
            seed=seed,
        )

    # 2. DeLong test for the AUC difference (correlated ROC curves).
    # delong_test returns the two-tailed p-value as a float.
    delong_p = delong_test(y_true, y_probs1, y_probs2)

    # 3. McNemar's Test for accuracy difference
    mcnemar = mcnemar_test(y_true, y_probs1, y_probs2)

    # 4. Permutation Test for AUC and F1 difference
    perm_auc = permutation_test(
        y_true,
        y_probs1,
        y_probs2,
        metric="auc",
        n_permutations=n_iterations,
        seed=seed,
    )
    perm_f1 = permutation_test(
        y_true,
        y_probs1,
        y_probs2,
        metric="f1",
        n_permutations=n_iterations,
        seed=seed,
    )

    # Display clean textual results
    print("-" * 70)
    print(f"{'Metric':<15} | {model1_name:<20} | {model2_name:<20} | {'Delta':<10}")
    print("-" * 70)
    for metric in metrics:
        res = bootstrap_results[metric]
        print(
            f"{metric.upper():<15} | "
            f"{res['model1_metric']:.4f} ({res['model1_ci'][0]:.3f}-{res['model1_ci'][1]:.3f}) | "
            f"{res['model2_metric']:.4f} ({res['model2_ci'][0]:.3f}-{res['model2_ci'][1]:.3f}) | "
            f"{res['delta']:.4f} ({res['delta_ci'][0]:.3f}-{res['delta_ci'][1]:.3f})"
        )
    print("-" * 70)
    print(f"DeLong AUC p-value:                {delong_p:.6f}")
    print(f"Bootstrap AUC difference p-value:  {bootstrap_results['auc']['p_value_bootstrap']:.6f}")
    print(f"Permutation AUC difference p-value:{perm_auc['p_value']:.6f}")
    print(f"Permutation F1 difference p-value: {perm_f1['p_value']:.6f}")
    print(f"McNemar paired accuracy p-value:   {mcnemar['p_value']:.6f}")
    print("Contingency Table (Matched-Pairs Correctness):")
    print(
        f"  Both Correct: {mcnemar['contingency_table']['a']} | "
        f"Only {model1_name} Correct: {mcnemar['contingency_table']['b']}"
    )
    print(
        f"  Only {model2_name} Correct: {mcnemar['contingency_table']['c']} | "
        f"Both Incorrect: {mcnemar['contingency_table']['d']}"
    )
    print("=" * 70)

    # Compile result package
    return {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "bootstrap": bootstrap_results,
        "delong_p": delong_p,
        "mcnemar": mcnemar,
        "permutation_auc": perm_auc,
        "permutation_f1": perm_f1,
    }


def _build_comparisons(data: dict, n_iterations: int, seed: int) -> list[dict]:
    """Construct the comparison list defensively from whatever real data exists."""
    comparisons: list[dict] = []

    # EfficientNet vs Ark+ (paired, real subset) or simulated arrays.
    if isinstance(data.get("cmp_eff_ark"), dict):
        c = data["cmp_eff_ark"]
        comparisons.append(
            run_comparison(
                c["y_true"], c["eff"], c["ark"],
                "EfficientNetB4", "Ark+ Swin",
                n_iterations=n_iterations, seed=seed,
            )
        )
    elif "t2_effnet" in data:
        comparisons.append(
            run_comparison(
                data["y_true"], data["t2_effnet"], data["t2_arkplus"],
                "EfficientNetB4", "Ark+ Swin",
                n_iterations=n_iterations, seed=seed,
            )
        )
    else:
        print(
            "Note: EfficientNet predictions not found "
            "(outputs/results/tiered_predictions_efficientnet.csv); "
            "skipping the EfficientNet-vs-Ark+ comparison."
        )

    y_true = data["y_true"]
    comparisons.append(
        run_comparison(
            y_true, data["t1_mobilenet"], data["tiered_system"],
            "MobileNetV2 (T1)", "Tiered System",
            n_iterations=n_iterations, seed=seed,
        )
    )
    comparisons.append(
        run_comparison(
            y_true, data["t2_arkplus"], data["tiered_system"],
            "Ark+ Swin (T2)", "Tiered System",
            n_iterations=n_iterations, seed=seed,
        )
    )
    return comparisons


def _write_csv(comparisons: list[dict], csv_path: str) -> None:
    """Write the structured per-metric comparison rows to CSV."""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    rows = []
    for comp in comparisons:
        m1, m2 = comp["model1_name"], comp["model2_name"]
        for metric in ["auc", "accuracy", "sensitivity", "specificity", "f1"]:
            boot = comp["bootstrap"][metric]
            rows.append(
                {
                    "Comparison": f"{m1} vs {m2}",
                    "Metric": metric.upper(),
                    "Model1_Score": boot["model1_metric"],
                    "Model1_CI_Lower": boot["model1_ci"][0],
                    "Model1_CI_Upper": boot["model1_ci"][1],
                    "Model2_Score": boot["model2_metric"],
                    "Model2_CI_Lower": boot["model2_ci"][0],
                    "Model2_CI_Upper": boot["model2_ci"][1],
                    "Delta": boot["delta"],
                    "Delta_CI_Lower": boot["delta_ci"][0],
                    "Delta_CI_Upper": boot["delta_ci"][1],
                    "p_value_bootstrap": boot["p_value_bootstrap"],
                    "p_value_delong": comp["delong_p"] if metric == "auc" else float("nan"),
                    "p_value_mcnemar": comp["mcnemar"]["p_value"]
                    if metric == "accuracy"
                    else float("nan"),
                    "p_value_permutation": comp["permutation_auc"]["p_value"]
                    if metric == "auc"
                    else float("nan"),
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved structured CSV metrics to: {csv_path}")


def _write_latex(comparisons: list[dict], latex_path: str) -> None:
    """Write a thesis-ready LaTeX comparison table from real results."""

    def latex_row(comp: dict, key: str, label: str) -> str:
        boot = comp["bootstrap"][key]
        p_str = f"{comp['delong_p']:.4f}" if key == "auc" else "N/A"
        return (
            f"{label} & "
            f"{boot['model1_metric']:.3f} ({boot['model1_ci'][0]:.3f}, {boot['model1_ci'][1]:.3f}) & "
            f"{boot['model2_metric']:.3f} ({boot['model2_ci'][0]:.3f}, {boot['model2_ci'][1]:.3f}) & "
            f"{boot['delta']:.3f} ({boot['delta_ci'][0]:.3f}, {boot['delta_ci'][1]:.3f}) & "
            f"{p_str} \\\\"
        )

    metric_labels = [
        ("auc", "AUC-ROC"),
        ("accuracy", "Accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("f1", "F1-Score"),
    ]
    sections = []
    for comp in comparisons:
        header = (
            f"\\multicolumn{{5}}{{l}}{{\\textbf{{{comp['model1_name']} vs "
            f"{comp['model2_name']}}}}} \\\\"
        )
        body = "\n".join(latex_row(comp, k, lbl) for k, lbl in metric_labels)
        sections.append(header + "\n" + body + " \\\\ \\hline")
    body_tex = "\n".join(sections)

    latex_table = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\caption{Statistical Comparison of Diagnosis Classifiers and Tiered "
        "Architectures (Bootstrap 95\\% CI and DeLong test on real test-set predictions)}\n"
        "\\label{tab:statistical_comparison_detailed}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\hline\n"
        "\\textbf{Metric} & \\textbf{Model 1} & \\textbf{Model 2} & "
        "\\textbf{Difference (Model 2 - 1)} & \\textbf{DeLong p-value} \\\\ \\hline\n"
        f"{body_tex}\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"Saved LaTeX thesis table to: {latex_path}")


def main() -> None:
    """CLI entrypoint for the comparative statistical suite."""
    parser = argparse.ArgumentParser(
        description=(
            "Comparative Statistical Diagnostics CLI. / "
            "Karsilastirmali Istatistiksel Teshis CLI Betigi."
        )
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/results/tiered_predictions.csv",
        help="Real Ark+ per-image predictions CSV. / Gercek Ark+ tahmin CSV yolu.",
    )
    parser.add_argument(
        "--eff-predictions",
        type=str,
        default="outputs/results/tiered_predictions_efficientnet.csv",
        help="Optional EfficientNet predictions CSV. / Istege bagli EfficientNet tahmin CSV.",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1000,
        help="Number of bootstrap iterations and permutations. / Yineleme sayisi.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. / Rastgelelik cekirdegi.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results/statistical_comparison.csv",
        help="Output CSV path. / Cikti CSV dosyasinin kaydedilecegi yol.",
    )
    args = parser.parse_args()

    print(
        f"Initializing comparative statistical suite "
        f"(iterations: {args.n_iterations}, seed: {args.seed})..."
    )

    # Real predictions are the only honest source. Fall back to a simulation ONLY
    # when explicitly allowed via XRAY_ALLOW_MOCK=1 (otherwise guard_mock raises).
    try:
        data = load_real_predictions(args.predictions, args.eff_predictions)
        print(f"Loaded REAL predictions from {args.predictions} (n={len(data['y_true'])}).")
    except FileNotFoundError:
        guard_mock(
            f"statistical_tests: real predictions missing at {args.predictions}; "
            "run scripts/export_predictions.py first"
        )
        print("XRAY_ALLOW_MOCK=1 -> using SIMULATED predictions (NOT valid for the thesis).")
        test_csv = "data/processed/test.csv"
        n = len(pd.read_csv(test_csv)) if os.path.exists(test_csv) else 500
        data = generate_tiered_simulation(n)

    comparisons = _build_comparisons(data, n_iterations=args.n_iterations, seed=args.seed)

    _write_csv(comparisons, args.output)
    _write_latex(comparisons, "outputs/results/statistical_comparison_table.tex")


if __name__ == "__main__":
    main()
