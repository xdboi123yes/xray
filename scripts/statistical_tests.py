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
    cohens_h,
    delong_test,
    holm_bonferroni,
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
        A dictionary containing y_true, patient_ids, t1_mobilenet, t2_effnet,
        t2_arkplus, and tiered_system predictions.
    """
    np.random.seed(42)
    y_true = (np.random.rand(n_samples) < 0.35).astype(int)
    patient_ids = np.array([f"{i // 2:08d}" for i in range(n_samples)])

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
        "patient_ids": patient_ids,
        "t1_mobilenet": t1,
        "t2_effnet": t2_eff,
        "t2_arkplus": t2_ark,
        "tiered_system": tiered,
    }


def _extract_patient_ids(image_ids: pd.Series) -> np.ndarray:
    """Derive 8-digit NIH patient identifiers from image filenames."""
    return image_ids.astype(str).str.split("/").str[-1].str.split("_").str[0].to_numpy()


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
        Dict with y_true, patient_ids, t1_mobilenet, t2_arkplus, tiered_system arrays,
        plus a nested ``cmp_eff_ark`` dict when the EfficientNet CSV is present.

    Raises:
        FileNotFoundError: If the Ark+ predictions CSV does not exist.
    """
    if not os.path.exists(ark_csv):
        raise FileNotFoundError(ark_csv)

    ark = pd.read_csv(ark_csv)
    data: dict[str, np.ndarray | dict[str, np.ndarray]] = {
        "y_true": ark["y_true"].to_numpy().astype(int),
        "patient_ids": _extract_patient_ids(ark["image_id"])
        if "image_id" in ark.columns
        else np.arange(len(ark)).astype(str),
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
                "patient_ids": _extract_patient_ids(merged["image_id"]),
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
    n_iterations: int = 2000,
    seed: int = 42,
    patient_ids: np.ndarray | None = None,
) -> dict:
    """Runs a complete suite of comparative statistical tests between two models.

    Uses patient-clustered bootstrap resampling when ``patient_ids`` is supplied,
    computes primary raw p-values (DeLong for AUC, McNemar for accuracy, patient-clustered
    bootstrap for sensitivity/specificity/F1), applies Holm-Bonferroni correction across
    the five compared metrics, and reports Cohen's h effect sizes for proportion metrics.
    """
    print(f"\nComparing {model1_name} vs {model2_name}...")

    y_true = np.asarray(y_true)
    y_probs1 = np.asarray(y_probs1, dtype=float)
    y_probs2 = np.asarray(y_probs2, dtype=float)

    # 1. Patient-clustered bootstrap CI for key metrics
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
            patient_ids=patient_ids,
        )

    # 2. DeLong test for the AUC difference (correlated ROC curves).
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

    # 5. Primary p-values + Holm-Bonferroni multiplicity correction + effect sizes
    primary_p_raw: list[float] = []
    primary_test_names: list[str] = []
    effect_sizes: dict[str, float] = {}
    for metric in metrics:
        res = bootstrap_results[metric]
        if metric == "auc":
            primary_p_raw.append(delong_p)
            primary_test_names.append("DeLong")
            effect_sizes[metric] = float(res["delta"])
        elif metric == "accuracy":
            primary_p_raw.append(float(mcnemar["p_value"]))
            primary_test_names.append("McNemar")
            effect_sizes[metric] = cohens_h(res["model1_metric"], res["model2_metric"])
        else:
            primary_p_raw.append(float(res["p_value_bootstrap"]))
            primary_test_names.append("Patient-Bootstrap" if patient_ids is not None else "Bootstrap")
            effect_sizes[metric] = cohens_h(res["model1_metric"], res["model2_metric"])

    primary_p_holm = holm_bonferroni(primary_p_raw)
    holm_by_metric = dict(zip(metrics, primary_p_holm, strict=False))
    raw_by_metric = dict(zip(metrics, primary_p_raw, strict=False))
    test_by_metric = dict(zip(metrics, primary_test_names, strict=False))

    # Display clean textual results
    unit = "patient-clustered" if patient_ids is not None else "image-level"
    print(f"Resampling unit: {unit} ({n_iterations} iterations)")
    print("-" * 92)
    print(
        f"{'Metric':<12} | {model1_name:<21} | {model2_name:<21} | "
        f"{'Delta (95% CI)':<22} | {'p_raw / p_holm'}"
    )
    print("-" * 92)
    for metric in metrics:
        res = bootstrap_results[metric]
        print(
            f"{metric.upper():<12} | "
            f"{res['model1_metric']:.3f} ({res['model1_ci'][0]:.3f}-{res['model1_ci'][1]:.3f}) | "
            f"{res['model2_metric']:.3f} ({res['model2_ci'][0]:.3f}-{res['model2_ci'][1]:.3f}) | "
            f"{res['delta']:+.3f} ({res['delta_ci'][0]:+.3f},{res['delta_ci'][1]:+.3f}) | "
            f"{raw_by_metric[metric]:.4f} / {holm_by_metric[metric]:.4f} ({test_by_metric[metric]})"
        )
    print("-" * 92)
    print(f"DeLong AUC p-value:                {delong_p:.6f}")
    print(f"Bootstrap AUC difference p-value:  {bootstrap_results['auc']['p_value_bootstrap']:.6f}")
    print(f"Permutation AUC difference p-value:{perm_auc['p_value']:.6f}")
    print(f"Permutation F1 difference p-value: {perm_f1['p_value']:.6f}")
    print(f"McNemar paired accuracy p-value:   {mcnemar['p_value']:.6f}")
    print("=" * 92)

    return {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "resampling_unit": unit,
        "bootstrap": bootstrap_results,
        "delong_p": delong_p,
        "mcnemar": mcnemar,
        "permutation_auc": perm_auc,
        "permutation_f1": perm_f1,
        "primary_p_raw": raw_by_metric,
        "primary_p_holm": holm_by_metric,
        "primary_test": test_by_metric,
        "effect_sizes": effect_sizes,
    }


def _build_comparisons(data: dict, n_iterations: int, seed: int) -> list[dict]:
    """Construct the comparison list defensively from whatever real data exists."""
    comparisons: list[dict] = []
    pids = data.get("patient_ids")

    # 1. EfficientNet-B4 vs Ark+ Swin (paired Tier-2 comparison)
    if isinstance(data.get("cmp_eff_ark"), dict):
        c = data["cmp_eff_ark"]
        comparisons.append(
            run_comparison(
                c["y_true"],
                c["eff"],
                c["ark"],
                "EfficientNetB4",
                "Ark+ Swin",
                n_iterations=n_iterations,
                seed=seed,
                patient_ids=c.get("patient_ids"),
            )
        )
    elif "t2_effnet" in data:
        comparisons.append(
            run_comparison(
                data["y_true"],
                data["t2_effnet"],
                data["t2_arkplus"],
                "EfficientNetB4",
                "Ark+ Swin",
                n_iterations=n_iterations,
                seed=seed,
                patient_ids=pids,
            )
        )
    else:
        print(
            "Note: EfficientNet predictions not found "
            "(outputs/results/tiered_predictions_efficientnet.csv); "
            "skipping the EfficientNet-vs-Ark+ comparison."
        )

    # 2. Direct baseline comparison: Tier 1 MobileNetV2 (A1) vs Tiered System (A13/A15)
    y_true = data["y_true"]
    comparisons.append(
        run_comparison(
            y_true,
            data["t1_mobilenet"],
            data["tiered_system"],
            "MobileNetV2 (T1)",
            "Tiered System",
            n_iterations=n_iterations,
            seed=seed,
            patient_ids=pids,
        )
    )
    # 3. Standalone Tier 2 Ark+ vs Tiered System
    comparisons.append(
        run_comparison(
            y_true,
            data["t2_arkplus"],
            data["tiered_system"],
            "Ark+ Swin (T2)",
            "Tiered System",
            n_iterations=n_iterations,
            seed=seed,
            patient_ids=pids,
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
                    "Resampling_Unit": comp["resampling_unit"],
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
                    "Effect_Size": comp["effect_sizes"][metric],
                    "Primary_Test": comp["primary_test"][metric],
                    "p_value_raw": comp["primary_p_raw"][metric],
                    "p_value_holm": comp["primary_p_holm"][metric],
                    "p_value_bootstrap": boot["p_value_bootstrap"],
                    "p_value_delong": comp["delong_p"] if metric == "auc" else float("nan"),
                    "p_value_mcnemar": comp["mcnemar"]["p_value"]
                    if metric == "accuracy"
                    else float("nan"),
                    "p_value_permutation": comp["permutation_auc"]["p_value"]
                    if metric == "auc"
                    else (comp["permutation_f1"]["p_value"] if metric == "f1" else float("nan")),
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved structured CSV metrics to: {csv_path}")


def _write_latex(comparisons: list[dict], latex_path: str) -> None:
    """Write a publication-ready LaTeX comparison table with 95% CIs, Holm p-values, and effect sizes."""

    def _fmt_p(val: float) -> str:
        return "$<$0.001" if val < 0.001 else f"{val:.3f}"

    def latex_row(comp: dict, key: str, label: str) -> str:
        boot = comp["bootstrap"][key]
        p_raw = _fmt_p(comp["primary_p_raw"][key])
        p_holm = _fmt_p(comp["primary_p_holm"][key])
        test_name = comp["primary_test"][key]
        es = comp["effect_sizes"][key]
        return (
            f"{label} & "
            f"{boot['model1_metric']:.3f} ({boot['model1_ci'][0]:.3f}--{boot['model1_ci'][1]:.3f}) & "
            f"{boot['model2_metric']:.3f} ({boot['model2_ci'][0]:.3f}--{boot['model2_ci'][1]:.3f}) & "
            f"{boot['delta']:+.3f} ({boot['delta_ci'][0]:+.3f}, {boot['delta_ci'][1]:+.3f}) & "
            f"{es:+.3f} & "
            f"{p_raw} / {p_holm} ({test_name}) \\\\"
        )

    metric_labels = [
        ("auc", "AUC-ROC"),
        ("accuracy", "Accuracy"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("f1", "$F_1$-Score"),
    ]
    sections = []
    for comp in comparisons:
        header = (
            f"\\multicolumn{{6}}{{l}}{{\\textbf{{{comp['model1_name']} vs. "
            f"{comp['model2_name']}}} ({comp['resampling_unit']} bootstrap)}} \\\\"
        )
        body = "\n".join(latex_row(comp, k, lbl) for k, lbl in metric_labels)
        sections.append(header + "\n" + body + "\n\\midrule")
    body_tex = "\n".join(sections)

    latex_table = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Paired statistical comparison with patient-clustered 95\\% bootstrap confidence "
        "intervals, effect sizes ($\\Delta$ for AUC; Cohen's $h$ for proportions), and raw / "
        "Holm--Bonferroni adjusted $p$-values.}\n"
        "\\label{tab:statistical_comparison_detailed}\n"
        "\\begin{tabular}{@{}lccccr@{}}\n"
        "\\toprule\n"
        "\\textbf{Metric} & \\textbf{Model 1 (95\\% CI)} & \\textbf{Model 2 (95\\% CI)} & "
        "\\textbf{$\\Delta$ (95\\% CI)} & \\textbf{Effect} & \\textbf{$p_{\\text{raw}} / p_{\\text{Holm}}$ (Test)} \\\\\n"
        "\\midrule\n"
        f"{body_tex}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    os.makedirs(os.path.dirname(latex_path) if os.path.dirname(latex_path) else ".", exist_ok=True)
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
