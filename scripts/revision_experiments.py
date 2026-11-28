"""MDPI peer-review revision experiment & analysis suite.

Provides reproducible, zero-fabrication CLI analyses addressing Reviewer 1 and
Reviewer 2 Comments 2, 3, 4, 6, 7, 9, 10, and 15:
  --dataset-audit       Patient-level split verification & NIH label breakdown (C2, C3)
  --routing-grid        Validation-only grid search over (tau0, W, delta) (R1, C6)
  --order-sensitivity   10-seed random permutations + drift-order stress tests (C7)
  --threshold-sweep     Clinical operating-point sweep (sensitivity, FNR, referral) (C15)
  --calibration-report  15-bin equal-width ECE, class-ECE, Brier, NLL, reliability plot (C9)
  --synthetic-audit     FID log summary + real-vs-synthetic qualitative visual grid (C4)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tcrp.evaluation.metrics import (
    brier_score,
    calibration_slope_intercept,
    classwise_ece,
    compute_all_metrics,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_bin_stats,
)


def simulate_dynamic_routing(
    t1_probs: np.ndarray,
    t2_probs: np.ndarray,
    tau0: float = 0.75,
    window: int = 50,
    delta: float = 0.05,
    low_band: float = 0.65,
    high_band: float = 0.85,
    tau_min: float = 0.50,
    tau_max: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the exact deadband dynamic router over a sequence of radiographs.

    Returns:
        Tuple of ``(routed_probs, escalated_mask, threshold_trace)``.
    """
    n = len(t1_probs)
    t1_conf = np.maximum(t1_probs, 1.0 - t1_probs)
    threshold = float(tau0)
    recent: list[float] = []
    escalated = np.zeros(n, dtype=bool)
    routed = np.empty(n, dtype=float)
    trace = np.empty(n, dtype=float)

    for i in range(n):
        c = float(t1_conf[i])
        recent.append(c)
        if len(recent) > window:
            recent.pop(0)
            mean_c = sum(recent) / window
            if mean_c < low_band:
                threshold = max(tau_min, threshold - delta)
            elif mean_c > high_band:
                threshold = min(tau_max, threshold + delta)
        trace[i] = threshold
        esc = c < threshold
        escalated[i] = esc
        routed[i] = float(t2_probs[i]) if esc else float(t1_probs[i])

    return routed, escalated, trace


def run_dataset_audit(
    raw_csv: str = "data/raw/Data_Entry_2017.csv",
    processed_dir: str = "data/processed",
    output_json: str = "outputs/results/dataset_audit.json",
) -> dict[str, Any]:
    """Audit raw NIH labels and verify patient-level disjointness across splits."""
    report: dict[str, Any] = {}
    if os.path.exists(raw_csv):
        raw = pd.read_csv(raw_csv)
        raw["Patient ID"] = raw["Image Index"].astype(str).str.split("_").str[0]
        mask_pneumo = raw["Finding Labels"].str.contains("Pneumothorax", na=False)
        mask_pneumo_only = raw["Finding Labels"] == "Pneumothorax"
        mask_normal = raw["Finding Labels"] == "No Finding"
        mask_other = ~(mask_pneumo | mask_normal)
        report["raw_nih"] = {
            "total_images": int(len(raw)),
            "total_patients": int(raw["Patient ID"].nunique()),
            "pneumothorax_all_images": int(mask_pneumo.sum()),
            "pneumothorax_all_patients": int(raw.loc[mask_pneumo, "Patient ID"].nunique()),
            "pneumothorax_only_images": int(mask_pneumo_only.sum()),
            "pneumothorax_comorbid_images": int((mask_pneumo & ~mask_pneumo_only).sum()),
            "no_finding_images": int(mask_normal.sum()),
            "no_finding_patients": int(raw.loc[mask_normal, "Patient ID"].nunique()),
            "excluded_other_pathology_images": int(mask_other.sum()),
            "excluded_other_pathology_patients": int(raw.loc[mask_other, "Patient ID"].nunique()),
        }

    splits: dict[str, Any] = {}
    patient_sets: dict[str, set[str]] = {}
    for split_name in ("train", "val", "calibration", "test"):
        p = os.path.join(processed_dir, f"{split_name}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        pids = df["Image Index"].astype(str).str.split("/").str[-1].str.split("_").str[0]
        pset = set(pids)
        patient_sets[split_name] = pset
        n_pos = int((df["Label"] == 1).sum())
        n_neg = int((df["Label"] == 0).sum())
        splits[split_name] = {
            "images": int(len(df)),
            "unique_patients": int(len(pset)),
            "pneumothorax": n_pos,
            "normal": n_neg,
            "prevalence_pct": round(100.0 * n_pos / max(1, len(df)), 2),
        }

    overlaps: dict[str, int] = {}
    names = list(patient_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlaps[f"{a}_vs_{b}"] = int(len(patient_sets[a] & patient_sets[b]))

    report["splits"] = splits
    report["pairwise_patient_overlaps"] = overlaps
    report["strictly_patient_disjoint"] = all(v == 0 for v in overlaps.values()) and bool(overlaps)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved dataset audit to {output_json} (disjoint={report['strictly_patient_disjoint']})")
    return report


def run_routing_grid(
    val_predictions_csv: str,
    output_csv: str = "outputs/results/routing_grid_val.csv",
) -> pd.DataFrame:
    """Validation-only grid search over (tau0, W, delta) for Reviewer 1 & Comment 6."""
    df = pd.read_csv(val_predictions_csv)
    y_true = df["y_true"].to_numpy(dtype=int)
    t1 = df["tier1_prob"].to_numpy(dtype=float)
    t2 = df["tier2_prob"].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for tau0 in (0.65, 0.70, 0.75, 0.80, 0.85):
        for window in (25, 50, 100):
            for delta in (0.00, 0.02, 0.05, 0.10):
                routed, escalated, trace = simulate_dynamic_routing(
                    t1, t2, tau0=tau0, window=window, delta=delta
                )
                m = compute_all_metrics(y_true, routed, threshold=0.5)
                rows.append(
                    {
                        "tau0": tau0,
                        "window": window,
                        "delta": delta,
                        "escalation_rate_pct": float(100.0 * escalated.mean()),
                        "mean_tau": float(trace.mean()),
                        "auc_roc": m["auc_roc"],
                        "accuracy": m["accuracy"],
                        "sensitivity": m["sensitivity"],
                        "specificity": m["specificity"],
                        "f1": m["f1"],
                        "ece": expected_calibration_error(y_true, routed, n_bins=15),
                    }
                )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved validation routing grid ({len(out_df)} configs) to {output_csv}")
    return out_df


def run_order_sensitivity(
    test_predictions_csv: str,
    n_permutations: int = 10,
    tau0: float = 0.75,
    window: int = 50,
    delta: float = 0.05,
    t1_ms: float = 5.4,
    t2_ms: float = 17.2,
    output_json: str = "outputs/results/order_sensitivity.json",
) -> dict[str, Any]:
    """Evaluate dynamic-threshold sensitivity to random permutations and drift orderings (Comment 7)."""
    df = pd.read_csv(test_predictions_csv)
    y_true = df["y_true"].to_numpy(dtype=int)
    t1 = df["tier1_prob"].to_numpy(dtype=float)
    t2 = df["tier2_prob"].to_numpy(dtype=float)
    t1_conf = np.maximum(t1, 1.0 - t1)

    def _eval_order(order: np.ndarray, label: str) -> dict[str, Any]:
        routed, esc, trace = simulate_dynamic_routing(
            t1[order], t2[order], tau0=tau0, window=window, delta=delta
        )
        m = compute_all_metrics(y_true[order], routed, threshold=0.5)
        esc_rate = float(esc.mean())
        return {
            "order": label,
            "referral_rate_pct": 100.0 * esc_rate,
            "auc_roc": m["auc_roc"],
            "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "expected_single_pass_ms": t1_ms + esc_rate * t2_ms,
            "tau_mean": float(trace.mean()),
            "tau_min": float(trace.min()),
            "tau_max": float(trace.max()),
        }

    perm_runs: list[dict[str, Any]] = []
    for seed in range(n_permutations):
        rng = np.random.default_rng(seed)
        perm_runs.append(_eval_order(rng.permutation(len(y_true)), f"random_seed_{seed}"))

    def _summarize(key: str) -> dict[str, float]:
        vals = [float(r[key]) for r in perm_runs]
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # Stress-test non-stationary arrival orders
    stress_tests = {
        "original_order": _eval_order(np.arange(len(y_true)), "original"),
        "easy_to_hard_sorted": _eval_order(np.argsort(-t1_conf), "easy_to_hard"),
        "hard_to_easy_sorted": _eval_order(np.argsort(t1_conf), "hard_to_easy"),
        "negatives_then_positives": _eval_order(np.argsort(y_true), "neg_then_pos"),
        "positives_then_negatives": _eval_order(np.argsort(-y_true), "pos_then_neg"),
    }

    report = {
        "n_permutations": n_permutations,
        "params": {"tau0": tau0, "window": window, "delta": delta},
        "random_order_summary": {
            k: _summarize(k)
            for k in (
                "referral_rate_pct",
                "auc_roc",
                "accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "expected_single_pass_ms",
            )
        },
        "random_order_runs": perm_runs,
        "drift_stress_tests": stress_tests,
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Saved order sensitivity report to {output_json}")
    return report


def run_threshold_sweep(
    test_predictions_csv: str,
    output_csv: str = "outputs/results/clinical_threshold_sweep.csv",
) -> pd.DataFrame:
    """Evaluate clinical operating points: sensitivity, specificity, FNR, PPV, NPV, referral rate (C15)."""
    df = pd.read_csv(test_predictions_csv)
    y_true = df["y_true"].to_numpy(dtype=int)
    prob_col = "prob_cal" if "prob_cal" in df.columns else "tiered_prob"
    probs = df[prob_col].to_numpy(dtype=float)

    # Candidate operating points: fixed grid + Youden's J + 90%/95%/98% sensitivity targets
    fine_grid = np.linspace(0.01, 0.99, 199)
    youden_best_thr, youden_best = 0.5, -1.0
    sens_targets = {0.90: None, 0.95: None, 0.98: None}
    for thr in fine_grid:
        m = compute_all_metrics(y_true, probs, threshold=float(thr))
        j = m["sensitivity"] + m["specificity"] - 1.0
        if j > youden_best:
            youden_best, youden_best_thr = j, float(thr)
        for target in sens_targets:
            if m["sensitivity"] >= target:
                sens_targets[target] = float(thr)  # highest threshold meeting target sensitivity

    named_points: list[tuple[str, float]] = [
        ("Default (0.50)", 0.50),
        ("Youden J Optimal", youden_best_thr),
    ]
    for target, thr in sens_targets.items():
        if thr is not None:
            named_points.append((f"Target Sensitivity >= {int(target*100)}%", thr))
    for fixed_thr in (0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80):
        named_points.append((f"Fixed ({fixed_thr:.2f})", fixed_thr))

    rows: list[dict[str, Any]] = []
    for label, thr in named_points:
        m = compute_all_metrics(y_true, probs, threshold=thr)
        y_pred = (probs >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fnr = fn / max(1, tp + fn)
        npv = tn / max(1, tn + fn)
        pos_flag_rate = (tp + fp) / max(1, len(y_true))
        rows.append(
            {
                "operating_point": label,
                "threshold": round(thr, 4),
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "fnr": fnr,
                "ppv": m["precision"],
                "npv": npv,
                "f1": m["f1"],
                "accuracy": m["accuracy"],
                "positive_flag_rate": pos_flag_rate,
                "missed_pneumothorax_n": fn,
                "false_alarms_n": fp,
            }
        )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved clinical threshold sweep to {output_csv}")
    return out_df


def run_calibration_report(
    test_predictions_csv: str,
    output_json: str = "outputs/results/calibration_summary.json",
    output_fig: str = "outputs/figures/reliability_diagram_15bin.png",
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute 15-bin equal-width calibration metrics pre/post temperature scaling + reliability figure (C9)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(test_predictions_csv)
    y_true = df["y_true"].to_numpy(dtype=int)
    p_uncal = df["prob_uncal"].to_numpy(dtype=float) if "prob_uncal" in df.columns else df["tiered_prob"].to_numpy(dtype=float)
    p_cal = df["prob_cal"].to_numpy(dtype=float) if "prob_cal" in df.columns else p_uncal

    def _summarize(probs: np.ndarray) -> dict[str, Any]:
        slope, intercept = calibration_slope_intercept(y_true, probs)
        cw = classwise_ece(y_true, probs, n_bins=n_bins)
        return {
            "n_bins": n_bins,
            "binning_scheme": "equal_width",
            "ece": expected_calibration_error(y_true, probs, n_bins=n_bins),
            **cw,
            "brier": brier_score(y_true, probs),
            "nll": negative_log_likelihood(y_true, probs),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "bins": reliability_bin_stats(y_true, probs, n_bins=n_bins),
        }

    report = {
        "before_temperature_scaling": _summarize(p_uncal),
        "after_temperature_scaling": _summarize(p_cal),
    }
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(report, fh, indent=2)

    # Plot side-by-side 15-bin reliability diagrams
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, title, stats in [
        (axes[0], "Before Temperature Scaling", report["before_temperature_scaling"]),
        (axes[1], "After Temperature Scaling", report["after_temperature_scaling"]),
    ]:
        bins = [b for b in stats["bins"] if b["count"] > 0]
        xs = [b["mean_prob"] for b in bins]
        ys = [b["empirical_pos_rate"] for b in bins]
        ax.plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1.5, label="Perfect calibration")
        ax.plot(xs, ys, "o-", color="#0d9488", linewidth=2.2, markersize=6, label="Empirical")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Predicted Probability (Pneumothorax)", fontsize=11.5, fontweight="bold")
        ax.set_title(
            f"{title}\nECE={stats['ece']:.3f} | Brier={stats['brier']:.3f} | NLL={stats['nll']:.3f}",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=10)
    axes[0].set_ylabel("Observed Positive Frequency", fontsize=11.5, fontweight="bold")
    fig.suptitle("Reliability Diagrams (15 Equal-Width Bins)", fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(output_fig) or ".", exist_ok=True)
    fig.savefig(output_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration summary to {output_json} and figure to {output_fig}")
    return report


def run_synthetic_audit(
    synthetic_dir: str = "data/synthetic",
    output_json: str = "outputs/results/synthetic_audit.json",
    output_fig: str = "outputs/figures/synthetic_audit_grid.png",
    generation_summary: str = "outputs/provenance/synthetic_generation_summary.json",
) -> dict[str, Any]:
    """Summarize synthetic generation FID logs & build a sample grid of accepted images (Comment 4).

    Generation parameters are copied from the Stage 1 ``generation_summary`` record, never
    assumed; the Stage 1 real-vs-synthetic grid (``synthetic_quality_grid.png``) is left untouched.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fid_log = os.path.join(synthetic_dir, "fid_log.csv")
    meta_csv = os.path.join(synthetic_dir, "synthetic_metadata.csv")
    accepted_dir = os.path.join(synthetic_dir, "accepted")
    rejected_dir = os.path.join(synthetic_dir, "rejected")

    from tcrp.config import get_settings

    aug = get_settings().augmentation
    summary: dict[str, Any] = {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "pipeline": "StableDiffusionImg2ImgPipeline",
        "guidance_scale": float(aug.sd_guidance_scale),
        "num_inference_steps": int(aug.sd_num_inference_steps),
    }
    if os.path.exists(generation_summary):
        with open(generation_summary) as fh:
            recorded = json.load(fh)
        for key in ("strength", "batch_size", "num_variations", "fid_threshold", "seed_scope",
                    "output_resolution", "unique_synthetic_patients", "non_train_patient_overlap"):
            if key in recorded:
                summary[key] = recorded[key]
    summary |= {
        "accepted_count": len(os.listdir(accepted_dir)) if os.path.isdir(accepted_dir) else 0,
        "rejected_count": len(os.listdir(rejected_dir)) if os.path.isdir(rejected_dir) else 0,
    }
    if os.path.exists(fid_log):
        fdf = pd.read_csv(fid_log)
        summary["batches_total"] = int(len(fdf))
        summary["batches_accepted"] = int(fdf["accepted"].sum())
        summary["fid_mean"] = float(fdf["fid"].mean())
        summary["fid_min"] = float(fdf["fid"].min())
        summary["fid_max"] = float(fdf["fid"].max())
        acc = fdf[fdf["accepted"].astype(bool)]
        if len(acc):
            summary["fid_accepted_mean"] = float(acc["fid"].mean())
            summary["fid_accepted_min"] = float(acc["fid"].min())
            summary["fid_accepted_max"] = float(acc["fid"].max())
    if os.path.exists(meta_csv):
        mdf = pd.read_csv(meta_csv)
        summary["tracked_accepted_records"] = int(len(mdf))

    # Build qualitative grid if accepted synthetic images exist
    if os.path.isdir(accepted_dir):
        imgs = sorted(
            [f for f in os.listdir(accepted_dir) if f.lower().endswith((".png", ".jpg"))]
        )[:8]
        if imgs:
            ncols = min(4, len(imgs))
            nrows = int(np.ceil(len(imgs) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
            axes_arr = np.atleast_1d(axes).ravel()
            for ax, fname in zip(axes_arr, imgs, strict=False):
                im = Image.open(os.path.join(accepted_dir, fname)).convert("L")
                ax.imshow(im, cmap="gray")
                ax.set_title(fname[:22], fontsize=8.5)
                ax.axis("off")
            for ax in axes_arr[len(imgs) :]:
                ax.axis("off")
            fig.suptitle("Sample Accepted Synthetic Pneumothorax Radiographs (FID <= 50)", fontsize=12, fontweight="bold")
            fig.tight_layout()
            os.makedirs(os.path.dirname(output_fig) or ".", exist_ok=True)
            fig.savefig(output_fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
            summary["qualitative_grid_path"] = output_fig

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved synthetic audit to {output_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MDPI revision analysis & audit suite")
    parser.add_argument("--dataset-audit", action="store_true")
    parser.add_argument("--routing-grid", action="store_true")
    parser.add_argument("--order-sensitivity", action="store_true")
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--calibration-report", action="store_true")
    parser.add_argument("--synthetic-audit", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--test-predictions", default="outputs/results/tiered_predictions.csv")
    parser.add_argument("--val-predictions", default="outputs/results/val_predictions.csv")
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument(
        "--require-inputs",
        action="store_true",
        help="Fail instead of silently skipping analyses whose prediction CSVs are missing.",
    )
    args = parser.parse_args()

    run_all = args.all
    if args.require_inputs:
        missing = [p for p in (args.test_predictions, args.val_predictions) if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"[revision_experiments] missing required predictions: {missing}")
    if run_all or args.dataset_audit:
        run_dataset_audit()
    if (run_all or args.routing_grid) and os.path.exists(args.val_predictions):
        run_routing_grid(args.val_predictions)
    if (run_all or args.order_sensitivity) and os.path.exists(args.test_predictions):
        run_order_sensitivity(args.test_predictions, n_permutations=args.n_permutations)
    if (run_all or args.threshold_sweep) and os.path.exists(args.test_predictions):
        run_threshold_sweep(args.test_predictions)
    if (run_all or args.calibration_report) and os.path.exists(args.test_predictions):
        run_calibration_report(args.test_predictions)
    if run_all or args.synthetic_audit:
        run_synthetic_audit()


if __name__ == "__main__":
    main()
