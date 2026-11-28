"""Generate thesis figures (PNG) from genuine evaluation artifacts.

Runs headless (Agg backend) and reads only artifacts that are genuine and
mutually consistent with ``ablation.json``:

* ``outputs/results/ablation.json``        -> ``thesis/figures/ablation_overview.png``
* ``outputs/results/A13_Tiered_ArkPlus.json`` and ``A14_CheXpert_ZeroShot.json``
  -> ``thesis/figures/cross_dataset_generalization.png``

Figures that require per-sample predictions (reliability diagram, DeLong
significance heatmap, the four analysis-notebook plots) are intentionally NOT
produced here: the only per-sample exports in the repository
(``tiered_predictions.csv``, ``val_predictions.csv``) come from a stale
pipeline whose Tier-1 column scores AUC ~0.78, contradicting the genuine
ablation results. They can be added once those predictions are regenerated.

Usage::

    python scripts/build_thesis_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(REPO_ROOT / ".matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = REPO_ROOT / "outputs" / "results"
PAPER_DIR = REPO_ROOT / "paper"
FIGURES_DIR = REPO_ROOT / "thesis" / "figures"

ACCENT = "#0d9488"  # teal, matching the application theme
MUTED = "#94a3b8"
HIGHLIGHT = "#ea580c"  # orange for the best configuration


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure to both paper/ and thesis/figures/ directories."""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_paper = PAPER_DIR / name
    fig.savefig(out_paper, dpi=300, bbox_inches="tight")
    print(f"wrote {out_paper.relative_to(REPO_ROOT)}")
    if FIGURES_DIR.parent.exists():
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_thesis = FIGURES_DIR / name
        fig.savefig(out_thesis, dpi=300, bbox_inches="tight")


def build_ablation_overview() -> None:
    """Horizontal AUC and ECE bars for every ablation with print-legible labels at 100% zoom."""
    rows = json.loads((RESULTS_DIR / "ablation.json").read_text())
    ids = [r["ablation_id"] for r in rows]
    names = [r["name"] for r in rows]
    aucs = [r["metrics"]["auc_roc"] for r in rows]
    eces = [r["metrics"].get("ece") for r in rows]
    labels = [f"{i}: {n}" for i, n in zip(ids, names, strict=False)]
    best = max(range(len(aucs)), key=lambda k: aucs[k])

    y = range(len(rows))
    fig, (ax_auc, ax_ece) = plt.subplots(1, 2, figsize=(15.5, 9.5), sharey=True)

    auc_min = min(0.75, min(aucs) - 0.03)
    auc_colors = [HIGHLIGHT if k == best else ACCENT for k in range(len(rows))]
    ax_auc.barh(list(y), aucs, color=auc_colors, height=0.72)
    ax_auc.set_xlim(auc_min, 0.96)
    ax_auc.set_xlabel("AUC-ROC", fontsize=14, fontweight="bold")
    ax_auc.set_title("Discrimination (AUC-ROC)", fontsize=15, fontweight="bold", pad=12)
    ax_auc.set_yticks(list(y))
    ax_auc.set_yticklabels(labels, fontsize=13, fontweight="semibold")
    ax_auc.tick_params(axis="x", labelsize=12.5)
    ax_auc.grid(axis="x", linestyle="--", alpha=0.35)
    ax_auc.invert_yaxis()
    for k, v in enumerate(aucs):
        ax_auc.text(v + 0.002, k, f"{v:.3f}", va="center", fontsize=12, fontweight="bold")

    ece_vals = [(e if e is not None else 0.0) for e in eces]
    ece_max = max([e for e in eces if e is not None] + [0.10]) + 0.025
    ax_ece.barh(list(y), ece_vals, color=MUTED, height=0.72)
    ax_ece.set_xlim(0, ece_max)
    ax_ece.set_xlabel("Expected Calibration Error (15 bins)", fontsize=14, fontweight="bold")
    ax_ece.set_title("Calibration (ECE, lower is better)", fontsize=15, fontweight="bold", pad=12)
    ax_ece.tick_params(axis="x", labelsize=12.5)
    ax_ece.grid(axis="x", linestyle="--", alpha=0.35)
    for k, e in enumerate(eces):
        text = "n/a" if e is None else f"{e:.3f}"
        ax_ece.text(
            max(e, 0.001) + 0.002 if e is not None else 0.002,
            k,
            text,
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle("Ablation Study Overview (Discrimination & Calibration)", fontsize=16, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.01,
        "Evaluated on the NIH ChestX-Ray14 test set except A14 (zero-shot CheXpert). Highlighted bar marks highest AUC.",
        ha="center",
        fontsize=11.5,
        color="#334155",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    save_figure(fig, "ablation_overview.png")
    plt.close(fig)


def build_cross_dataset_figure() -> None:
    """Grouped bars comparing the proposed system on NIH (A13) vs CheXpert (A14)."""
    nih = json.loads((RESULTS_DIR / "A13_Tiered_ArkPlus.json").read_text())["metrics"]
    chex = json.loads((RESULTS_DIR / "A14_CheXpert_ZeroShot.json").read_text())["metrics"]

    keys = ["auc_roc", "accuracy", "sensitivity", "specificity", "f1"]
    labels = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
    nih_vals = [nih[k] for k in keys]
    chex_vals = [chex[k] for k in keys]

    x = range(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([i - width / 2 for i in x], nih_vals, width, label="NIH (in-domain, A13)", color=ACCENT)
    bars2 = ax.bar(
        [i + width / 2 for i in x], chex_vals, width,
        label="CheXpert (zero-shot, A14)", color=MUTED,
    )

    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Cross-dataset generalization of the proposed tiered system",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(frameon=True, fontsize=11)
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "A14 is the full CheXpert validation frontal cohort (202 images, ~3.5% prevalence), evaluated zero-shot at a fixed 0.5 threshold.\nThe collapse in specificity and accuracy reflects the prevalence/threshold shift; AUC is the threshold-free metric.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#334155",
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "cross_dataset_generalization.png")
    plt.close(fig)


def main() -> None:
    for builder in (build_ablation_overview, build_cross_dataset_figure):
        builder()


if __name__ == "__main__":
    main()
