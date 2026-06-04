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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "outputs" / "results"
FIGURES_DIR = REPO_ROOT / "thesis" / "figures"

ACCENT = "#0d9488"  # teal, matching the application theme
MUTED = "#94a3b8"
HIGHLIGHT = "#ea580c"  # orange for the best configuration


def build_ablation_overview() -> Path:
    """Horizontal AUC and ECE bars for every ablation (A1-A15)."""
    rows = json.loads((RESULTS_DIR / "ablation.json").read_text())
    ids = [r["ablation_id"] for r in rows]
    names = [r["name"] for r in rows]
    aucs = [r["metrics"]["auc_roc"] for r in rows]
    eces = [r["metrics"].get("ece") for r in rows]
    labels = [f"{i}  {n}" for i, n in zip(ids, names, strict=True)]
    best = max(range(len(aucs)), key=lambda k: aucs[k])

    y = range(len(rows))
    fig, (ax_auc, ax_ece) = plt.subplots(1, 2, figsize=(13, 7), sharey=True)

    auc_colors = [HIGHLIGHT if k == best else ACCENT for k in range(len(rows))]
    ax_auc.barh(list(y), aucs, color=auc_colors)
    ax_auc.set_xlim(0.85, 0.93)
    ax_auc.set_xlabel("AUC-ROC")
    ax_auc.set_title("Discrimination (AUC-ROC)")
    ax_auc.set_yticks(list(y))
    ax_auc.set_yticklabels(labels, fontsize=8)
    ax_auc.invert_yaxis()
    for k, v in enumerate(aucs):
        ax_auc.text(v + 0.0015, k, f"{v:.3f}", va="center", fontsize=7)

    ece_vals = [(e if e is not None else 0.0) for e in eces]
    ax_ece.barh(list(y), ece_vals, color=MUTED)
    ax_ece.set_xlabel("Expected Calibration Error")
    ax_ece.set_title("Calibration (ECE, lower is better)")
    for k, e in enumerate(eces):
        text = "n/a" if e is None else f"{e:.3f}"
        ax_ece.text(0.001, k, text, va="center", fontsize=7)

    fig.suptitle("Ablation study overview (A1-A15)", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "All configurations evaluated on the NIH test set except A14, "
        "which is a zero-shot evaluation on CheXpert.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = FIGURES_DIR / "ablation_overview.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def build_cross_dataset_figure() -> Path:
    """Grouped bars comparing the proposed system on NIH (A13) vs CheXpert (A14)."""
    nih = json.loads((RESULTS_DIR / "A13_Tiered_ArkPlus.json").read_text())["metrics"]
    chex = json.loads((RESULTS_DIR / "A14_CheXpert_ZeroShot.json").read_text())["metrics"]

    keys = ["auc_roc", "accuracy", "sensitivity", "specificity", "f1"]
    labels = ["AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
    nih_vals = [nih[k] for k in keys]
    chex_vals = [chex[k] for k in keys]

    x = range(len(keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar([i - width / 2 for i in x], nih_vals, width, label="NIH (in-domain, A13)", color=ACCENT)
    bars2 = ax.bar([i + width / 2 for i in x], chex_vals, width, label="CheXpert (zero-shot, A14)", color=MUTED)

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Cross-dataset generalization of the proposed tiered system", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "cross_dataset_generalization.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for builder in (build_ablation_overview, build_cross_dataset_figure):
        out = builder()
        print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
