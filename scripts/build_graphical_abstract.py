"""Generate a clean graphical abstract image (PNG) for the TCR-P system."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(REPO_ROOT / ".matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def build_graphical_abstract() -> Path:
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    bg = patches.Rectangle((0, 0), 12, 6.5, color="#f8fafc")
    ax.add_patch(bg)

    # Title Banner
    ax.text(6, 6.0, "Tiered Confidence Routing for Pneumothorax (TCR-P)", fontsize=16, fontweight="bold", ha="center", color="#0f172a")
    ax.text(6, 5.6, "Clinically-Aware Selective Prediction & Uncertainty Calibration Stack", fontsize=11, ha="center", color="#475569")

    # Step 1: Input Radiograph Box
    b1 = patches.FancyBboxPatch((0.5, 2.2), 2.2, 2.6, boxstyle="round,pad=0.3", ec="#94a3b8", fc="#ffffff", lw=1.5)
    ax.add_patch(b1)
    ax.text(1.6, 4.3, "Input Chest X-Ray", fontsize=11, fontweight="bold", ha="center", color="#1e293b")
    ax.text(1.6, 3.4, "Frontal View\n(NIH / CheXpert)", fontsize=9.5, ha="center", color="#64748b")
    ax.text(1.6, 2.5, "Standard Preprocessing", fontsize=8.5, ha="center", color="#0d9488", fontweight="bold")

    # Arrow 1 -> 2
    ax.annotate("", xy=(3.1, 3.5), xytext=(2.7, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0d9488"))

    # Step 2: Tier 1 MobileNetV2
    b2 = patches.FancyBboxPatch((3.2, 2.2), 2.4, 2.6, boxstyle="round,pad=0.3", ec="#0d9488", fc="#f0fdf4", lw=2)
    ax.add_patch(b2)
    ax.text(4.4, 4.3, "Tier 1: Screening", fontsize=11, fontweight="bold", ha="center", color="#065f46")
    ax.text(4.4, 3.7, "MobileNetV2", fontsize=10, fontweight="bold", ha="center", color="#0d9488")
    ax.text(4.4, 3.0, "High-Speed Screening\nConfidence Score c(x)", fontsize=9, ha="center", color="#334155")
    ax.text(4.4, 2.4, "Fast Inference", fontsize=8.5, ha="center", color="#166534")

    # Arrow 2 -> Router Decision
    ax.annotate("", xy=(5.9, 3.5), xytext=(5.6, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#0284c7"))

    # Step 3: Confidence Router
    diamond = patches.Polygon([[6.6, 4.1], [7.6, 3.5], [6.6, 2.9], [5.6, 3.5]], ec="#0284c7", fc="#e0f2fe", lw=2)
    ax.add_patch(diamond)
    ax.text(6.6, 3.6, "Confidence\nRouter", fontsize=10, fontweight="bold", ha="center", color="#0369a1")
    ax.text(6.6, 3.05, "c(x) ≥ τ ?", fontsize=9, fontweight="bold", ha="center", color="#0284c7")

    # Branch 1: Accept High Confidence (Up / Right)
    ax.annotate("Yes (High Conf.)", xy=(8.2, 4.7), xytext=(6.6, 4.1),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#16a34a", connectionstyle="angle,angleA=90,angleB=0,rad=5"))
    b_pass = patches.FancyBboxPatch((8.3, 4.3), 3.2, 1.0, boxstyle="round,pad=0.2", ec="#16a34a", fc="#f0fdf4", lw=1.5)
    ax.add_patch(b_pass)
    ax.text(9.9, 4.8, "Direct Decision (Tier 1)", fontsize=10, fontweight="bold", ha="center", color="#15803d")
    ax.text(9.9, 4.4, "Rapid clearance (~75% of cohort)", fontsize=8.5, ha="center", color="#166534")

    # Branch 2: Escalate Low Confidence (Down / Right)
    ax.annotate("No (Escalate)", xy=(7.4, 1.8), xytext=(6.6, 2.9),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#ea580c", connectionstyle="angle,angleA=-90,angleB=0,rad=5"))

    # Step 4: Tier 2 Specialist + Uncertainty Stack
    b4 = patches.FancyBboxPatch((7.5, 0.4), 4.0, 2.6, boxstyle="round,pad=0.3", ec="#ea580c", fc="#fff7ed", lw=2)
    ax.add_patch(b4)
    ax.text(9.5, 2.6, "Tier 2 Specialist + Uncertainty Stack", fontsize=11, fontweight="bold", ha="center", color="#9a3412")
    ax.text(9.5, 2.1, "EfficientNet-B4 / Ark+ Swin", fontsize=10, fontweight="bold", ha="center", color="#c2410c")
    ax.text(9.5, 1.5, "• MC-Dropout (Epistemic)\n• Test-Time Augmentation (TTA)\n• Conformal Prediction Sets (95% Coverage)", fontsize=9, ha="center", color="#431407")
    ax.text(9.5, 0.6, "Calibrated Output + Abstraction / Human Deferral", fontsize=8.5, fontweight="bold", ha="center", color="#c2410c")

    out_paper = REPO_ROOT / "paper" / "graphical_abstract.png"
    out_paper.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_paper, dpi=300, bbox_inches="tight")
    print(f"wrote {out_paper.relative_to(REPO_ROOT)}")
    plt.close(fig)
    return out_paper


if __name__ == "__main__":
    build_graphical_abstract()
