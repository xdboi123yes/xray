"""Uncertainty quantification package providing MC Dropout and Test-Time Augmentation."""

from __future__ import annotations

from core.uncertainty.mc_dropout import (
    analyze_uncertainty_batch,
    analyze_uncertainty_vs_correctness,
    compute_mutual_information,
    compute_predictive_entropy,
    plot_uncertainty_distribution,
    print_uncertainty_summary,
)
from core.uncertainty.tta import TestTimeAugmenter

__all__ = [
    "TestTimeAugmenter",
    "analyze_uncertainty_batch",
    "analyze_uncertainty_vs_correctness",
    "compute_mutual_information",
    "compute_predictive_entropy",
    "plot_uncertainty_distribution",
    "print_uncertainty_summary",
]
