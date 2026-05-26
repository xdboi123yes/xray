"""Uncertainty quantification package providing MC Dropout and Test-Time Augmentation."""

from __future__ import annotations

from core.uncertainty.mc_dropout import (
    compute_predictive_entropy,
    compute_mutual_information,
    analyze_uncertainty_batch,
    analyze_uncertainty_vs_correctness,
    plot_uncertainty_distribution,
    print_uncertainty_summary,
)
from core.uncertainty.tta import TestTimeAugmenter

__all__ = [
    "compute_predictive_entropy",
    "compute_mutual_information",
    "analyze_uncertainty_batch",
    "analyze_uncertainty_vs_correctness",
    "plot_uncertainty_distribution",
    "print_uncertainty_summary",
    "TestTimeAugmenter",
]
