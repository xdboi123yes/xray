"""Post-hoc temperature scaling, split-conformal prediction, and uncertainty analysis."""

from __future__ import annotations

from tcrp.uncertainty.analysis import (
    TestTimeAugmenter,
    analyze_uncertainty_batch,
    analyze_uncertainty_vs_correctness,
    compute_mutual_information,
    compute_predictive_entropy,
    plot_uncertainty_distribution,
    print_uncertainty_summary,
)
from tcrp.uncertainty.calibration import (
    ModelWithTemperature,
    calibrate_temperature,
    compute_ece,
    plot_reliability_diagram,
)
from tcrp.uncertainty.conformal import ConformalEvaluationReport, ConformalPredictor

__all__ = [
    "ConformalEvaluationReport",
    "ConformalPredictor",
    "ModelWithTemperature",
    "TestTimeAugmenter",
    "analyze_uncertainty_batch",
    "analyze_uncertainty_vs_correctness",
    "calibrate_temperature",
    "compute_ece",
    "compute_mutual_information",
    "compute_predictive_entropy",
    "plot_reliability_diagram",
    "plot_uncertainty_distribution",
    "print_uncertainty_summary",
]
