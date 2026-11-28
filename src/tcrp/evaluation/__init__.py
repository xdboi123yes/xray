"""Clinical classification metrics, hypothesis testing, and HiResCAM explainability."""

from __future__ import annotations

from tcrp.evaluation.explainability import (
    ReshapeTransform,
    XRayHiResCAM,
    combined_overlay,
    resolve_target_layers,
    side_by_side,
    suppress_background,
)
from tcrp.evaluation.metrics import (
    brier_score,
    calibration_slope_intercept,
    classwise_ece,
    compute_all_metrics,
    expected_calibration_error,
    find_optimal_threshold,
    negative_log_likelihood,
    reliability_bin_stats,
    wilson_ci,
)
from tcrp.evaluation.provenance import (
    ProvenanceVerificationResult,
    build_provenance_manifest,
    compute_file_sha256,
    verify_provenance_manifest,
)
from tcrp.evaluation.stats import (
    bootstrap_ci,
    cohens_h,
    delong_test,
    holm_bonferroni,
    mcnemar_test,
    permutation_test,
)

__all__ = [
    "ProvenanceVerificationResult",
    "ReshapeTransform",
    "XRayHiResCAM",
    "bootstrap_ci",
    "brier_score",
    "build_provenance_manifest",
    "calibration_slope_intercept",
    "classwise_ece",
    "cohens_h",
    "combined_overlay",
    "compute_all_metrics",
    "compute_file_sha256",
    "delong_test",
    "expected_calibration_error",
    "find_optimal_threshold",
    "holm_bonferroni",
    "mcnemar_test",
    "negative_log_likelihood",
    "permutation_test",
    "reliability_bin_stats",
    "resolve_target_layers",
    "side_by_side",
    "suppress_background",
    "verify_provenance_manifest",
    "wilson_ci",
]
