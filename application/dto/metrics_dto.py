"""Classification metrics and statistical testing Data Transfer Objects (DTOs).

Defines strict type schemas for single-model metrics, confidence intervals,
and paired model comparative statistics.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassificationMetricsDTO(BaseModel):
    """Detailed clinical performance metrics for binary chest X-ray classifiers."""

    auc_roc: float = Field(..., ge=0.0, le=1.0, description="Area Under the ROC curve.")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall classification accuracy.")
    sensitivity: float = Field(..., ge=0.0, le=1.0, description="Sensitivity / Recall / True Positive Rate.")
    specificity: float = Field(..., ge=0.0, le=1.0, description="Specificity / True Negative Rate.")
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision / Positive Predictive Value.")
    f1: float = Field(..., ge=0.0, le=1.0, description="F1 diagnostic score.")
    mcc: float = Field(..., ge=-1.0, le=1.0, description="Matthews Correlation Coefficient.")
    ece: float | None = Field(None, ge=0.0, le=1.0, description="Expected Calibration Error.")


class ConfidenceIntervalDTO(BaseModel):
    """Represents an empirical point estimate and confidence interval bound."""

    metric: str = Field(..., description="Name of the targeted classification metric.")
    point_estimate: float = Field(..., description="Observed point estimate value.")
    lower_bound: float = Field(..., description="Lower confidence interval boundary.")
    upper_bound: float = Field(..., description="Upper confidence interval boundary.")
    alpha: float = Field(0.05, description="Significance level (e.g. 0.05 for 95% CI).")


class StatisticalTestResultDTO(BaseModel):
    """Results structure representing clinical hypothesis testing comparisons."""

    test_name: str = Field(..., description="Name of the test (e.g. DeLong, McNemar, Permutation).")
    observed_difference: float = Field(..., description="Point delta value: Model 2 - Model 1.")
    statistic: float | None = Field(None, description="Calculated paired test statistic value.")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Calculated two-sided probability p-value.")
    interpretation: str = Field(..., description="Textual explanation of statistical significance.")
