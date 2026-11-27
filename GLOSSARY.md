# TCR-P Pneumothorax Screening

Domain vocabulary for the TCR-P tiered confidence-routing pneumothorax screening system and its MDPI *Diagnostics* peer-review revision.

## Architecture & Routing

**TCR-P (Tiered Confidence-Routing System)**:
A two-stage chest radiograph screening cascade where a fast Tier 1 classifier resolves confident cases and escalates uncertain cases to one selected Tier 2 specialist backbone.
_Avoid_: ensemble, combined Stage-2 model

**Tier 1**:
The lightweight screening classifier (MobileNetV2) that evaluates every incoming radiograph and resolves high-confidence cases without invoking Tier 2.
_Avoid_: baseline-only stage

**Tier 2**:
A single selected high-capacity specialist backbone (either EfficientNet-B4 or Ark+ Swin, never both combined) that evaluates escalated radiographs with uncertainty quantification.
_Avoid_: Stage-2 ensemble, combined backbone

**Adaptive Routing Threshold**:
The label-free confidence cutoff ($\tau_t$) updated via a deadband controller over a sliding FIFO window of recent Tier-1 confidences to regulate the escalation rate.
_Avoid_: static cutoff

**Escalation**:
Routing a radiograph whose Tier-1 confidence falls below the Adaptive Routing Threshold from Tier 1 to Tier 2 for deeper analysis.
_Avoid_: deferral, human referral

**Human Deferral**:
Flagging a Tier-2 prediction for radiologist review when total predictive variance exceeds the uncertainty threshold or the conformal prediction set contains more than one class.
_Avoid_: escalation, Tier-2 referral

## Data & Splits

**Patient-Disjoint Partition**:
A dataset split in which every radiograph belonging to the same patient ID appears in at most one subset (train, validation, calibration, or test), guaranteeing zero cross-split patient overlap.
_Avoid_: image-level split, random split

**Validation Split**:
The patient-disjoint subset used for model selection, early stopping, validation-only routing tuning, and post-hoc temperature scaling ($T_{\text{cal}}$).
_Avoid_: calibration set

**Calibration Split**:
The dedicated patient-disjoint subset reserved exclusively for computing split-conformal non-conformity quantiles ($\hat{q}$) on temperature-scaled probabilities, strictly separate from the Validation Split.
_Avoid_: validation set

**Synthetic Augmentation Batch**:
Synthetic pneumothorax radiographs generated from real Training Split pneumothorax seeds via Stable Diffusion img2img and admitted into training only when batch Inception-v3 FID meets the quality threshold.

## Uncertainty & Revision Tracks

**Uncertainty Decomposition**:
The Law of Total Variance split of Tier-2 predictive variance into within-augmentation epistemic variance (Monte Carlo Dropout) and between-augmentation input-perturbation sensitivity (Test-Time Augmentation).

**Track A**:
The non-rerun MDPI revision package updating text, notation, tables, figures, and transparent leakage disclosures against the submitted baseline results.

**Track B**:
The full patient-disjoint GPU rerun producing fresh checkpoints, leak-free synthetic data, updated tables/figures, latency benchmarks, and a SHA-256 provenance manifest.
