# Architecture

The research codebase is consolidated into a single minimal Python package,
`src/tcrp/`, with narrow domain interfaces and acyclic internal contracts
enforced by `import-linter` (see `pyproject.toml`).

```text
scripts/ & notebooks/  -->  tcrp.training  -->  tcrp.{data, models, uncertainty, evaluation}
                                                        |
                                                        v
                                               tcrp.{config, routing}
```

## Modules

| Module | Path | Responsibility |
| --- | --- | --- |
| Configuration & Integrity | `tcrp.config` | Type-safe YAML/env settings (`Settings`, `get_settings`) and `guard_mock` integrity enforcement. |
| Confidence Routing | `tcrp.routing` | `StaticThresholdRouter` and `DynamicThresholdRouter` (label-free Adaptive Routing Threshold deadband controller). |
| Data & Synthetics | `tcrp.data` | Strict four-way Patient-Disjoint Partition (`splits.py`), NIH/CheXpert datasets (`datasets.py`), classical and Mixup/CutMix augmentations (`augmentation.py`), and FID-gated Synthetic Augmentation Batch generation (`synthetic.py`) behind injectable generator/FID adapters. |
| Models & Cascade | `tcrp.models` | Deep `BaseClassifier` centralizing BatchNorm-safe MC Dropout, TTA, and Law-of-Total-Variance Uncertainty Decomposition; `ModelFactory` with `Tier1MobileNet`, `Tier2EfficientNet`, and `Tier2ArkPlus`; `TieredSystem` cascade for Escalation and Human Deferral. |
| Uncertainty & Calibration | `tcrp.uncertainty` | Validation-split post-hoc temperature scaling (`calibration.py`), disjoint Calibration-split split-conformal prediction (`conformal.py`), and entropy/mutual-information/TTA analysis (`analysis.py`). |
| Evaluation & Explainability | `tcrp.evaluation` | Binary, calibration, and conformal metrics (`metrics.py`), patient-clustered bootstrap, DeLong, McNemar, Holm-Bonferroni, Cohen's h, and Wilson CI (`stats.py`), plus HiResCAM and Tier 1 vs. Tier 2 attention maps (`explainability.py`). |
| Training & Export | `tcrp.training` | Observer-aware `Trainer`, `TrainingConfig`, `train_model`, checkpointing, class-weighted loss, LR scheduler (`trainer.py`), lifecycle observers (`observers.py`), `AblationRunner` (`ablation.py`), and `ONNXModelExporter` (`export.py`). |

## Tiered Inference Path (TCR-P)

1. **Tier 1** (`Tier1MobileNet`) evaluates every incoming radiograph and outputs a fast prediction and confidence.
2. **Adaptive Routing Threshold** (`DynamicThresholdRouter`) updates its sliding FIFO window and decides whether to resolve in Tier 1 or trigger **Escalation** to Tier 2.
3. **Tier 2** (a single selected specialist backbone: `Tier2EfficientNet` or `Tier2ArkPlus`) executes BatchNorm-safe MC Dropout + TTA with Law-of-Total-Variance **Uncertainty Decomposition**.
4. **Conformal Prediction** (`ConformalPredictor`, calibrated on temperature-scaled **Calibration Split** probabilities) builds the prediction set; high variance or a multi-class set triggers **Human Deferral** for radiologist review.
