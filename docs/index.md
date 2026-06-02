# Chest X-Ray Tiered Classification

Technical documentation for the bachelor thesis project: a **confidence-routed,
uncertainty-aware tiered classifier** for pneumothorax detection on chest
radiographs.

The system cascades a lightweight screening model (Tier 1, MobileNetV2) into a
deep specialist (Tier 2, EfficientNet-B4 or Ark+/Swin) only when Tier 1 is not
confident enough, and layers Monte Carlo dropout, test-time augmentation and
conformal prediction on top for calibrated, flag-for-review uncertainty.

- **Training data:** NIH ChestX-ray14
- **Out-of-domain test:** CheXpert (zero-shot)
- **Tracking:** MLflow
- **Serving:** FastAPI backend + React/TypeScript frontend

## Documentation map

| Page | What it covers |
|------|----------------|
| [Architecture](architecture.md) | The layered (core / application / infrastructure / web) design and dependency rules. |
| [Ablation Studies](ablations.md) | The full A1-A15 ablation matrix, how genuine metrics are produced, and the no-fabrication policy. |
| [Reproducibility](reproducibility.md) | End-to-end reproduction on Colab, the CI quality gate, and the evaluation scripts. |

!!! note "Clinical disclaimer"
    This project is a research and educational artifact. It is **not** a medical
    device and must not be used for clinical diagnosis.
