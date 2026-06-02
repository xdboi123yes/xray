# Architecture

The codebase follows a layered architecture with a strictly enforced dependency
direction. The boundary is verified in CI by `import-linter` (see
`pyproject.toml`), so violations fail the build.

```
web/  ->  application/  ->  core/        (allowed)
infrastructure/  ->  core/               (allowed)
core/  -/->  application/  -/->  web/     (forbidden)
```

## Layers

| Layer | Path | Responsibility |
|-------|------|----------------|
| Domain | `core/` | Pure business logic: model interfaces, the tiered system, routing strategies, uncertainty (MC dropout, TTA, conformal, calibration), evaluation metrics, explainability. No I/O. |
| Infrastructure | `infrastructure/` | Data access (NIH / CheXpert repositories and datasets), the training loop and observers, checkpoint/registry persistence, ONNX export. |
| Application | `application/` | Use-case services (inference, training, evaluation, calibration, synthetic), DTOs, and the ablation orchestration. |
| Presentation | `web/` | FastAPI backend (REST + WebSocket) and the React/TypeScript frontend. |

## Key design patterns

- **Factory** — `core/models/factory.py` instantiates Tier 1 / Tier 2 backbones
  from a string key via a self-registration decorator, so model selection is
  data-driven (config), not hardcoded.
- **Strategy** — routing (`core/routing/`) swaps static vs dynamic escalation
  thresholds behind a common interface.
- **Observer** — the trainer (`infrastructure/training/`) emits lifecycle events
  to pluggable observers (MLflow, checkpointing, early stopping, carbon tracking).
- **Repository** — datasets are accessed through repository interfaces so NIH,
  CheXpert, and mixed (NIH + synthetic) sources are interchangeable.

## The tiered inference path

1. Tier 1 (MobileNetV2) produces a fast prediction and confidence.
2. If confidence is below the routing threshold, the case escalates to Tier 2.
3. Tier 2 runs Monte Carlo dropout + test-time augmentation to produce a
   calibrated probability and an epistemic-variance estimate.
4. A conformal predictor maps the Tier 2 distribution to a prediction set; cases
   with high variance or ambiguous sets are flagged for human review.
