# Ablation Studies (A1-A15)

The thesis evaluates a 15-row ablation matrix that isolates the contribution of
each component: tiered routing, the dynamic threshold, MC dropout, TTA,
conformal prediction, synthetic / classical augmentation, the Ark+ backbone, and
out-of-domain generalization.

The canonical definition of every row lives in one place —
`scripts/ablation_spec.py` — and is shared by both the evaluator and the table
compiler, so the dashboard metadata and the evaluation configuration can never
drift apart.

| ID | Configuration | Tier 1 | Tier 2 | Uncertainty |
|----|---------------|--------|--------|-------------|
| A1 | Tier 1 only | MobileNetV2 | — | None |
| A2 | Tier 2 only (EfficientNet) | — | EfficientNet-B4 | None |
| A3 | Tiered, static threshold | MobileNetV2 | EfficientNet-B4 | None |
| A4 | Tiered, dynamic threshold | MobileNetV2 | EfficientNet-B4 | None |
| A5 | Tiered + MC dropout | MobileNetV2 | EfficientNet-B4 | MC |
| A6 | Tiered + MC + TTA | MobileNetV2 | EfficientNet-B4 | MC + TTA |
| A7 | Tiered + Conformal | MobileNetV2 | EfficientNet-B4 | MC + TTA + Conformal |
| A8 | Without synthetic augmentation | MobileNetV2 | EfficientNet-B4 | MC + TTA |
| A9 | Without any augmentation | MobileNetV2 | EfficientNet-B4 | MC + TTA |
| A10 | Always Tier 2 (no routing) | — | EfficientNet-B4 | MC + TTA |
| A11 | Tier 2 = Ark+ (no MC/TTA) | — | Ark+ Swin | None |
| A12 | Tier 2 = Ark+ + MC + TTA | — | Ark+ Swin | MC + TTA |
| A13 | Proposed tiered + Ark+ | MobileNetV2 | Ark+ Swin | MC + TTA + Conformal |
| A14 | Zero-shot CheXpert | MobileNetV2 | Ark+ Swin | MC + TTA + Conformal |
| A15 | A13 + Mixup/CutMix | MobileNetV2 | Ark+ Swin | MC + TTA + Conformal |

## No fabricated numbers

Every metric in the table is **computed**, never invented. The pipeline is:

1. `scripts/evaluate_ablation.py --ablation <ID>` loads the row's *real* trained
   weights, runs inference on the NIH test split (A1-A12, A15), computes
   `auc_roc`, `accuracy`, `ece` and the full confusion-derived set, and writes
   `outputs/results/<run_name>.json`. A13 and A14 are owned by their dedicated
   scripts (`evaluate_tiered.py`, `evaluate_chexpert.py`).
2. `scripts/build_ablation_json.py` compiles `outputs/results/ablation.json` by
   reading those per-row markers. A row is marked `provenance: evaluation_json`
   only when both `auc_roc` and `accuracy` are present as real numbers;
   otherwise it stays an honest `preliminary_placeholder` with null metrics.
3. The dashboard renders placeholder rows with an em-dash and a "Preliminary"
   badge — it never shows a number that was not measured.

This honesty is enforced by tests (`tests/unit/test_ablation_spec.py`,
`tests/unit/test_ablation.py`) that assert genuine rows carry numeric metrics
and placeholder rows stay null.
