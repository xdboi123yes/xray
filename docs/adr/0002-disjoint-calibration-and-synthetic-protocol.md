# Enforce strict patient-disjoint splits, disjoint calibration order, and leak-free synthetic regeneration

Status: accepted

For the Track B MDPI peer-review rerun, all splits (train, validation, calibration, test) are rebuilt with zero pairwise patient overlap, temperature scaling ($T_{\text{cal}}$) is fitted exclusively on `val.csv` before split-conformal quantile ($\hat{q}$) estimation on disjoint `calibration.csv` using temperature-scaled probabilities, and synthetic pneumothorax radiographs are regenerated via multi-batch FID-gated Stable Diffusion strictly from the new patient-disjoint `train.csv` before Tier-2 training.

## Considered Options

- Reuse existing Drive checkpoints and synthetic images (fast, but carries legacy cross-split patient leakage).
- Fit temperature scaling and conformal quantiles on the same calibration split (violates split-conformal exchangeability).
- Rebuild everything strictly patient-disjoint in the exact causal order: splits → multi-batch synthetics → model training → validation temperature scaling → calibration conformal quantile → evaluation & provenance manifest.

## Consequences

- Requires full Colab Pro L4 GPU execution (budgeted within 400 compute units), including multi-hour multi-batch synthetic generation with checkpoint/resume support.
- Directly resolves Reviewer 2 Comments 2, 4, 5, 7, 8, 9, 10, 11, 13, and 14 with auditable SHA-256 provenance.
- Legacy notebook bugs (Cell 9 `q_hat` running before Cell 10 `temperature.pt`, `generate_synthetics.py` hardcoded `batch_size=5`, and `use_synthetic=False` during Tier-2 training) must be fixed before launching the L4 run.
