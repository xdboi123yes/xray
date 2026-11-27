# 05: Fix disjoint temperature-before-conformal calibration order and end-to-end provenance verification

**What to build:** Reorder the production calibration pipeline so post-hoc temperature scaling fits strictly on the Validation Split before split-conformal quantile calibration runs on the strictly disjoint Calibration Split using temperature-scaled probabilities; verify all Track B evaluation, uncertainty decomposition, statistical comparison, and provenance-manifest gates align cleanly with the deepened package; prove the complete causal scientific contract via the CPU end-to-end research pipeline test.

**Blocked by:** 02: Deepen and simplify research package architecture

**Status:** resolved

- [x] Post-hoc temperature scaling runs before split-conformal calibration in the production notebook and fits exclusively on the Validation Split.
- [x] Split-conformal quantile estimation consumes the Validation Split temperature scalar and computes non-conformity scores exclusively on the strictly disjoint Calibration Split without sample reuse.
- [x] Conformal calibration and evaluation log marginal coverage, class-conditional coverage, singleton-set rate, two-class abstention-set rate, and empty-fallback rate.
- [x] The CPU end-to-end research pipeline test verifies the full causal chain: Patient-Disjoint Partition (zero overlap) -> mocked Synthetic Augmentation Batch -> Tier 1 and Tier 2 training -> Validation Split temperature scaling -> temperature-scaled Calibration Split conformal quantile -> TCR-P inference (Escalation and Human Deferral) -> provenance manifest verification.
- [x] All calibration, uncertainty decomposition, routing, order-sensitivity, statistical, and provenance unit/integration tests pass green.

## Comments

- Added `ConformalEvaluationReport`, `ConformalPredictor.evaluate_predictions`, and `ConformalPredictor.evaluate` in `src/tcrp/uncertainty/conformal.py` with temperature-scalar persistence, double-scaling and finite-positive guards, and structured logging of marginal coverage, class-conditional coverage, singleton-set rate, two-class abstention-set rate, and empty-fallback rate.
- Added `src/tcrp/evaluation/provenance.py` (`compute_file_sha256`, `build_provenance_manifest`, `verify_provenance_manifest`) verifying SHA-256 artifact integrity, four-way split zero-overlap manifests, and strict `val -> calibration` disjoint calibration metadata.
- Added `temperature` support to `BaseClassifier.mc_tta_forward_decomposed` / `mc_tta_forward` and `TieredSystem`, aligned `scripts/evaluate_ablation.py`, `scripts/evaluate_tiered.py` (with per-backbone calibration artifact scoping and temperature parity checks), and `scripts/generate_report_figures.py`.
- Reordered `notebooks/xray_colab_produce_all.ipynb` so Step 9 fits temperature scaling strictly on `data/processed/val.csv` before Step 10 runs split-conformal calibration on disjoint `data/processed/calibration.csv` with `verify_patient_disjointness`, prevented Step 14 from clobbering `val_predictions.csv`, and wired Step 18.5 integrity gate to `build_provenance_manifest` and `verify_provenance_manifest`.
- Upgraded `tests/e2e/test_full_pipeline.py` and added `tests/unit/test_conformal_and_provenance.py`; verified 120 passed tests, ruff, mypy, import-linter, and English-only comment checks.
