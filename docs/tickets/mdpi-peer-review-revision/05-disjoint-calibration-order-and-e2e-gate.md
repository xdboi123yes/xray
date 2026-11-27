# 05: Fix disjoint temperature-before-conformal calibration order and end-to-end provenance verification

**What to build:** Reorder the production calibration pipeline so post-hoc temperature scaling fits strictly on the Validation Split before split-conformal quantile calibration runs on the strictly disjoint Calibration Split using temperature-scaled probabilities; verify all Track B evaluation, uncertainty decomposition, statistical comparison, and provenance-manifest gates align cleanly with the deepened package; prove the complete causal scientific contract via the CPU end-to-end research pipeline test.

**Blocked by:** 02: Deepen and simplify research package architecture

**Status:** open

- [ ] Post-hoc temperature scaling runs before split-conformal calibration in the production notebook and fits exclusively on the Validation Split.
- [ ] Split-conformal quantile estimation consumes the Validation Split temperature scalar and computes non-conformity scores exclusively on the strictly disjoint Calibration Split without sample reuse.
- [ ] Conformal calibration and evaluation log marginal coverage, class-conditional coverage, singleton-set rate, two-class abstention-set rate, and empty-fallback rate.
- [ ] The CPU end-to-end research pipeline test verifies the full causal chain: Patient-Disjoint Partition (zero overlap) -> mocked Synthetic Augmentation Batch -> Tier 1 and Tier 2 training -> Validation Split temperature scaling -> temperature-scaled Calibration Split conformal quantile -> TCR-P inference (Escalation and Human Deferral) -> provenance manifest verification.
- [ ] All calibration, uncertainty decomposition, routing, order-sensitivity, statistical, and provenance unit/integration tests pass green.
