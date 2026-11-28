# 02: Deepen and simplify research package architecture

**What to build:** Conduct an architecture review on the pruned research codebase to identify shallow pass-through wrappers across application, infrastructure, and core layers, apply the user-approved deepening refactors into a single minimal, cohesive Python package with narrow interfaces, replace the legacy web-coupled end-to-end test with a pure research pipeline end-to-end test, and verify the entire test suite, coverage threshold, linter, and type checker stay green.

**Blocked by:** 01: Clone repository, archive full-stack application, and prune to research-only layout

**Status:** resolved

- [x] An architectural review report identifies shallow pass-through modules and deepening candidates using canonical domain glossary and deep-module terminology.
- [x] User-approved deepening refactors consolidate the research code into a minimal, well-structured Python package with clear, narrow interfaces.
- [x] All experiment scripts, production notebook hooks, and unit tests import cleanly from the simplified package structure without circular or broken imports.
- [x] The legacy web-coupled E2E test is replaced by a CPU-runnable end-to-end research pipeline test covering splitting, training, calibration, and TCR-P inference.
- [x] Full unit and integration tests pass at or above the project coverage threshold alongside green lint and static type checks.

## Comments

- Consolidated 62 modules across `core/`, `application/`, `infrastructure/`, and `config/settings.py` into 18 cohesive modules under `src/tcrp/` (per ADR-0001).
- Removed 11 dead web/legacy modules (all preserved in `/Users/alperen/Developer/xray-app`), unified BatchNorm-safe MC Dropout / TTA / Law-of-Total-Variance Uncertainty Decomposition inside `tcrp.models.base.BaseClassifier`, delegated `TieredSystem` routing to `tcrp.routing`, extracted `tcrp.data.splits.build_patient_disjoint_splits`, and unified synthetic generation + FID gating behind injectable adapters in `tcrp.data.synthetic.SyntheticBatchPipeline`.
- Verified: 95/95 tests passing on CPU (82.08% coverage, gate 72%), `mypy src/tcrp/` (0 errors, 25 files), `ruff check`, `lint-imports` (2/2 contracts kept), and `check_comment_language.py`.
