# 02: Deepen and simplify research package architecture

**What to build:** Conduct an architecture review on the pruned research codebase to identify shallow pass-through wrappers across application, infrastructure, and core layers, apply the user-approved deepening refactors into a single minimal, cohesive Python package with narrow interfaces, replace the legacy web-coupled end-to-end test with a pure research pipeline end-to-end test, and verify the entire test suite, coverage threshold, linter, and type checker stay green.

**Blocked by:** 01: Clone repository, archive full-stack application, and prune to research-only layout

**Status:** open

- [ ] An architectural review report identifies shallow pass-through modules and deepening candidates using canonical domain glossary and deep-module terminology.
- [ ] User-approved deepening refactors consolidate the research code into a minimal, well-structured Python package with clear, narrow interfaces.
- [ ] All experiment scripts, production notebook hooks, and unit tests import cleanly from the simplified package structure without circular or broken imports.
- [ ] The legacy web-coupled E2E test is replaced by a CPU-runnable end-to-end research pipeline test covering splitting, training, calibration, and TCR-P inference.
- [ ] Full unit and integration tests pass at or above the project coverage threshold alongside green lint and static type checks.
