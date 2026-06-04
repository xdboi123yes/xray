# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Genuine, reproducible thesis tables generated from the evaluation artifacts
  (`scripts/build_thesis_tables.py`): the A1–A15 ablation table, the
  EfficientNet-B4 vs Ark+ Swin comparison (bootstrap CIs + DeLong/McNemar),
  the NIH dataset-statistics table and the hyperparameter table.
- Genuine thesis figures (`scripts/build_thesis_figures.py`): ablation overview
  (AUC/ECE across A1–A15) and NIH-vs-CheXpert cross-dataset generalization.
- Calibration metrics in `core/evaluation/metrics.py`: `brier_score` and
  `calibration_slope_intercept` (Newton/IRLS), with unit tests, wired into the
  ablation evaluator.
- Expanded the thesis bibliography from 5 to 45 references and added a drawing
  spec for the author-drawn diagrams (`thesis/figures/SPEC.md`).
- A genuine results-highlight block in the README.

### Changed
- The whole project and thesis are now English-only (notebooks, PLAN.md, thesis
  scaffold); the frontend stays bilingual (EN/TR).
- Replaced fabricated placeholder numbers in the thesis chapters (threshold
  table, conformal coverage, per-tier hyperparameters, framework versions) with
  genuine values from the evaluation artifacts and configuration.
- Marked the A14 CheXpert zero-shot result as a preliminary small-sample probe,
  pending a larger-cohort re-run.

### Removed
- Untracked the 205 committed `experiments/mlruns/` MLflow log files and removed
  stray coverage artifacts (the directory was already gitignored).

## [2.0.0] - 2026-05-25

### Added
- Modular client-side router with 5 distinct pages (`Inference`, `Dashboard`, `History`, `Ablation`, `About`) using react-router-dom `HashRouter`.
- Interactive `<GradCAMViewer />` supporting overlay opacity range controls, visibility toggle, and high-fidelity side-by-side diagnostic comparisons.
- High-fidelity print-ready bilingual A4 clinical PDF diagnostic report exporter via `jsPDF` and `html2canvas` integrated into results.
- Dynamically synchronized operating threshold slider with debounced FastAPI state synchronization.
- Pydantic P0/P1 type-safe DTO and clinical application service layers for model training, evaluation, and calibration.
- Pydantic-settings configuration loading with hierarchical DEV/PROD overrides.
- Automated CI workflows for linting, type-checks, import contracts, and UI language safety.

### Changed
- Replaced monolithic `ResultCard` layout with modular, dedicated sub-components.
- Transitioned frontend dependencies to concrete, standard, released packages.
- Restructured `web/backend` endpoints into modular sub-routers (`/ablation`, `/history`, `/models`, etc.).

### Fixed
- Fixed event loop blockages in WebSockets streaming by adopting `asyncio.sleep` over `time.sleep`.
- Cleaned up cross-layer import boundary violations (0 legacy `src/` leaks in core layers).
- Patched insecure wildcard CORS configurations with environment config variables.
