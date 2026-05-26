# Changelog

All notable changes to this project will be documented in this file.

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
