# Reproducibility

## One-notebook production (Colab)

`notebooks/xray_colab_produce_all.ipynb` reproduces every downstream artifact
from the trained weights. It is driven by an idempotent, never-throwing engine:
each step declares its primary output marker, and a global `SKIP_EXISTING`
switch means a step is skipped when its output already exists, blocked when its
inputs are missing, and only re-run when forced. Datasets are downloaded
conditionally — NIH only when a step that consumes it will actually run, and
CheXpert only when the A14 zero-shot evaluation will run.

The notebook produces, in order:

1. Base model training (Tier 1, Tier 2 EfficientNet, Tier 2 Ark+) — skipped if
   weights are restored from Drive.
2. Ablation training runs A8, A9, A11, A12, A15.
3. Conformal and temperature calibration.
4. Evaluation of the full A1-A15 table into `outputs/results/<run_name>.json`
   (A13/A14 via their dedicated scripts, the rest via `evaluate_ablation.py`).
5. `ablation.json` compilation, statistical tests, per-image predictions,
   thesis figures, the analysis notebooks, ONNX export, and the latency/carbon
   benchmark.

Outputs are mirrored to a single Google Drive tree so a later session restores
them and skips completed work.

## Evaluating the ablation table

```bash
# One row at a time
python scripts/evaluate_ablation.py --ablation A6

# Every non-external row (A1-A12, A15)
python scripts/evaluate_ablation.py --all

# Then compile the dashboard table
python scripts/build_ablation_json.py
```

These require the NIH test split and the trained weights, so they run on Colab
(GPU + data), not on a typical local machine.

## Continuous integration

Two GitHub Actions workflows gate every push and pull request:

- **CI Quality Gate** (`.github/workflows/ci.yml`) — ruff, mypy, `import-linter`,
  a code-comment language check, EN/TR i18n key parity, the pytest suite with a
  coverage floor, a print-leak guard, and the frontend production build.
- **UI Language Compliance** (`.github/workflows/ui-language.yml`) — builds the
  frontend and scans the bundle for stray Turkish characters outside the
  translation chunks.
- **Docker Images** (`.github/workflows/docker.yml`) — builds the API and
  frontend images so the multi-stage Dockerfiles stay valid.
- **Documentation** (`.github/workflows/docs.yml`) — builds this site.

To reproduce the CI checks faithfully, match the pinned tool versions
(Python 3.10, `mypy==1.10`, `ruff==0.4.5`); "passes locally" only equals
"passes on CI" when the versions match.

## Local CI-faithful environment

Create an isolated environment that matches the CI tool versions, then run the
exact gates CI runs:

```bash
conda create -n xray-ci python=3.10 -y
conda activate xray-ci
pip install -r requirements.txt -r requirements-dev.txt
pip install "mypy==1.10.0" "ruff==0.4.5"

# The full gate (mirrors .github/workflows/ci.yml)
ruff check core application infrastructure web tests scripts
mypy core application infrastructure
lint-imports
python scripts/check_comment_language.py
python scripts/check_i18n_parity.py
pytest tests/ --cov=core --cov=application --cov-fail-under=72

# Frontend production build
cd web/frontend && npm ci && npm run build
```

## Generating the thesis tables and figures

The result tables and figures are regenerated from the genuine evaluation
artifacts (no GPU required — they read `outputs/results/`):

```bash
python scripts/build_thesis_tables.py     # -> thesis/tables/*.tex
python scripts/build_thesis_figures.py    # -> thesis/figures/*.png
```

Re-run both after any new evaluation so the thesis numbers track
`outputs/results/ablation.json` exactly.
