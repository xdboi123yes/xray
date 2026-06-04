# Chest X-Ray Tiered Classification — Master Refactor & Extension Plan
**Bachelor Thesis Project · "1 Decembrie 1918" University of Alba Iulia**
**Target delivery:** ~10 weeks (early Q3 2026) · **Current date:** 2026-05-24

---

## 0. What This Document Is, How to Read It

This plan was prepared from an **actual** scan of the project and is an **action-oriented** master plan. It contains:

1. **Language policy** — project/thesis and code-commenting rules (Section 1)
2. **Executive summary** — current-state inventory and the exit goal (Section 2)
3. **Restructured architecture** — layer-by-layer folder structure and dependency direction (Section 3)
4. **Design pattern details** — code-level example signatures (Section 4)
5. **Ark+ integration** — strategy, implementation, ablation (Section 5)
6. **ML pipeline improvements** — trainer, synthetic data, statistics, fairness (Section 6)
7. **Web application** — FastAPI contracts, React component hierarchy (Section 7)
8. **DevOps & reproducibility** — Docker, Makefile, CI, monitoring (Section 8)
9. **Test strategy** — pyramid, critical scenarios, fixtures (Section 9)
10. **Thesis writing plan** — chapter by chapter, figure sources (Section 10)
11. **10-week roadmap** — with milestones (Section 11)
12. **Risk register** and decision-log template (Section 12)
13. **Cowork prompt** — copy-paste into a new session (Section 13)
14. **Open questions** and **DoD** — minimum acceptance criteria (Sections 14-15)
15. **Appendices** — dependencies, pyproject example (Section 16+)

This plan file is **live**: `[x]` marks are added as phases complete, and decisions are appended to the `Decision Log` section.

---

## 1. Language Policy (Hard Constraint)

> **Update (2026-06-03):** This project originally mandated a bilingual split (Turkish for the thesis / README / UI / notebook prose, English for code). That policy was **reversed**: the entire project and thesis are now **English-only**. The section below reflects the current policy.

This rule is **binding** for every file and every written output in the project.

### 1.1 English (everything)
- **The entire thesis** — LaTeX sources (`thesis/chapters/*.tex`), abstract, titles, figure captions, table headers, bibliography.
- **README.md**, **CHANGELOG.md**, `PLAN.md`, and all documentation.
- **Notebook markdown cells** and all notebook text/markdown blocks.
- **Code comments (`#`, `//`, `/* */`)**, **docstrings** (Google-style), **type hints**, and TODO/FIXME comments.
- **Logging messages**, **exception messages**, and **API error messages**.
- **Variable / function / class / file / folder names**, MLflow run/parameter/metric names, git commit messages, test names, OpenAPI descriptions, and `pyproject.toml` / Dockerfile / Makefile comments.
- **All user-facing frontend strings** — the English locale (`web/frontend/src/i18n/en.json`) is the default.

### 1.2 The only allowed Turkish (intentional)
- **`web/frontend/src/i18n/tr.json`** — the optional Turkish UI translation behind the bilingual EN/TR toggle. English is the default/fallback (`i18n/index.ts` `fallbackLng:'en'`).
- The language-switch labels in `en.json` (`"Türkçe"`, `"Türkçe (TR)"`).
- The Turkish **detectors / fixtures**: `scripts/check_comment_language.py`, `.github/workflows/ui-language.yml`, `tests/unit/test_ci_gates_probe.py`.

### 1.3 Verification
- A pre-commit / CI check (`scripts/check_comment_language.py`) fails on Turkish characters (`ç ğ ı ö ş ü`) in Python/TS comments and docstrings.
- The `ui-language.yml` workflow scans the built frontend bundle for stray Turkish outside the `tr` locale chunk.
- The `check_i18n_parity.py` check keeps the EN and TR locales key-aligned.

---

## 2. Executive Summary

### 2.1 Thesis in One Sentence
*A clinically-aware pneumothorax detection system using confidence-routed tiered inference (MobileNetV2 → EfficientNetB4/Ark+) + MC-Dropout + Test-Time Augmentation + Conformal Prediction + Stable-Diffusion synthetic-data enrichment, trained on NIH ChestX-ray14 and tested zero-shot on CheXpert.*

### 2.2 Current State (Factual)

> Historical starting-state snapshot from the original scan; most ❌ items below have since been implemented.
| Component | Status | Note |
|---|---|---|
| Tier 1 (MobileNetV2) | ✅ Trained | `outputs/models/best_tier1_mobilenet.pth` · optimal threshold 0.75 |
| Tier 2 (EfficientNetB4) | ✅ Trained | `outputs/models/best_tier2_efficientnet.pth` · MC + TTA implemented |
| TieredSystem routing | ✅ Working | Static + dynamic threshold |
| Conformal Prediction | ✅ Calibrated | `q_hat.pt` present |
| Temperature Scaling | ✅ Present | `temperature.pt` |
| MLflow tracking | ✅ Present | `experiments/mlruns/` |
| Gradio demo | ✅ Working | `demo/app.py` |
| Plotly threshold dashboard | ✅ Present | `dashboard/` |
| Docker multi-stage | ✅ Present | builder + runtime stages |
| docker-compose (5 services) | ✅ Present | demo, dashboard, evaluate, figures, mlflow profiles |
| Stable Diffusion synthetic data | 🟡 Code present, generation missing | `src/data/synthetic_gen.py` |
| FID evaluation | 🟡 Code present | `src/data/fid_eval.py` |
| CheXpert cross-dataset | ❌ Missing | No code |
| Ark+ integration | ❌ Missing | New addition |
| FastAPI + React | ❌ Missing | Only Gradio exists |
| Test suite | ❌ Missing | No `tests/` folder |
| Type hints + docstring | ❌ Missing | Nowhere |
| Abstract base classes | ❌ Missing | Implicit interface |
| Factory/Strategy/Observer patterns | ❌ Missing | Hardcoded imports |
| Mixed precision + checkpoint resume | ❌ Missing | Primitive trainer |
| Statistical testing (DeLong/bootstrap) | ❌ Missing | Critical for the thesis |
| ONNX export + quantization | ❌ Missing | Needed for the mobile claim |
| CI/CD (GitHub Actions) | ❌ Missing | For reproducibility |
| Subgroup fairness analysis | ❌ Missing | Modern medical-AI requirement |
| Decision curve analysis | ❌ Missing | Clinical utility metric |
| DICOM input support | ❌ Missing | Clinical data format |
| Carbon footprint (CodeCarbon) | ❌ Missing | Tier1 economic evidence |
| MLflow Model Registry | ❌ Missing | For A/B comparison |
| Failure case taxonomy notebook | ❌ Missing | Critical thesis figure |

### 2.3 Exit Goal (Definition of Done)

The package put in front of the jury at the thesis defense:
1. **Code:** Layered architecture, 95%+ type-hint+docstring coverage, 80%+ test coverage for core/, active pre-commit hooks.
2. **Models:** Tier1 MobileNet, Tier2 EfficientNetB4 (existing weights preserved), Tier2 Ark+ (new), all ONNX-exportable.
3. **Evaluation:** A1-A15 ablation table, NIH + CheXpert cross-dataset metrics, DeLong tests, bootstrap CIs.
4. **Web:** FastAPI backend (OpenAPI-documented) + React/TS frontend (Inference + Dashboard + History + About), brought up with a single command in Docker.
5. **DevOps:** Multi-stage Docker (CPU + GPU variants), docker-compose dev/prod, GitHub Actions (lint + test + build), MLflow Model Registry.
6. **Thesis:** ~80 pages of English LaTeX, 18-25 figures, all reproducible via `scripts/generate_*.py`.
7. **Demo:** Live demo video, deployed to gradio.app or HuggingFace Spaces.

---

## 3. Restructured Architecture

### 3.1 Folder Structure (Complete)

```
xray/                              # Repo root
├── core/                          # Domain layer (pure business logic)
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── base_model.py          # BaseClassifier(ABC)
│   │   ├── base_augmentation.py   # BaseAugmentation(ABC)
│   │   ├── base_router.py         # BaseRouter(ABC)
│   │   ├── base_repository.py     # BaseRepository(ABC)
│   │   └── base_observer.py       # TrainingObserver(ABC)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py             # ModelFactory
│   │   ├── tier1_mobilenet.py     # Tier1MobileNet(BaseClassifier)
│   │   ├── tier2_efficientnet.py  # Tier2EfficientNet(BaseClassifier)
│   │   ├── tier2_ark.py           # Tier2ArkPlus(BaseClassifier)  [NEW]
│   │   └── tiered_system.py       # TieredSystem(BaseClassifier)
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── static_router.py       # StaticThresholdRouter
│   │   ├── dynamic_router.py      # DynamicThresholdRouter
│   │   └── learned_router.py      # LearnedRouter [STRETCH]
│   ├── augmentation/
│   │   ├── __init__.py
│   │   ├── classical.py           # ClassicalAugmentation
│   │   ├── diffusion.py           # DiffusionAugmentation
│   │   ├── mixup_cutmix.py        # MixupCutmixAugmentation [NEW]
│   │   └── composer.py            # AugmentationPipeline (Composite)
│   ├── uncertainty/
│   │   ├── __init__.py
│   │   ├── mc_dropout.py          # MCDropout
│   │   ├── tta.py                 # TestTimeAugmentation
│   │   ├── conformal.py           # ConformalPredictor
│   │   └── calibration.py         # TemperatureScaling, PlattScaling
│   └── explainability/
│       ├── __init__.py
│       ├── gradcam.py             # GradCAM
│       ├── gradcam_pp.py          # GradCAM++ [NEW]
│       └── tier_comparison.py     # TierComparisonViewer
│
├── infrastructure/                # Infra layer (DB, file, external services)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── repository.py          # NIHRepository, CheXpertRepository
│   │   ├── dataset.py             # NIHChestXrayDataset (refactored)
│   │   ├── chexpert_dataset.py    # CheXpertDataset [NEW]
│   │   ├── dicom_loader.py        # DICOM -> PIL loader [NEW]
│   │   ├── preprocessing.py       # Lung CLAHE, histogram match
│   │   └── synthetic_gen.py       # StableDiffusionGenerator (refactored)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Trainer (Observer-aware)
│   │   ├── losses.py              # FocalLoss, LabelSmoothing
│   │   ├── scheduler.py           # CosineWarmup, ReduceLROnPlateau
│   │   ├── amp_helper.py          # MixedPrecisionContext [NEW]
│   │   └── observers/
│   │       ├── __init__.py
│   │       ├── mlflow_observer.py
│   │       ├── checkpoint_observer.py
│   │       ├── early_stopping.py
│   │       ├── lr_logger.py
│   │       └── carbon_tracker.py  # CodeCarbon integration [NEW]
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── checkpoint.py          # CheckpointManager (resume support)
│   │   ├── model_registry.py      # MLflow Model Registry wrapper
│   │   └── prediction_log.py      # SQLite prediction history
│   └── export/
│       ├── __init__.py
│       ├── onnx_exporter.py       # ONNX + INT8 quantization [NEW]
│       └── torchscript_exporter.py
│
├── application/                   # Application layer (use cases)
│   ├── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── inference_service.py   # InferenceService (lazy load, thread-safe)
│   │   ├── training_service.py    # TrainingService
│   │   ├── evaluation_service.py  # EvaluationService
│   │   ├── calibration_service.py # CalibrationService
│   │   └── synthetic_service.py   # SyntheticDataService
│   ├── dto/
│   │   ├── __init__.py
│   │   ├── prediction_dto.py      # @dataclass PredictionDTO
│   │   ├── metrics_dto.py
│   │   ├── ablation_dto.py
│   │   └── training_config_dto.py
│   └── orchestration/
│       ├── __init__.py
│       └── ablation_runner.py     # Orchestrates A1-A15 experiments
│
├── web/                           # Presentation layer
│   ├── backend/                   # FastAPI
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI(...)
│   │   ├── deps.py                # Dependency injection
│   │   ├── settings.py            # pydantic-settings
│   │   ├── routes/
│   │   │   ├── inference.py       # POST /predict, /predict/batch, WS /predict
│   │   │   ├── evaluation.py      # GET /metrics, /ablation
│   │   │   ├── threshold.py       # GET/PUT /threshold
│   │   │   ├── models.py          # GET /models, POST /models/load
│   │   │   ├── history.py         # GET /history, DELETE /history/{id}
│   │   │   ├── health.py          # GET /health
│   │   │   └── synthetic.py       # POST /synthetic/generate (admin)
│   │   ├── middleware/
│   │   │   ├── error_handler.py
│   │   │   ├── logging.py         # structlog
│   │   │   ├── rate_limit.py      # slowapi
│   │   │   └── cors.py
│   │   ├── schemas/               # Pydantic request/response models
│   │   │   ├── prediction.py
│   │   │   ├── threshold.py
│   │   │   └── error.py
│   │   └── ws/
│   │       └── inference_ws.py    # WebSocket streaming
│   │
│   └── frontend/                  # React + Vite + TS
│       ├── package.json
│       ├── tsconfig.json
│       ├── tailwind.config.ts
│       ├── vite.config.ts
│       ├── index.html
│       ├── public/
│       │   └── logo.svg
│       └── src/
│           ├── main.tsx
│           ├── App.tsx
│           ├── router.tsx
│           ├── locale/
│           │   ├── tr.ts          # Turkish UI strings (default)
│           │   └── en.ts          # English (stretch)
│           ├── api/
│           │   ├── client.ts      # axios + react-query setup
│           │   ├── inference.ts
│           │   ├── threshold.ts
│           │   ├── history.ts
│           │   └── types.ts       # TS types matching DTOs
│           ├── store/             # Zustand
│           │   ├── inferenceStore.ts
│           │   ├── thresholdStore.ts
│           │   ├── historyStore.ts
│           │   └── themeStore.ts
│           ├── hooks/
│           │   ├── usePredict.ts
│           │   ├── useThreshold.ts
│           │   └── useDarkMode.ts
│           ├── components/
│           │   ├── layout/
│           │   │   ├── Navbar.tsx
│           │   │   ├── Sidebar.tsx
│           │   │   └── Footer.tsx
│           │   ├── upload/
│           │   │   ├── UploadZone.tsx
│           │   │   ├── DicomUpload.tsx
│           │   │   └── BatchUploader.tsx
│           │   ├── result/
│           │   │   ├── ResultCard.tsx
│           │   │   ├── PredictionBadge.tsx
│           │   │   ├── ConfidenceBar.tsx
│           │   │   ├── UncertaintyBar.tsx
│           │   │   ├── ConformalSet.tsx
│           │   │   ├── TierBadge.tsx
│           │   │   ├── FlaggedBanner.tsx
│           │   │   └── ExportMenu.tsx
│           │   ├── gradcam/
│           │   │   ├── GradCAMViewer.tsx
│           │   │   ├── HeatmapOverlay.tsx
│           │   │   └── TierCompare.tsx
│           │   ├── dashboard/
│           │   │   ├── ThresholdSlider.tsx
│           │   │   ├── MetricsChart.tsx (Recharts)
│           │   │   ├── LiveStats.tsx
│           │   │   └── AblationTable.tsx
│           │   └── common/
│           │       ├── ThemeToggle.tsx
│           │       ├── LoadingSkeleton.tsx
│           │       └── ProgressSteps.tsx
│           ├── pages/
│           │   ├── InferencePage.tsx       # /
│           │   ├── DashboardPage.tsx       # /dashboard
│           │   ├── HistoryPage.tsx         # /history
│           │   ├── AblationPage.tsx        # /ablation [NEW]
│           │   └── AboutPage.tsx           # /about
│           ├── styles/
│           │   └── globals.css
│           └── utils/
│               ├── format.ts
│               └── pdf-report.ts     # jsPDF report generator
│
├── tests/                         # Test suite
│   ├── conftest.py                # Pytest fixtures
│   ├── unit/
│   │   ├── test_model_factory.py
│   │   ├── test_routing.py
│   │   ├── test_metrics.py
│   │   ├── test_conformal.py
│   │   ├── test_calibration.py
│   │   ├── test_augmentation.py
│   │   ├── test_uncertainty.py
│   │   └── test_dto.py
│   ├── integration/
│   │   ├── test_inference_service.py
│   │   ├── test_training_service.py
│   │   ├── test_api_inference.py
│   │   ├── test_api_threshold.py
│   │   ├── test_websocket.py
│   │   └── test_repository.py
│   ├── e2e/
│   │   ├── test_full_pipeline.py
│   │   └── test_ablation_runner.py
│   └── fixtures/
│       ├── dummy_xray.png         # 224x224 synthetic
│       ├── dummy_dicom.dcm
│       ├── mini_tier1_weights.pth
│       └── mini_tier2_weights.pth
│
├── scripts/                       # CLI entrypoints
│   ├── train_tier1.py             # python -m scripts.train_tier1
│   ├── train_tier2.py             # --backbone {efficientnet_b4|ark_plus}
│   ├── preprocess.py
│   ├── generate_synthetics.py
│   ├── calibrate_conformal.py
│   ├── calibrate_temperature.py
│   ├── evaluate_tiered.py
│   ├── evaluate_chexpert.py       # [NEW] Cross-dataset eval
│   ├── run_ablation.py            # A1-A15 runner
│   ├── export_onnx.py             # [NEW]
│   ├── benchmark_latency.py       # [NEW] FPS, memory, energy
│   ├── statistical_tests.py       # [NEW] DeLong, bootstrap CI
│   ├── generate_report_figures.py # Thesis figures
│   └── deploy_huggingface.py      # [NEW]
│
├── notebooks/
│   ├── xray_colab_training.ipynb  # PRESERVED
│   ├── error_analysis.ipynb       # [NEW] Failure case taxonomy
│   ├── tier_disagreement.ipynb    # [NEW] T1 vs T2 disagreement
│   ├── subgroup_analysis.ipynb    # [NEW] Age/sex fairness
│   └── synthetic_quality.ipynb    # [NEW] FID + qualitative review
│
├── config/                        # Environment-aware config
│   ├── base.yaml                  # All defaults
│   ├── dev.yaml                   # Development override
│   ├── prod.yaml                  # Production override
│   ├── test.yaml                  # Test override (mini batches)
│   └── settings.py                # pydantic-settings Loader
│
├── docker/
│   ├── Dockerfile.api             # FastAPI runtime (CPU + optional GPU)
│   ├── Dockerfile.frontend        # nginx + built React
│   ├── Dockerfile.training        # CUDA + dev deps
│   ├── Dockerfile.synthetic       # Diffusers + GPU
│   └── nginx.conf
│
├── thesis/                        # LaTeX source (Turkish)
│   ├── main.tex
│   ├── chapters/
│   │   ├── 01_giris.tex
│   │   ├── 02_ilgili_calismalar.tex
│   │   ├── 03_metodoloji.tex
│   │   ├── 04_deneysel_kurulum.tex
│   │   ├── 05_sonuclar.tex
│   │   ├── 06_tartisma.tex
│   │   └── 07_sonuc.tex
│   ├── figures/                   # Auto-generated, do NOT commit edits
│   ├── tables/
│   ├── bibliography.bib
│   └── thesis.sty
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # lint + test on PR
│       ├── docker.yml             # build + push on tag
│       └── docs.yml               # Sphinx docs build
│
├── docker-compose.yml             # Dev
├── docker-compose.prod.yml        # Prod overrides
├── Makefile
├── pyproject.toml                 # ruff, mypy, pytest config
├── .pre-commit-config.yaml
├── .env.example
├── .gitignore
├── requirements.txt               # Runtime deps
├── requirements-dev.txt           # Dev/test deps
├── requirements-training.txt      # Heavy training deps (separate)
├── README.md                      # Turkish
├── CHANGELOG.md                   # Turkish
├── CONTRIBUTING.md
├── LICENSE
├── PLAN.md                        # This file (Turkish)
└── src/                           # LEGACY — not deleted, gradually shimmed
    └── (existing structure with import shims)
```

### 3.2 src/ Folder Strategy

Eski `src/` silinmez. Her dosya yerine, **import shim** koyulur:

```python
# src/models/tier1_mobilenet.py (legacy shim)
"""Backward-compatibility shim. Import from core.models instead."""
from core.models.tier1_mobilenet import Tier1MobileNet  # noqa: F401
import warnings

warnings.warn(
    "src.models.tier1_mobilenet is deprecated, use core.models.tier1_mobilenet",
    DeprecationWarning,
    stacklevel=2,
)
```

The notebook, old scripts, and the existing Gradio demo do not break. One version later (`v1.1.0`) it is removed entirely.

### 3.3 Dependency Direction (Architectural Discipline)

```
web/  ->  application/  ->  core/         (allowed)
infrastructure/  ->  core/                (allowed)
core/  ↛  application/  ↛  web/           (forbidden)
```

It is checked at compile time via the `import-linter` config in `pyproject.toml`.

---

## 4. Design Pattern Details (Code-Level)

### 4.1 Factory Pattern — ModelFactory

```python
# core/models/factory.py
from __future__ import annotations
from typing import Type, Any
from core.interfaces.base_model import BaseClassifier


class ModelFactory:
    """Registry-driven factory for tiered classifier instantiation.

    Why: Tier selection is data-driven (config.yaml), not code-driven.
    Avoids hardcoded if/elif chains and keeps `core/models/` extensible.

    Example:
        >>> model = ModelFactory.create("ark_plus", num_classes=2, pretrained=True)
        >>> isinstance(model, BaseClassifier)
        True
    """

    _registry: dict[str, Type[BaseClassifier]] = {}

    @classmethod
    def register(cls, key: str):
        def decorator(model_cls: Type[BaseClassifier]) -> Type[BaseClassifier]:
            if key in cls._registry:
                raise ValueError(f"Model '{key}' already registered")
            cls._registry[key] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, model_type: str, **kwargs: Any) -> BaseClassifier:
        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model_type='{model_type}'. "
                f"Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[model_type](**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls._registry.keys())
```

Models self-register via decorator:
```python
# core/models/tier1_mobilenet.py
@ModelFactory.register("mobilenet_v2")
class Tier1MobileNet(BaseClassifier): ...
```

### 4.2 Strategy Pattern — Routing

```python
# core/interfaces/base_router.py
from abc import ABC, abstractmethod


class BaseRouter(ABC):
    """Strategy interface for tier routing decisions."""

    @abstractmethod
    def route(self, confidence: float) -> int:
        """Return tier number (1 = fast, 2 = deep)."""

    @abstractmethod
    def update(self, confidence: float) -> None:
        """Optionally update internal state (rolling window, etc.)."""

    @property
    @abstractmethod
    def current_threshold(self) -> float: ...
```

Concrete: `StaticThresholdRouter`, `DynamicThresholdRouter`, optional `LearnedRouter` (small MLP, stretch goal).

### 4.3 Strategy + Composite — Augmentation

```python
# core/augmentation/composer.py
class AugmentationPipeline:
    """Composite of BaseAugmentation strategies; albumentations-compatible.

    Why: Ablations A8/A9 (no synthetic / no aug) require runtime pipeline
    rewiring. A hardcoded compose() does not allow this.
    """

    def __init__(self, augmentations: list[BaseAugmentation]):
        self._augs = augmentations

    def apply(self, image: np.ndarray) -> np.ndarray:
        for aug in self._augs:
            image = aug.apply(image)
        return image

    @classmethod
    def from_config(cls, cfg: dict) -> "AugmentationPipeline":
        # Reads cfg['augmentations'] = ["classical", "diffusion", "mixup"]
        ...
```

### 4.4 Observer Pattern — Training Events

```python
# core/interfaces/base_observer.py
class TrainingObserver(ABC):
    def on_train_start(self, trainer): pass
    def on_epoch_start(self, epoch, trainer): pass
    def on_epoch_end(self, epoch, metrics, trainer): pass
    def on_train_end(self, trainer): pass
    def on_validation_end(self, epoch, val_metrics, trainer): pass
```

Concrete observers: `MLflowObserver`, `CheckpointObserver`, `EarlyStoppingObserver`, `LRLoggerObserver`, `CarbonTrackerObserver` (codecarbon), `TQDMObserver`.

Usage:
```python
trainer = Trainer(model, optimizer, criterion, device, cfg)
trainer.add_observer(MLflowObserver(run_name="Tier2_ArkPlus"))
trainer.add_observer(CheckpointObserver("outputs/models/best_tier2_ark.pth"))
trainer.add_observer(EarlyStoppingObserver(patience=7, monitor="val_auc"))
trainer.add_observer(CarbonTrackerObserver())  # Energy figure for thesis
trainer.train(train_loader, val_loader)
```

### 4.5 Repository Pattern — Data Access

```python
# core/interfaces/base_repository.py
class BaseRepository(ABC):
    @abstractmethod
    def get_train_dataset(self) -> Dataset: ...
    @abstractmethod
    def get_val_dataset(self) -> Dataset: ...
    @abstractmethod
    def get_test_dataset(self) -> Dataset: ...
    @abstractmethod
    def get_calibration_dataset(self) -> Dataset: ...  # For conformal
```

Concrete: `NIHRepository`, `CheXpertRepository` (new), `MixedRepository` (NIH + synthetic).

### 4.6 Dependency Injection

`web/backend/deps.py`:
```python
@lru_cache  # singleton
def get_inference_service() -> InferenceService:
    settings = get_settings()
    model_factory = ModelFactory
    tier1 = model_factory.create("mobilenet_v2", ...)
    tier1.load_state_dict(torch.load(settings.tier1_weights_path))
    tier2 = model_factory.create(settings.tier2_backbone, ...)
    tier2.load_state_dict(torch.load(settings.tier2_weights_path))
    router = StaticThresholdRouter(threshold=settings.confidence_threshold)
    conformal = ConformalPredictor.load(settings.conformal_path)
    system = TieredSystem(tier1, tier2, router, conformal)
    return InferenceService(system)


@router.post("/predict")
async def predict(
    file: UploadFile,
    svc: InferenceService = Depends(get_inference_service),
):
    ...
```

---

## 5. Ark+ Integration (Detail)

### 5.1 Strategy
The Ark+ pretrained encoder is integrated as Tier 2's alternative backbone.

1. **Primary:** Download official Ark+ checkpoint from GitHub `JLiangLab/Ark` (Swin-Base, pretrained on multiple CXR datasets)
2. **Secondary:** Fallback to `swin_base_patch4_window7_224` ImageNet pretrained (via timm)
3. **Tertiary:** `swin_tiny_patch4_window7_224` (for CI and low-RAM dev environments)

### 5.2 Implementation Skeleton

```python
# core/models/tier2_ark.py
from typing import Literal
from pathlib import Path
import timm
import torch
import torch.nn as nn

from core.interfaces.base_model import BaseClassifier
from core.models.factory import ModelFactory

ArkVariant = Literal["base", "small", "tiny"]


@ModelFactory.register("ark_plus")
class Tier2ArkPlus(BaseClassifier):
    """Tier 2 deep model with Ark+ (or Swin fallback) backbone.

    Args:
        num_classes: Number of output classes.
        variant: Swin variant (default: 'base' matches Ark+ paper).
        pretrained: Try Ark+ -> ImageNet Swin -> random.
        freeze_epochs: Freeze backbone for N epochs (gradual unfreezing).
        ark_checkpoint_path: Local path to Ark+ encoder .pth (optional).
    """

    def __init__(
        self,
        num_classes: int = 2,
        variant: ArkVariant = "base",
        pretrained: bool = True,
        freeze_epochs: int = 5,
        ark_checkpoint_path: str | None = None,
    ):
        super().__init__()
        self._variant = variant
        self._freeze_epochs = freeze_epochs
        self.backbone, feat_dim = self._build_backbone(
            variant, pretrained, ark_checkpoint_path
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )
        self._frozen = False
        if freeze_epochs > 0:
            self._freeze_backbone()

    def _build_backbone(self, variant, pretrained, ark_path):
        model_name = f"swin_{variant}_patch4_window7_224"
        backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        if ark_path and Path(ark_path).exists():
            state = torch.load(ark_path, map_location="cpu")
            backbone.load_state_dict(state, strict=False)
            print(f"[Tier2ArkPlus] Loaded Ark+ weights from {ark_path}")
        else:
            print(f"[Tier2ArkPlus] Fallback: ImageNet-pretrained Swin-{variant}")
        feat_dim = backbone.num_features
        return backbone, feat_dim

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze_at_epoch(self, current_epoch: int) -> None:
        if self._frozen and current_epoch >= self._freeze_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self._frozen = False

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)

    # BaseClassifier contract:
    def get_confidence(self, logits): ...
    def mc_forward(self, x, T=20): ...
    def tta_forward(self, x, n=10): ...
    def mc_tta_forward(self, x, T=20, n=10): ...
```

### 5.3 Ark+ Checkpoint Download

`scripts/download_ark_plus.py`:
```python
ARK_URL = "https://github.com/JLiangLab/Ark/releases/download/v1.0/ark_plus_swin_base.pth"
# If unavailable: HuggingFace mirror, or fall back to Swin-Base ImageNet
```

### 5.4 Ablation Table Expansion

| ID | Name | Tier1 | Tier2 | Aug | Notes |
|----|------|-------|-------|-----|--------|
| A1 | Tier1 only | MobileNetV2 | — | Classic | Existing |
| A2 | Tier2 only (no MC/TTA) | — | EfficientNetB4 | Classic | Existing |
| A3 | Tiered (static threshold) | MNv2 | EffB4+MC+TTA | Classic | Existing |
| A4 | Tiered (dynamic threshold) | MNv2 | EffB4+MC+TTA | Classic | Existing |
| A5 | Tiered + MC only (no TTA) | MNv2 | EffB4+MC | Classic | Existing |
| A6 | Tiered + MC + TTA | MNv2 | EffB4+MC+TTA | Classic | Existing |
| A7 | Tiered + Conformal | MNv2 | EffB4+MC+TTA+Conf | Classic | Existing |
| A8 | No synthetic aug | MNv2 | EffB4 | Classic only | Retrain |
| A9 | No augmentation at all | MNv2 | EffB4 | None | Retrain |
| A10 | Always Tier2 (no routing) | — | EffB4+MC+TTA | Classic | Existing |
| **A11** | **Tier2 = Ark+ (no MC/TTA)** | — | ArkPlus | Classic | **NEW** |
| **A12** | **Tier2 = Ark+ + MC+TTA** | — | ArkPlus+MC+TTA | Classic | **NEW** |
| **A13** | **Tiered + Ark+ Tier2** | MNv2 | ArkPlus+MC+TTA+Conf | Classic+Synth | **NEW** |
| **A14** | **A13 + CheXpert zero-shot** | MNv2 | ArkPlus | Classic+Synth | **NEW generalization** |
| **A15** | **A13 with Mixup/Cutmix** | MNv2 | ArkPlus | Classic+Synth+Mixup | **NEW regularization** |

### 5.5 Direct Comparison Table (Key Thesis Figure)
A single table: EfficientNetB4 vs ArkPlus, under identical conditions, with paired bootstrap CIs.

---

## 6. ML Pipeline Improvements

### 6.1 Trainer Class — Production-Grade

New `Trainer` features:
- **Mixed precision (AMP):** `torch.cuda.amp.autocast` + `GradScaler`
- **Gradient accumulation:** `accumulate_grad_batches` from config
- **LR warmup:** `LinearWarmup` + cosine schedule
- **Checkpoint resume:** `--resume-from path/to/ckpt.pth`, epoch/optimizer/scheduler state restore
- **Multi-GPU (optional):** `torch.nn.parallel.DistributedDataParallel`
- **Gradient clipping:** `clip_grad_norm_`
- **EMA (Exponential Moving Average) weights:** Optional
- **Stochastic Weight Averaging (SWA):** Stretch
- **Carbon tracking:** CodeCarbon `kWh` measurement (table in thesis)

### 6.2 Trainer Signature

```python
class Trainer:
    def __init__(
        self,
        model: BaseClassifier,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: TrainingConfig,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        use_amp: bool = True,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float | None = 1.0,
        ema_decay: float | None = None,
    ): ...

    def add_observer(self, obs: TrainingObserver) -> None: ...

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: str | None = None,
    ) -> dict: ...  # Returns final metrics

    def _train_epoch(self, epoch, loader) -> dict: ...
    def _val_epoch(self, epoch, loader) -> dict: ...
    def _save_checkpoint(self, epoch, metrics) -> None: ...
    def _load_checkpoint(self, path) -> int: ...  # returns start_epoch
```

### 6.3 Synthetic Data Pipeline (The Thesis's Other Half-Contribution)

The existing `src/data/synthetic_gen.py` generates pneumothorax images with Stable Diffusion. Gaps:

1. **Conditional generation:** Severity (mild/moderate/severe), location (apical/basal) prompts
2. **FID quality gate:** Reject the generated batch if FID > threshold (the existing `fid_eval.py` is not integrated)
3. **Manual review UI:** `notebooks/synthetic_quality.ipynb` — grid view + accept/reject
4. **Provenance tracking:** Metadata for each synthetic image (prompt, seed, FID, generation date)
5. **Ablation entegrasyonu:** A8 = no synthetic, retrain otomatik

`SyntheticDataService` new workflow:
```python
service = SyntheticDataService(cfg)
batch = service.generate(
    n_samples=500,
    prompt_template="chest X-ray showing {severity} pneumothorax in {location} lung",
    severities=["mild", "moderate", "severe"],
    locations=["apical", "basal", "lateral"],
    seed=42,
)
filtered = service.quality_gate(batch, fid_threshold=50.0)
service.save(filtered, output_dir="data/synthetic/v2/")
```

### 6.4 Statistical Testing Module (Vital for the Thesis)

```python
# scripts/statistical_tests.py
from core.evaluation.stats import (
    delong_test,           # AUC comparison p-value
    bootstrap_ci,          # 95% CI for all metrics
    mcnemar_test,          # Paired prediction comparison
    permutation_test,      # Distribution-free comparison
)

# Example:
# A6 (EffB4) vs A13 (Ark+) bootstrap AUC comparison:
results = bootstrap_ci(
    y_true, y_probs_a6, y_probs_a13,
    metric="auc", n_iterations=10000, seed=42,
)
# results = {"a6_auc": 0.882, "a6_ci": (0.871, 0.893),
#            "a13_auc": 0.901, "a13_ci": (0.890, 0.912),
#            "delta": 0.019, "delta_ci": (0.005, 0.033),
#            "p_value_delong": 0.0023}
```

### 6.5 Decision Curve Analysis

New notebook: `notebooks/decision_curve_analysis.ipynb`. More clinically relevant than AUC: a "net benefit Y at threshold X" curve.

### 6.6 Subgroup Fairness Analysis

The NIH CSV has `Patient Age`, `Patient Gender`, `View Position` columns. Subgroup AUC:
- Age bins: <40, 40-60, 60-80, 80+
- Gender: M/F
- View: PA, AP

Per-subgroup AUC + CI + DeLong vs overall. Thesis Chapter 5.4 "Fairness Analysis".

### 6.7 Domain Shift / CheXpert Cross-Dataset

`scripts/evaluate_chexpert.py`:
1. Download CheXpert (Stanford ML Group, free academic license)
2. Image preprocessing (drop CheXpert laterals, keep AP/PA)
3. Label mapping: "Pneumothorax" 1.0 -> 1, NaN -> 0, -1 (uncertain) -> exclude
4. Inference, metrics, calibration drift figure
5. Thesis Chapter 5.5 "Cross-Dataset Generalization"

---

## 7. Web Application (Full Detail)

### 7.1 FastAPI Backend Endpoint Contracts

OpenAPI/Swagger otomatik `/docs` adresinde.

**POST /api/v1/predict**
- Multipart form: `file=<image>`, `return_gradcam=true`, `return_uncertainty=true`
- Response 200:
  ```json
  {
    "request_id": "uuid",
    "prediction": "Pneumothorax" | "No Finding",
    "confidence": 0.873,
    "tier_used": 2,
    "mc_variance": 0.045,
    "mc_passes": 20,
    "tta_passes": 10,
    "conformal_set": ["Pneumothorax"],
    "conformal_coverage": 0.95,
    "flagged_for_review": false,
    "inference_time_ms": 423.5,
    "gradcam_tier1_b64": "iVBORw0KG...",
    "gradcam_tier2_b64": "iVBORw0KG...",
    "model_version": "tier2_efficientnet_v1.2.0",
    "timestamp": "2026-05-24T12:34:56Z"
  }
  ```
- Error 400: `{"error": "Invalid image format", "code": "INVALID_INPUT"}`
- Error 503: Model not loaded

**POST /api/v1/predict/batch**
- Multipart with multiple files (max 50)
- Returns array of PredictionDTO

**WebSocket /api/v1/ws/predict**
- Client -> `{"type": "upload", "image_b64": "..."}`
- Server -> `{"type": "progress", "step": "preprocessing", "percent": 20}`
- Server -> `{"type": "progress", "step": "tier1_inference", "percent": 40}`
- Server -> `{"type": "progress", "step": "tier2_escalation", "percent": 70}`
- Server -> `{"type": "result", "data": PredictionDTO}`

**GET /api/v1/metrics**
- Response: Current model performance (from test set)
- Query: `?dataset=nih|chexpert&model=tier2_efficientnet|tier2_arkplus`

**GET /api/v1/threshold** -> `{"value": 0.75, "mode": "static"}`
**PUT /api/v1/threshold** -> Body: `{"value": 0.78}`, runtime update

**GET /api/v1/ablation** -> A1-A15 results table

**GET /api/v1/models** -> Loaded models, versions
**POST /api/v1/models/load** -> Load from MLflow Model Registry: `{"name": "tier2_arkplus", "version": "2"}`

**GET /api/v1/history?limit=50&offset=0** -> Past predictions (SQLite)
**DELETE /api/v1/history/{id}** -> Delete prediction

**GET /api/v1/health** -> `{"status": "ok", "gpu": true, "models_loaded": ["tier1", "tier2"], "version": "2.0.0", "uptime_s": 12345}`

### 7.2 React Frontend Component Hierarchy

```
App
├── Layout
│   ├── Navbar (logo, theme toggle, nav links)
│   └── <Outlet/>
├── InferencePage (/)
│   ├── UploadZone (drag-drop, paste, DICOM)
│   ├── ProgressSteps (4 step: preprocess -> T1 -> T2 -> CAM)
│   └── ResultCard
│       ├── ImagePreview (original + GradCAM overlay tabs)
│       │   ├── HeatmapOverlay (opacity slider)
│       │   └── TierCompare (side-by-side T1 vs T2 CAM)
│       ├── PredictionBadge (color-coded)
│       ├── ConfidenceBar
│       ├── UncertaintyBar (MC variance)
│       ├── ConformalSet (set members + coverage chip)
│       ├── TierBadge ("Fast Path" / "Deep Routing")
│       ├── InferenceTimeChip
│       ├── FlaggedBanner (conditional)
│       └── ExportMenu (JSON, PDF report, PNG with overlay)
├── DashboardPage (/dashboard)
│   ├── ThresholdSlider (0.50–0.95)
│   ├── MetricsChart × 4 (Recharts)
│   ├── LiveStats panel
│   └── ApplyThresholdButton (PUT /api/threshold)
├── HistoryPage (/history)
│   ├── HistoryTable (sortable, filterable, pagination)
│   ├── BulkExport (CSV/JSON)
│   └── ConfirmDeleteModal
├── AblationPage (/ablation)
│   ├── AblationTable (A1-A15)
│   ├── ComparisonChart (paired AUC + CI)
│   └── DownloadReport (PDF)
└── AboutPage (/about)
    ├── ArchitectureDiagram (SVG)
    ├── ModelComparisonTable
    ├── SystemStatus (GPU, uptime, model versions)
    └── References
```

### 7.3 Design System

- **Renkler:**
  - Primary: `#0F172A` (slate-900, navy)
  - Accent (clinical): `#0D9488` (teal-600)
  - Warning (pneumothorax positive): `#DC2626` (red-600)
  - Success (normal): `#059669` (emerald-600)
  - Surface: `#F8FAFC` light / `#0F172A` dark
- **Font:** Inter (UI), JetBrains Mono (metrics)
- **Shadow scale:** `shadow-sm`, `shadow`, `shadow-md`, `shadow-lg`
- **Radius:** `rounded-xl` default
- **Spacing:** Tailwind defaults
- **Motion:** Framer Motion, 200ms cubic-bezier(0.4, 0, 0.2, 1)
- **Dark mode:** `class`-based, persisted in localStorage
- **Responsive:** sm/md/lg/xl/2xl, mobile-first
- **Language:** English default (`i18n/en.json`); Turkish translation available via the toggle (`i18n/tr.json`).

### 7.4 State Management (Zustand)

```typescript
// store/inferenceStore.ts
interface InferenceState {
  currentResult: PredictionDTO | null;
  isLoading: boolean;
  progressStep: 'idle' | 'preprocess' | 'tier1' | 'tier2' | 'cam' | 'done';
  history: PredictionDTO[];
  setResult: (r: PredictionDTO) => void;
  setProgress: (step: string) => void;
  reset: () => void;
}
```

### 7.5 PDF Report Generation

On the frontend, via jsPDF + html2canvas:
- Logo + date
- Image thumbnail + GradCAM
- Prediction + confidence + tier
- Conformal set + coverage
- Model versions + signatures
- "Not a clinical diagnosis" disclaimer

---

## 8. DevOps & Reproducibility

### 8.1 Docker

**Dockerfile.api** (multi-stage, slim):
- Stage 1 `builder`: deps install, wheel cache
- Stage 2 `runtime`: minimal libs, non-root user, healthcheck
- ARG `BASE_IMAGE` for GPU variant (`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`)
- Size target: <2 GB CPU, <6 GB GPU

**Dockerfile.frontend:**
- Stage 1 `builder`: node:20-alpine, `npm ci && npm run build`
- Stage 2 `runtime`: nginx:alpine + built `/dist`, custom nginx.conf

**Dockerfile.training:** Full deps + CUDA, dev tools (jupyter, ipython)

### 8.2 docker-compose.yml (dev)

```yaml
services:
  api:
    build: { context: ., dockerfile: docker/Dockerfile.api }
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./outputs:/app/outputs:ro
      - ./data/processed:/app/data/processed:ro
    depends_on:
      mlflow: { condition: service_started }
    healthcheck:
      test: curl -f http://localhost:8000/api/v1/health
      interval: 30s

  frontend:
    build: { context: ., dockerfile: docker/Dockerfile.frontend }
    ports: ["3000:80"]
    depends_on:
      api: { condition: service_healthy }

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports: ["5000:5000"]
    volumes:
      - ./experiments/mlruns:/mlruns
    command: mlflow server --host 0.0.0.0 --backend-store-uri sqlite:///mlruns/mlflow.db

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    profiles: ["with-cache"]

  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: xray
      POSTGRES_DB: xray_history
    profiles: ["with-db"]
```

### 8.3 docker-compose.prod.yml (overrides)
- `restart: always`
- Resource limits (`mem_limit`, `cpus`)
- Read-only volumes
- No exposed MLflow port (internal only)
- nginx SSL termination via certbot sidecar

### 8.4 Makefile

```makefile
.DEFAULT_GOAL := help

# --- Setup ---
install:           ## Install runtime deps
	pip install -r requirements.txt

install-dev:       ## Install dev + test deps
	pip install -r requirements-dev.txt
	pre-commit install

install-training:  ## Install full training deps
	pip install -r requirements-training.txt

# --- Training ---
train-tier1:       ## Train MobileNetV2 Tier1
	python -m scripts.train_tier1 --run-name Tier1_MobileNetV2

train-tier2:       ## Train EfficientNetB4 Tier2
	python -m scripts.train_tier2 --backbone efficientnet_b4 --run-name Tier2_EfficientNet

train-tier2-ark:   ## Train Ark+ Tier2
	python -m scripts.train_tier2 --backbone ark_plus --run-name Tier2_ArkPlus

# --- Evaluation ---
evaluate-nih:      ## Evaluate on NIH test set
	python -m scripts.evaluate_tiered --dataset nih --run-name FinalEval_NIH

evaluate-chexpert: ## Cross-dataset zero-shot on CheXpert
	python -m scripts.evaluate_chexpert --run-name FinalEval_CheXpert

ablation:          ## Run all ablations A1-A15
	python -m scripts.run_ablation --start A1 --end A15

stats:             ## Run statistical tests
	python -m scripts.statistical_tests

benchmark:         ## Latency + memory + carbon
	python -m scripts.benchmark_latency

# --- Synthetic ---
generate-synthetic: ## Generate SD synthetic batch
	python -m scripts.generate_synthetics --n 500 --version v2

# --- Export ---
export-onnx:       ## ONNX export + INT8 quantization
	python -m scripts.export_onnx --model tier1
	python -m scripts.export_onnx --model tier2 --quantize int8

# --- Web ---
serve-api:         ## Start FastAPI (dev)
	uvicorn web.backend.app:app --reload --host 0.0.0.0 --port 8000

serve-frontend:    ## Start React dev server
	cd web/frontend && npm run dev

serve:             ## Start full stack via docker-compose
	docker-compose up --build

serve-prod:        ## Start production stack
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# --- Quality ---
test:              ## Run all tests
	pytest tests/ -v --cov=core --cov=application --cov-report=html

test-unit:         ## Unit tests only
	pytest tests/unit -v

test-integration:  ## Integration tests
	pytest tests/integration -v

lint:              ## Lint code
	ruff check core/ application/ infrastructure/ web/ tests/
	mypy core/ application/ infrastructure/

format:            ## Auto-format
	ruff format core/ application/ infrastructure/ web/ tests/

check-imports:     ## Verify architectural boundaries
	lint-imports

check-lang:        ## Verify code comments are English-only
	python -m scripts.check_comment_language

# --- Docker ---
docker-build:      ## Build all images
	docker-compose build

docker-push:       ## Push to registry
	docker-compose push

# --- Thesis ---
figures:           ## Generate all thesis figures
	python -m scripts.generate_report_figures --output thesis/figures

thesis-build:      ## Compile LaTeX
	cd thesis && latexmk -pdf -interaction=nonstopmode main.tex

clean:             ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov

help:              ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install install-dev install-training train-tier1 train-tier2 train-tier2-ark \
        evaluate-nih evaluate-chexpert ablation stats benchmark generate-synthetic \
        export-onnx serve-api serve-frontend serve serve-prod test test-unit \
        test-integration lint format check-imports check-lang docker-build docker-push \
        figures thesis-build clean help
```

### 8.5 Pre-commit Hooks (`.pre-commit-config.yaml`)

- ruff (lint + format)
- mypy (strict, on core/ application/)
- end-of-file-fixer, trailing-whitespace
- pytest-fast (unit only, <30s)
- detect-secrets
- **`check-comment-language`** — custom hook: greps Python/TS files for non-ASCII letters in comment lines, fails if found (enforces Madde 1.4 language policy)

### 8.6 GitHub Actions CI

**`.github/workflows/ci.yml`:**
```yaml
on: [push, pull_request]
jobs:
  lint-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - run: pip install -r requirements-dev.txt
      - run: make lint
      - run: make check-imports
      - run: make check-lang
      - run: make test-unit
      - uses: codecov/codecov-action@v4
```

**`.github/workflows/docker.yml`:** On tag push, multi-arch image build + ghcr.io push

### 8.7 Configuration System

`config/settings.py`:
```python
from pathlib import Path
from typing import Literal
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env + YAML overlay."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["dev", "prod", "test"] = "dev"
    tier1_weights_path: Path = Path("outputs/models/best_tier1_mobilenet.pth")
    tier2_weights_path: Path = Path("outputs/models/best_tier2_efficientnet.pth")
    tier2_backbone: Literal["efficientnet_b4", "ark_plus"] = "efficientnet_b4"
    confidence_threshold: float = Field(0.75, ge=0.5, le=0.95)
    mc_dropout_passes: int = 20
    tta_passes: int = 10
    conformal_alpha: float = 0.05
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]
    mlflow_tracking_uri: str = "http://localhost:5000"
    history_db_url: str = "sqlite:///./outputs/history.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`config/base.yaml` (compatible with old `config.yaml`), `dev.yaml`, `prod.yaml`, `test.yaml` overlay.

### 8.8 Reproducibility

- All random seeds from `config['training']['seed']`
- `torch.use_deterministic_algorithms(True)` during training
- DVC or at minimum MD5 checksum manifest for `data/processed/`
- MLflow run logs everything: code hash, params, metrics, artifacts, model versions
- Thesis supplementary "exact reproduction" commands:
  ```
  git checkout v1.0-thesis-defense
  make install
  make train-tier1 && make train-tier2 && make train-tier2-ark
  make ablation && make stats && make figures
  ```

### 8.9 Logging

- `structlog` JSON logger, log level from `.env`
- All `print()` calls replaced with structured logger
- API request/response logger middleware
- Full log file uploaded as MLflow run artifact

### 8.10 Monitoring (Production Optional)

- Prometheus endpoint `/metrics` (FastAPI Prometheus instrumentator)
- Grafana dashboard (latency p50/p95/p99, tier escalation rate, error rate)
- Sentry integration (`SENTRY_DSN`)

---

## 9. Test Strategy

### 9.1 Test Pyramid

| Level | Target Count | Runtime | What It Tests |
|--------|-------------|----------------|---------------|
| Unit | ~80 tests | <30s | Pure functions, factory, routing logic, metrics |
| Integration | ~25 tests | <3 minutes | Services, FastAPI TestClient, repository |
| E2E | ~5 tests | <10 minutes | Full pipeline (dummy data), ablation runner |

### 9.2 Critical Test Scenarios

**Unit:**
- `ModelFactory.create("unknown")` raises `ValueError` with available models listed
- `StaticThresholdRouter.route(0.5)` returns 2 when threshold=0.75
- `DynamicThresholdRouter` lowers threshold after window of low confidences
- `compute_all_metrics` returns correct sensitivity/specificity from known inputs
- `ConformalPredictor.predict_set` always returns >=1 class
- `bootstrap_ci(n_iter=10)` returns CI tuple with lower <= upper

**Integration:**
- `InferenceService.predict(dummy_image)` returns PredictionDTO with valid fields
- `POST /predict` with valid PNG returns 200 + correct schema
- `POST /predict` with corrupt file returns 400
- WebSocket `/ws/predict` streams 4 progress messages + 1 result
- `EvaluationService.evaluate_full_system` runs on 100-image mini dataset

**E2E:**
- `make train-tier1` with `--epochs 1 --batch-size 4 --data-fraction 0.01` finishes <2 min
- `make ablation --start A3 --end A4` produces 2 MLflow runs
- ONNX export -> ONNX Runtime inference matches PyTorch within 1e-4

### 9.3 Fixtures

`tests/conftest.py`:
- `dummy_xray`: 224x224 random tensor
- `mini_dataset`: 20 train + 5 val + 5 test
- `mini_tier1_weights`: Random init, saved tmp
- `mock_inference_service`: pytest fixture with mocked models
- `temp_mlflow_dir`: Isolated MLflow tracking

---

## 10. Thesis Writing Plan

### 10.1 LaTeX Structure

`thesis/main.tex`:
```latex
\documentclass[12pt,a4paper]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[turkish]{babel}
\usepackage{thesis}  % University template if available
\begin{document}
\include{chapters/00_onbilgi}
\include{chapters/01_giris}
\include{chapters/02_ilgili_calismalar}
\include{chapters/03_metodoloji}
\include{chapters/04_deneysel_kurulum}
\include{chapters/05_sonuclar}
\include{chapters/06_tartisma}
\include{chapters/07_sonuc}
\bibliography{bibliography}
\end{document}
```

### 10.2 Chapter-by-Chapter Content + Figure Sources

| Chapter | Pages | Key Content | Figure/Table | Generation Script |
|-------|-------|----------------|-------------|----------------|
| **1. Introduction** | 5-7 | Motivation, problem, contributions, structure | Fig 1.1: System overview | Manual (Inkscape/Excalidraw) |
| **2. Related Work** | 8-10 | CXR CNNs (CheXNet etc.), uncertainty in medical AI, foundation models (Ark+), tiered inference | — | — |
| **3. Methodology** | 12-15 | Tiered architecture, MC Dropout, TTA, Conformal, Stable Diffusion enrichment | Fig 3.1: Routing flowchart, Fig 3.2: SD pipeline | `generate_report_figures.py --section method` |
| **4. Experimental Setup** | 6-8 | Datasets (NIH + CheXpert), splits, training config, ablation design | Tbl 4.1: Data statistics, Tbl 4.2: Hyperparameters | `generate_report_figures.py --section setup` |
| **5. Results** | 18-22 | Ablation table, threshold analysis, calibration, conformal coverage, cross-dataset, subgroup, decision curve | Fig 5.1–5.12 (12 figures), Tbl 5.1–5.6 | `generate_report_figures.py --section results` |
| **6. Discussion** | 6-8 | Clinical utility, limitations, ethics, future work | — | — |
| **7. Conclusion** | 2-3 | Summary of contributions, future work | — | — |
| **Total** | 60-75 | ~80 with appendices | ~25 figures, 8 tables | |

### 10.3 Key Figures

1. **System overview** (Fig 1.1) — Single-page diagram of tiered flow, color-coded
2. **Routing decision flowchart** (Fig 3.1) — Confidence -> tier decision tree
3. **Stable Diffusion sample grid** (Fig 3.3) — 3x3 grid, severity x location
4. **FID quality curve** (Fig 3.4) — FID over generation iterations
5. **Tier1 vs Tier2 GradCAM comparison** (Fig 5.4) — Disagreement cases
6. **Threshold sweep 4-panel** (Fig 5.5) — Acc/Sens/Spec/%Tier2 vs threshold
7. **Reliability diagram** (Fig 5.6) — Before/after temperature scaling
8. **Conformal coverage histogram** (Fig 5.7) — Empirical vs target coverage
9. **Cross-dataset transfer** (Fig 5.9) — NIH vs CheXpert AUC ± CI
10. **DeLong significance heatmap** (Fig 5.10) — Pairwise p-values, A1-A15
11. **Decision curve analysis** (Fig 5.11) — Net benefit
12. **Carbon footprint comparison** (Fig 5.12) — Always-Tier2 vs Tiered kWh

### 10.4 Tezdeki Tablolar

1. Tbl 4.1: Dataset statistics (NIH train/val/test/cal, CheXpert test)
2. Tbl 4.2: Hiperparametre grid
3. Tbl 5.1: **Ana ablation tablosu** (A1-A15, AUC + CI + p-value)
4. Tbl 5.2: Per-subgroup AUC (age, gender, view)
5. Tbl 5.3: Kalibrasyon metrikleri (ECE, Brier, slope, intercept)
6. Tbl 5.4: Gecikme + bellek + karbon (Tier1, Tier2-Eff, Tier2-Ark, Tiered)
7. Tbl 5.5: Cross-dataset (NIH-trained on CheXpert)
8. Tbl 5.6: Comparison with published baselines

### 10.5 Reproducibility Eki

Tez ekinde:
- Git commit SHA (`v1.0-thesis-defense` tag)
- Conda env export
- All `make` commands
- Expected output files + MD5

---

## 11. 10-Week Roadmap

### Week 1: Architecture Refactor Foundation
- [ ] Repo branch: `refactor/layered-architecture`
- [ ] Create the folder structure (`core/`, `application/`, `infrastructure/`, `web/`, `tests/`)
- [ ] Write the `core/interfaces/` ABCs (5 ABCs)
- [ ] `pyproject.toml`, ruff, mypy, pre-commit, import-linter setup
- [ ] **Set up the `check-comment-language` pre-commit hook (Section 1.4)**
- [ ] CI baseline: GitHub Actions lint job
- [ ] `core/models/factory.py` + `Tier1MobileNet` refactor
- [ ] Set up the src/ shims
- [ ] Smoke test: does the notebook still run?

### Week 2: Models + Routing + Augmentation
- [ ] `Tier2EfficientNet`'i `BaseClassifier`'a port et
- [ ] `Tier2ArkPlus` skeleton + Swin-Base fallback
- [ ] Ark+ checkpoint indirme scripti
- [ ] `core/routing/static_router.py` + `dynamic_router.py`
- [ ] `core/augmentation/` trio (classical/diffusion/mixup)
- [ ] Type hints + docstring 100% (for core/)
- [ ] Unit tests: factory + routing (>=15 test)

### Week 3: Trainer + Observers + Resume
- [ ] `infrastructure/training/trainer.py` — AMP, gradient accumulation, clip, EMA, warmup
- [ ] Observers (MLflow, Checkpoint, EarlyStop, LRLogger, Carbon)
- [ ] `CheckpointManager` (full resume support)
- [ ] Re-train Tier1 with new trainer (sanity check vs existing weights)
- [ ] Unit + integration tests
- [ ] **Milestone:** Old training pipeline replaced with new trainer, MLflow run shows no regression

### Week 4: Ark+ Training + Ablation Expansion
- [ ] Ark+ checkpoint download or Swin-Base fallback
- [ ] Tier2 ArkPlus full training (~24-48 hours, with GPU)
- [ ] A11, A12, A13 ablation runs
- [ ] MLflow Model Registry: tier2_efficientnet v1, tier2_arkplus v1
- [ ] EfficientNet vs Ark+ paired comparison (bootstrap CI)
- [ ] **Milestone:** Ark+ trained, A13 results in MLflow

### Week 5: Synthetic Data + CheXpert
- [ ] SD pipeline ported to `SyntheticDataService`
- [ ] Conditional generation prompts (severity x location)
- [ ] FID quality gate integration
- [ ] `notebooks/synthetic_quality.ipynb` manual review UI
- [ ] CheXpert download + preprocessing script
- [ ] `CheXpertDataset` + `CheXpertRepository`
- [ ] `scripts/evaluate_chexpert.py`
- [ ] A8, A9, A14 ablation
- [ ] **Milestone:** All A1-A15 results in MLflow

### Week 6: Deepening Evaluation
- [ ] `scripts/statistical_tests.py` (DeLong + bootstrap + McNemar + permutation)
- [ ] Reliability diagram (calibration metric)
- [ ] Decision curve analysis notebook
- [ ] Subgroup fairness notebook (age/sex/view)
- [ ] Error analysis notebook (failure taxonomy)
- [ ] Tier disagreement notebook (T1 vs T2 conflict examples)
- [ ] Carbon tracking (codecarbon rerun)
- [ ] **Milestone:** All thesis Chapter 5 tables + 70% of figures ready

### Week 7: FastAPI Backend
- [ ] `web/backend/app.py` + all routes
- [ ] Pydantic schemas (request/response)
- [ ] Dependency injection (`deps.py`)
- [ ] WebSocket streaming endpoint
- [ ] SQLite history persistence
- [ ] Middleware (error, logging, rate limit, CORS)
- [ ] OpenAPI docs review
- [ ] Integration tests (TestClient + WebSocket testclient)
- [ ] **Milestone:** `curl localhost:8000/api/v1/predict` works end-to-end

### Week 8: React Frontend (Critical Pages)
- [ ] Vite + React 18 + TS + Tailwind + Zustand + React Query setup
- [ ] `locale/tr.ts` Turkish UI strings
- [ ] Inference page (UploadZone -> ResultCard -> GradCAM)
- [ ] WebSocket progress steps
- [ ] Dashboard page (threshold slider + Recharts grafikler)
- [ ] Dark/light mode + responsive
- [ ] PDF report export (Turkish disclaimer)
- [ ] **Milestone:** Inference + Dashboard pages fully functional

### Week 9: Frontend Completion + Docker + Deploy
- [ ] History page (table + filter + export)
- [ ] Ablation page (A1-A15 visualization)
- [ ] About page (architecture SVG + system status)
- [ ] DICOM upload support
- [ ] Multi-stage Dockerfile.api + Dockerfile.frontend
- [ ] docker-compose dev + prod
- [ ] HuggingFace Spaces deploy (Gradio + optional FastAPI)
- [ ] **Milestone:** `docker-compose up` brings up full stack, public URL

### Week 10: Thesis Writing + Polishing
- [ ] LaTeX chapters 1-4 draft
- [ ] Generate all figures via `generate_report_figures.py`
- [ ] ONNX export + INT8 + benchmark
- [ ] Latency/memory/carbon table finalize
- [ ] LaTeX chapters 5-7
- [ ] Proofread + supervisor review
- [ ] Defense presentation (slides + demo video)
- [ ] **Milestone:** Thesis submitted + defense ready

### Buffer (Week 11, optional)
- Stretch goals: learned router, EMA/SWA, monitoring stack, EN i18n

---

## 12. Risk Register & Decision Log

### 12.1 Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|----------|------|------------|
| R1 | Ark+ checkpoint cannot be downloaded / license barrier | Medium | High | Swin-Base ImageNet fallback; thesis notes "with publicly available alternative" |
| R2 | No / insufficient GPU, Tier2 training too long | Medium | High | Colab Pro/Pro+, Kaggle GPU, RunPod spot; mixed precision + smaller batch |
| R3 | CheXpert download blocked from Turkey/EU | Low | Medium | HuggingFace mirror, manual academic email |
| R4 | Stable Diffusion FID stays low, synthetic data hurts more than helps | Medium | Medium | A8 ablation tests this; FID gate + manual review |
| R5 | Layered refactor breaks existing weights | Low | High | Shim layer + state_dict key mapping tests |
| R6 | Thesis deadline slips (10 weeks not enough) | Medium | High | Stretch goals (EMA/SWA, monitoring) dropped; web minimal version (Inference+Dashboard) suffices |
| R7 | Jury may say "tiered systems are not new" | Low | Medium | Related work references 2024-2026 medical AI tiered papers; emphasize conformal + Ark+ + SD trio as core contribution |
| R8 | Conformal coverage not empirically tested (small calibration set) | Medium | Medium | Bootstrap CI for coverage estimate, "coverage achieved Y ± Z" report |
| R9 | Web stack takes too long, the ML part stays weak | High | High | **Web is stretch.** ML + thesis Chapter 5 first; web minimal demo in weeks 8-9 |
| R10 | Memory issues during ablation A11-A15 (ArkPlus + MC + TTA) | Medium | Medium | Gradient checkpointing, smaller batch, swap to ONNX inference |
| R11 | Code-comment language enforcement breaks legacy notebooks | Low | Low | Pre-commit hook excludes `notebooks/` and `src/` (legacy); only enforces on new files |

### 12.2 Decision Log

> This section is filled in during development. Every major technical decision is logged with a date + rationale.

- [ ] **D1 (2026-05-?):** Ark+ vs Swin-Base choice finalized (...)
- [ ] **D2:** Frontend stack: React + Vite (no Next.js — no SSR needed)
- [ ] **D3:** Persistence: SQLite (dev) + Postgres (prod opt-in)
- [ ] **D4:** ...

---

## 13. Cowork / New Session Prompt

> If you copy the prompt below into a new Cowork session, Claude will start working from this plan. Be sure to place the `PLAN.md` file in the folder.

```
# Chest X-Ray Tiered Classification — Continuation Session

## Context
This is a bachelor's thesis project at "1 Decembrie 1918" University of Alba Iulia.
I'm building a tiered confidence-routed chest X-ray classification system
(MobileNetV2 -> EfficientNetB4/Ark+) with MC Dropout, Test-Time Augmentation,
Conformal Prediction, and Stable Diffusion synthetic data augmentation.

The full master plan is in PLAN.md. Read it FIRST before doing anything else.
Pay particular attention to Madde 1 — the language policy is non-negotiable.

## Current State
- Working directory: /Users/alperen/Desktop/xray
- Existing trained weights: outputs/models/best_tier1_mobilenet.pth,
  best_tier2_efficientnet.pth
- Working Gradio demo: demo/app.py
- src/ has the original (pre-refactor) code — keep it via shims, do not delete

## What I want from you in this session
[FILL IN THE SPECIFIC PHASE — e.g., "Implement Phase 1 (Week 1) —
Architectural Foundation"]

## Hard Constraints (from PLAN.md)
1. LANGUAGE: Project + thesis are Turkish. ALL code comments, docstrings,
   variable/function/class names, log messages, exception messages, commit
   messages, and test names MUST be English. Pre-commit hook enforces this.
   See Madde 1 of PLAN.md for the full policy.
2. Layered architecture: core/ -> application/ -> web/,
   infrastructure/ -> core/. Verified at CI time with import-linter.
3. Every public function: type hints + Google-style docstring (English).
4. No bare except. No print() in production. No magic numbers.
5. Design patterns: Factory (models), Strategy (routing/aug),
   Observer (training), Repository (data).
6. Existing trained weights MUST remain loadable.
7. Notebooks must still run.
8. Python 3.10+, use match/case where appropriate.
9. Pre-commit hooks (ruff + mypy + import-linter + pytest-fast + check-lang).

## Quality Gates
- Tests pass: `make test`
- Lint clean: `make lint`
- Imports respect architecture: `make check-imports`
- Comment language enforced: `make check-lang`
- Smoke test: existing Gradio demo (`python demo/app.py`) still works

## Output Style
Turkish in chat, concise, blunt. Call me "kanka" or "kral".
Show diffs/code, not lectures. If you're about to spend tokens on a long
explanation, ask first whether I want it.
```

---

## 14. Open Questions

- [ ] Ark+ resmi checkpoint URL'i / dosya boyutu (manual check needed)
- [ ] CheXpert v1.0 small vs full (size + download time)
- [ ] GPU budget: Colab Pro, local, or cloud? (must be settled before Week 4)
- [ ] Did the thesis template come from the university, or is it standard LaTeX?
- [ ] Demo deploy: HuggingFace Spaces mi, kendi VPS mi?
- [ ] Will multi-label extension (further work) appear in the thesis, or as "future work"?

---

## 15. Definition of Done — Minimum to Submit the Thesis

**Acceptable minimum (if the 10-week limit is reached):**
- ✅ Architecture refactor complete, tests passing, old weights work
- ✅ Tier2 alternative trained with Ark+ or Swin-Base
- ✅ A1-A13 ablation complete (A14-A15 stretch)
- ✅ DeLong + bootstrap statistical tests done
- ✅ Reliability diagram + conformal coverage + 1 fairness analysis
- ✅ FastAPI minimal backend + React minimal frontend (Inference + Dashboard)
- ✅ Tek-komut Docker stack
- ✅ Thesis Chapters 1-5 written, all key figures rendered
- ✅ Reproducibility eki (git tag + komutlar)
- ✅ Language policy (Section 1) 100% applied, enforced in CI

**Stretch (quality-first):**
- 🌟 A14-A15 ablation, CheXpert cross-dataset
- 🌟 Decision curve + subgroup + error analysis notebooks fully done
- 🌟 ONNX + INT8 + benchmark + carbon
- 🌟 History + Ablation + About pages
- 🌟 DICOM upload
- 🌟 HuggingFace deploy
- 🌟 Prometheus + Grafana
- 🌟 EN i18n

---

## 16. Appendix A: New Dependencies (`requirements.txt` additions)

```
# New runtime
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.6
pydantic-settings>=2.2
python-multipart>=0.0.9
slowapi>=0.1.9       # rate limit
structlog>=24.1
sqlalchemy>=2.0
alembic>=1.13        # migrations
pydicom>=2.4         # DICOM
onnx>=1.15
onnxruntime>=1.17

# New model
timm>=0.9.16

# New evaluation
scipy>=1.12          # DeLong, permutation
statsmodels>=0.14
codecarbon>=2.3      # carbon tracking
```

`requirements-dev.txt`:
```
pytest>=8.0
pytest-cov>=4.1
pytest-asyncio>=0.23
ruff>=0.4
mypy>=1.9
pre-commit>=3.6
import-linter>=2.0
httpx>=0.27          # FastAPI TestClient
```

`requirements-training.txt`:
```
diffusers>=0.27
transformers>=4.40
accelerate>=0.29
optuna>=3.6          # hyperparam sweep
```

---

## 17. Appendix B: pyproject.toml Template (Key Sections)

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
strict = true
files = ["core", "application", "infrastructure"]
exclude = ["src", "tests"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.importlinter]
root_packages = ["core", "application", "infrastructure", "web"]

[[tool.importlinter.contracts]]
name = "core has no outer dependencies"
type = "forbidden"
source_modules = ["core"]
forbidden_modules = ["application", "infrastructure", "web"]

[[tool.importlinter.contracts]]
name = "application does not depend on web"
type = "forbidden"
source_modules = ["application"]
forbidden_modules = ["web"]
```

---

## 18. Appendix C: Language-Policy Pre-commit Hook Skeleton

`scripts/check_comment_language.py`:
```python
"""Check that code comments and docstrings are English-only.

This script greps Python and TypeScript source files for non-ASCII
letters (specifically Turkish-only characters) appearing in comment
or docstring positions. Fails the commit if any are found.

Allowed: variable names, identifiers, string literals (user-facing text).
Forbidden: comments, docstrings, log messages.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
# Match comment lines or docstring blocks
COMMENT_PATTERNS = {
    ".py": [re.compile(r"^\s*#.*"), re.compile(r'^\s*""".*?"""', re.DOTALL)],
    ".ts": [re.compile(r"^\s*//.*"), re.compile(r"/\*.*?\*/", re.DOTALL)],
    ".tsx": [re.compile(r"^\s*//.*"), re.compile(r"/\*.*?\*/", re.DOTALL)],
}

EXCLUDED_PATHS = {"src/", "notebooks/", "thesis/", "web/frontend/src/locale/"}


def check_file(path: Path) -> list[str]:
    """Return list of violation messages for the given file."""
    if any(str(path).startswith(ex) for ex in EXCLUDED_PATHS):
        return []
    suffix = path.suffix
    if suffix not in COMMENT_PATTERNS:
        return []
    content = path.read_text(encoding="utf-8")
    violations = []
    for pattern in COMMENT_PATTERNS[suffix]:
        for match in pattern.finditer(content):
            text = match.group(0)
            if any(c in TURKISH_CHARS for c in text):
                line_no = content[: match.start()].count("\n") + 1
                violations.append(f"{path}:{line_no}: Turkish in comment/docstring: {text[:60]!r}")
    return violations


def main(argv: list[str]) -> int:
    files = [Path(p) for p in argv[1:]]
    all_violations: list[str] = []
    for f in files:
        all_violations.extend(check_file(f))
    if all_violations:
        print("Language policy violations (Madde 1):", file=sys.stderr)
        for v in all_violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

`.pre-commit-config.yaml` additional block:
```yaml
- repo: local
  hooks:
    - id: check-comment-language
      name: Check code comments are English-only
      entry: python scripts/check_comment_language.py
      language: system
      types_or: [python, ts, tsx]
```

---

**Belge Sonu.**
For questions/changes, append to the `Decision Log` section and `git commit -m "plan: ..."`.
