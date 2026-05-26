# Tiered Confidence-Based Chest X-Ray Pathology Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xdboi123yes/xray/blob/main/notebooks/xray_colab_training_auto.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Tiered Confidence-Based Inference System for chest X-ray pathology classification (pneumothorax detection).
At inference time, each image is dynamically routed through a lightweight fast model (**MobileNetV2**)
or a deeper, more expensive model (**EfficientNet-B4** / **Ark+ Swin**) based on a confidence threshold.

---

## 🚀 Quick Start — Train on Colab

Click the **Open in Colab** badge above, select a T4 GPU, and run cells top-to-bottom with `Shift+Enter`.
The notebook automates everything end-to-end:

1. Clones the repo from GitHub
2. Installs dependencies (with automatic kernel restart)
3. Downloads the NIH ChestX-ray14 dataset from Kaggle
4. Preprocesses and splits the data (train/val/test/calibration)
5. Downloads the Ark+ checkpoint (with Swin-Base ImageNet fallback)
6. Trains **Tier 1 (MobileNetV2)** + **Tier 2 (EffNet-B4)** + **Tier 2 (Ark+ Swin)**
7. Performs temperature calibration and computes ECE
8. Runs the ablation matrix (A8–A15)
9. Runs statistical tests (DeLong + Bootstrap)
10. Exports models to ONNX with INT8 quantization
11. Syncs outputs to Drive and downloads a single ZIP

The only manual step is uploading `kaggle.json` (Kaggle Account > API > Create New Token).

---

## 🛠️ Local Setup

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-training.txt    # additional training-only deps
```

### 2. Dataset

The NIH ChestX-ray14 dataset is required:

- Place `Data_Entry_2017.csv` in `data/raw/`
- Extract all `.png` images into a single flat directory (default: `data/raw/images/`)

Using the Kaggle CLI:

```bash
kaggle datasets download -d nih-chest-xrays/data -p data/raw --unzip
```

### 3. Preprocess

```bash
python scripts/preprocess.py --image-dir data/raw/images
```

Produces `train.csv`, `val.csv`, `test.csv`, and `image_dir.txt` under `data/processed/`.
Filters for Pneumothorax and No Finding classes, with 5:1 class balancing.

### 4. Configuration

All hyperparameters live in `config.yaml`. Review before training.

### 5. Training

Programmatic API (matches the notebook):

```python
from application.dto.training_config_dto import TrainingConfigDTO
from application.services.training_service import TrainingService

cfg = TrainingConfigDTO(
    backbone='mobilenet_v2',
    run_name='Tier1_Local',
    epochs=50,
    batch_size=32,
    lr_backbone=1e-4,
    lr_head=1e-3,
    early_stopping_patience=7,
    seed=42,
    use_synthetic=False,
)
TrainingService().train_model(cfg)
```

Or via CLI:

```bash
python scripts/train_tier1.py
python scripts/train_tier2.py
```

Output: `outputs/models/<run_name>/best_model.pth`.

### 6. MLflow Tracking

```bash
mlflow ui --backend-store-uri experiments/mlruns
```

Open `http://localhost:5000`.

### 7. Synthetic Data (optional)

Generate synthetic Pneumothorax samples using Stable Diffusion:

```bash
python scripts/generate_synthetics.py
```

### 8. Clinical Statistical Tests

DeLong, McNemar, DCA, Bootstrap CIs:

```bash
python scripts/statistical_tests.py --output outputs/results/statistical_comparison.csv
```

### 9. Ablation Matrix

A8–A15 ablation experiments:

```python
from application.orchestration.ablation_runner import AblationRunner
AblationRunner().run_all(dry_run=False)
```

Then regenerate the honest `ablation.json`:

```bash
python scripts/build_ablation_json.py
```

### 10. ONNX Export

```bash
python scripts/export_onnx.py --model tier1
python scripts/export_onnx.py --model tier2 --quantize
```

### 11. FastAPI Backend + React Frontend

```bash
make serve-api          # port 8000
make serve-frontend     # port 5173
```

### 12. Docker Compose (production-ready)

```bash
make serve              # dev: 3000 / 8000 / 5000
make serve-prod         # prod: port 80, resource limits enabled
```

---

## 📁 Project Structure

```
xray/
├── application/                # Use cases / DTOs / services
│   ├── dto/
│   ├── services/               # TrainingService, CalibrationService, ...
│   └── orchestration/          # AblationRunner
├── core/                       # Domain models, interfaces, evaluation
│   ├── models/                 # Tier 1/2 backbones (factory pattern)
│   ├── uncertainty/            # Temperature scaling, conformal prediction
│   └── evaluation/             # Stats (DeLong, bootstrap, McNemar)
├── infrastructure/             # Data, persistence, training framework
│   ├── data/                   # NIHChestXrayDataset
│   └── training/               # Trainer, observers (checkpoint, mlflow, ...)
├── scripts/                    # CLI entry points
├── notebooks/                  # Colab + analysis notebooks
├── web/                        # React frontend + FastAPI backend
├── docker/                     # Dockerfiles
├── config.yaml                 # Centralized configuration
└── requirements*.txt
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

Lint and type checks:

```bash
ruff check .
mypy .
lint-imports                    # import-linter
```

---

## 📚 More

- **CHANGELOG.md** — release history
- **CONTRIBUTING.md** — contribution guide

---

## 📜 License

MIT — see [LICENSE](LICENSE).
