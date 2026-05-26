# Tiered Confidence-Based Chest X-Ray Pathology Classification

This project implements a Tiered Confidence-Based Inference System for chest X-ray pathology classification, specifically targeting pneumothorax detection. The system dynamically routes each input image through either a lightweight fast model (MobileNetV2) or a deep, computationally expensive model (EfficientNetB4) based on a confidence threshold evaluated at inference time.

## Setup Instructions

### 1. Environment Setup

It is recommended to use a virtual environment (like `venv` or `conda`).

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

This project uses the NIH ChestX-ray14 dataset.

1. Download the NIH ChestX-ray14 dataset from Kaggle or the NIH National Library of Medicine.
2. Ensure the `Data_Entry_2017.csv` metadata file is placed in `data/raw/`.
3. Extract the image files. By default, the dataset script expects the images to be in a single directory which you will provide to the dataset class.

### 3. Preprocessing

Split the data into training, validation, and test sets.

```bash
python scripts/preprocess.py
```

This script will:

- Read `Data_Entry_2017.csv` from `data/raw/`.
- Filter for 'Pneumothorax' and 'No Finding' classes.
- Perform a stratified split based on the ratios defined in `config.yaml` (default: 70/15/15).
- Save `train.csv`, `val.csv`, and `test.csv` in `data/processed/`.

### 4. Configuration

All hyperparameters for model architecture, training, data augmentation, and evaluation are centralized in `config.yaml`. Review and adjust these parameters before running training scripts or experiments.

### 5. Tracking

This project uses MLflow for experiment tracking. Logs and artifacts will be saved by default to the `experiments/mlruns` directory.

### 6. Training

To train the lightweight Tier 1 model:
```bash
python scripts/train_tier1.py
```

To train the deep Tier 2 model:
```bash
python scripts/train_tier2.py
```

### 7. Synthetic Data Augmentation

Generate high-quality synthetic Pneumothorax images using Stable Diffusion to combat class imbalance:
```bash
python scripts/generate_synthetics.py
```

### 8. Clinical Statistical Tests & DCA Evaluations

To run DeLong tests, McNemar accuracy comparison, Decision Curve Analysis (DCA), and bootstrap CI calculations:
```bash
python -m scripts.statistical_tests
```

### 9. FastAPI Backend & React Frontend (Clinician Suite)

To launch the system locally:

```bash
# Start the FastAPI Backend service (Port 8000)
make serve-api

# Start the React Frontend interface (Port 5173 / Port 3000)
make serve-frontend
```

### 10. Single-Command Docker Orchestration (Production-Ready)

To launch the entire microservice architecture (FastAPI, React, Nginx Proxy, and MLflow) with one command:

```bash
# Developer Environment (Port 3000 -> Frontend, Port 8000 -> API, Port 5000 -> MLflow)
make serve

# Production Environment (Port 80 -> Frontend, resource limits enabled)
make serve-prod
```

