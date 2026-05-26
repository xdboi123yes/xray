# Real Ablation Run Specification Checklist

This document details the step-by-step commands, configurations, and environment resources required to honestly execute the 9 primary ablation experiments (A1-A15) mapped in the thesis. 

These runs must be executed on a dedicated GPU instance (e.g., RunPod, Google Colab Pro, or local RTX 3090/4090/A100 server) in the week 5 "ML story closure" phase.

---

## Hardware Pre-requisites
- **GPU**: Minimum 16GB VRAM (RTX 3080/4080, A10G, or T4).
- **RAM**: Minimum 16GB System RAM.
- **Disk Space**: ~15GB for datasets and MLflow local persistent runs database.

---

## Setup Environment
Ensure dependencies are properly provisioned:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## Ablation Command Specifications

### A1: Tier 1 Only (MobileNetV3)
Trains only the MobileNetV3-Large screener backbone.
```bash
python scripts/train_tier1.py --backbone mobilenet_v3 --epochs 50 --batch-size 64
```

### A2: Tier 2 Only (EfficientNetB4)
Trains only the specialist EfficientNetB4 classifier.
```bash
python scripts/train_tier2.py --backbone efficientnet_b4 --epochs 50 --batch-size 32
```

### A3: Tier 2 Only (Ark+ Swin)
Trains only the specialist Swin backbone (Ark+ custom).
```bash
python scripts/train_tier2.py --backbone ark_plus --epochs 50 --batch-size 32 --no-mc-tta
```

### A6: Proposed Kademeli (Static Routing)
Evaluates cascading routing using a static routing threshold rather than dynamically routed predictions.
```bash
python scripts/evaluate_tiered.py --tier1-backbone mobilenet_v3 --tier2-backbone efficientnet_b4 --threshold 0.75 --routing static
```

### A8: Proposed System (Without Diffusion)
Trains the cascading model configurations omitting synthetic Stable Diffusion images from the training loaders.
```bash
python scripts/train_tier2.py --backbone efficientnet_b4 --epochs 50 --batch-size 32 --exclude-synthetic
```

### A9: Proposed System (Without Any Augmentations)
Baseline tiered model trained with classical augmentations only (omitting mixup, cutmix, and SD synthetic data).
```bash
python scripts/train_tier2.py --backbone efficientnet_b4 --epochs 50 --batch-size 32 --no-mixup-cutmix --exclude-synthetic
```

### A13: Proposed Kademeli + Ark+ (Dynamic Routing)
Proposed dynamic routing system with MobileNetV3 (T1) and Ark+ (T2) backbones under MC Dropout.
```bash
python scripts/evaluate_tiered.py --tier1-backbone mobilenet_v3 --tier2-backbone ark_plus --routing dynamic --uncertainty mc_dropout
```

### A14: Zero-Shot CheXpert Cohort Validation
Cross-dataset out-of-domain evaluation of the A13 cascading model evaluated strictly on the CheXpert validation cohort.
```bash
python scripts/evaluate_chexpert.py --tier1-backbone mobilenet_v3 --tier2-backbone ark_plus
```

### A15: Mixup/Cutmix Regularized proposed system
Dynamic proposed cascading configuration utilizing extensive regularization routines during the training loops.
```bash
python scripts/train_tier2.py --backbone ark_plus --epochs 50 --batch-size 32 --mixup 0.8 --cutmix 1.0
```

---

## Verification of Run Log Provenance
Once each command successfully completes:
1. Access local MLflow dashboard: `mlflow ui --port 5000`
2. Confirm the presence of corresponding `auc_roc`, `accuracy`, and `ece` values under `experiments/mlruns/762301816938414973/`.
3. Verify that the files' physical modification timestamps match the real clock execution time.
4. Run `python scripts/build_ablation_json.py` to compile the true values into `outputs/results/ablation.json` with `"provenance": "mlflow_run"`.
