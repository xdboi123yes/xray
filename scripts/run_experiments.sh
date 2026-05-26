#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Full Ablation Experiments (A1-A10)..."

echo "=========================================="
echo "A1: Training Tier 1 Model Only"
echo "=========================================="
python scripts/train_tier1.py --run-name A1_Tier1Only

echo "=========================================="
echo "A2: Training Tier 2 Model Only (No MC/TTA)"
echo "=========================================="
python scripts/train_tier2.py --run-name A2_Tier2Only --no-mc-tta

echo "=========================================="
echo "A3: Evaluating Full Tiered System (Static Threshold)"
echo "=========================================="
python scripts/evaluate_tiered.py --run-name A3_Static_Threshold --dynamic-threshold false

echo "=========================================="
echo "A4: Evaluating Full Tiered System (Dynamic Threshold)"
echo "=========================================="
python scripts/evaluate_tiered.py --run-name A4_Dynamic_Threshold --dynamic-threshold true

echo "=========================================="
echo "A5: Dynamic Threshold + MC Dropout (no TTA)"
echo "=========================================="
# A5 uses MC Dropout but disables TTA by temporarily modifying config
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model']['tta_passes'] = 1
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
python scripts/evaluate_tiered.py --run-name A5_MC_Only --dynamic-threshold true
# Restore config
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model']['tta_passes'] = 10
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"

echo "=========================================="
echo "A6: Dynamic Threshold + MC Dropout + TTA"
echo "=========================================="
python scripts/evaluate_tiered.py --run-name A6_MC_TTA --dynamic-threshold true

echo "=========================================="
echo "A7: Full System + Conformal Prediction"
echo "=========================================="
# A7 is same as A6 but with conformal predictor (loaded automatically if q_hat.pt exists)
python scripts/evaluate_tiered.py --run-name A7_Full_Conformal --dynamic-threshold true

echo "=========================================="
echo "A8: Without Synthetic Augmentation"
echo "=========================================="
echo "NOTE: A8 requires retraining models without synthetic data."
echo "Disable synthetic_augmentation in config.yaml and retrain before running A8."
echo "Skipping A8 - requires manual config change and retraining."

echo "=========================================="
echo "A9: Without Any Augmentation"
echo "=========================================="
echo "NOTE: A9 requires retraining with no augmentation at all."
echo "Skipping A9 - requires manual config change and retraining."

echo "=========================================="
echo "A10: Classical Ensemble (No Routing)"
echo "=========================================="
# Force all images to Tier 2 by setting threshold to 1.0
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model']['confidence_threshold'] = 1.0
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
python scripts/evaluate_tiered.py --run-name A10_Ensemble --dynamic-threshold false
# Restore config
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model']['confidence_threshold'] = 0.75
with open('config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"

echo "=========================================="
echo "Generating Final Thesis Figures"
echo "=========================================="
python scripts/generate_report_figures.py

echo ""
echo "=========================================="
echo "Ablation Experiments Completed Successfully!"
echo "=========================================="
echo ""
echo "For programmatic ablation with config overrides, use:"
echo "  python -m src.evaluation.ablation --experiments A1 A2 A3"
echo "  python -m src.evaluation.ablation --list"
