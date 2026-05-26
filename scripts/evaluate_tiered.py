import os
import argparse
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pandas as pd
import json
import mlflow

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import get_settings
from infrastructure.data.dataset import NIHChestXrayDataset
from core.augmentation.classical import ClassicalAugmentation
from core.models.factory import ModelFactory
import core.models.tier1_mobilenet  # registers mobilenet_v2
import core.models.tier2_efficientnet  # registers efficientnet_b4
import core.models.tier2_ark  # registers ark_plus
from core.models.tiered_system import TieredSystem
from core.evaluation.metrics import compute_all_metrics
from core.uncertainty.conformal import ConformalPredictor

def setup_mlflow_local(experiment_name="chest_xray_tiered", tracking_uri="experiments/mlruns"):
    os.makedirs(tracking_uri, exist_ok=True)
    absolute_uri = f"file://{os.path.abspath(tracking_uri)}"
    mlflow.set_tracking_uri(absolute_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow initialized. Tracking URI: {absolute_uri}, Experiment: {experiment_name}")

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tiered System")
    parser.add_argument('--run-name', type=str, required=True, help="MLflow run name")
    parser.add_argument('--dynamic-threshold', type=str2bool, default=False, help="Enable dynamic thresholding")
    parser.add_argument('--tier2-backbone', type=str, default="efficientnet_b4", choices=["efficientnet_b4", "ark_plus"],
                        help="Tier 2 model backbone type")
    args = parser.parse_args()

    settings = get_settings()
    
    setup_mlflow_local()
    mlflow.start_run(run_name=args.run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Models
    tier1 = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    
    print(f"Loading Tier 2 model backbone: {args.tier2_backbone}...")
    tier2 = ModelFactory.create(args.tier2_backbone, num_classes=2, pretrained=False).to(device)
    
    tier1_path = 'outputs/models/best_tier1_mobilenet.pth'
    tier2_path = f'outputs/models/best_tier2_{args.tier2_backbone}.pth'
    
    if not os.path.exists(tier1_path) or not os.path.exists(tier2_path):
        print(f"Warning: Trained model weights not found at {tier1_path} or {tier2_path}.")
        print("Evaluation requires both models to be trained. Continuing with untrained weights for testing purposes.")
        
    if os.path.exists(tier1_path):
        tier1.load_state_dict(torch.load(tier1_path, map_location=device))
    if os.path.exists(tier2_path):
        tier2.load_state_dict(torch.load(tier2_path, map_location=device))
        
    tier1.eval()
    tier2.eval()
    
    # Conformal Predictor
    cp_path = 'outputs/results/q_hat.pt'
    cp = ConformalPredictor(alpha=settings.evaluation.conformal_coverage) # coverage parameter maps to conformal_alpha or error toleration 1 - coverage
    # For compatibility, let's map settings conformal_coverage
    cp.alpha = 1.0 - settings.evaluation.conformal_coverage

    if os.path.exists(cp_path):
        cp.load(cp_path)
        print(f"Conformal calibration loaded (q_hat={cp.q_hat:.4f})")
    else:
        # Calibrate conformal predictor on validation set
        print("No saved conformal calibration found. Calibrating on validation set...")
        val_csv = 'data/processed/val.csv'
        if os.path.exists(val_csv):
            cal_dataset = NIHChestXrayDataset(
                csv_file=val_csv,
                transform=ClassicalAugmentation(image_size=settings.data.image_size, is_training=False)._pipeline
            )
            cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=False)
            cp.calibrate(tier2, cal_loader, device)
            os.makedirs('outputs/results', exist_ok=True)
            cp.save(cp_path)
            print(f"Conformal q_hat saved to {cp_path} (q_hat={cp.q_hat:.4f})")
    
    # Load threshold
    threshold_path = 'outputs/models/tier1_mobilenet_threshold.json'
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            t_data = json.load(f)
            static_threshold = t_data.get('optimal_threshold', settings.model.confidence_threshold)
    else:
        static_threshold = settings.model.confidence_threshold
        
    # Build dynamic/static config dict for TieredSystem compatibility
    config_dict = {
        "model": {
            "confidence_threshold": static_threshold,
            "threshold_window_size": settings.model.threshold_window_size,
            "threshold_delta": 0.0 if not args.dynamic_threshold else settings.model.threshold_delta,
            "mc_dropout_passes": settings.model.mc_dropout_passes,
            "tta_passes": settings.model.tta_passes,
        },
        "data": {
            "image_size": settings.data.image_size,
        }
    }
        
    # Tiered System
    tiered_system = TieredSystem(tier1, tier2, config_dict, conformal_predictor=cp)
    
    # Dataset
    image_size = settings.data.image_size
    test_csv = 'data/processed/test.csv'
    
    if not os.path.exists(test_csv):
        print("Error: Test set not found. Please run preprocessing first.")
        mlflow.end_run()
        return
        
    test_dataset = NIHChestXrayDataset(
        csv_file=test_csv,
        transform=ClassicalAugmentation(image_size=image_size, is_training=False)._pipeline
    )
    
    # DataLoader (batch size 1 for sequential routing simulation)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Evaluating {len(test_dataset)} images...")
    
    y_true = []
    y_probs = []
    tier2_count = 0
    total_inference_time = 0.0
    
    for images, labels, image_ids in tqdm(test_loader, desc="Evaluating System"):
        images = images.to(device)
        label = labels.item()
        image_id = image_ids[0] if isinstance(image_ids, (list, tuple)) else image_ids
        
        result = tiered_system.route(images, image_id=image_id)
        
        y_true.append(label)
        # 'Pneumothorax' is class 1. If prediction is Pneumothorax, confidence applies to class 1.
        if result.prediction == 'Pneumothorax':
            prob = result.confidence
        else:
            prob = 1.0 - result.confidence
            
        y_probs.append(prob)
        
        if result.tier_used == 2:
            tier2_count += 1
            
        total_inference_time += result.inference_time_ms
        
    metrics = compute_all_metrics(y_true, y_probs, threshold=0.5)
    
    percent_tier2 = (tier2_count / len(test_dataset)) * 100
    avg_inference_time = total_inference_time / len(test_dataset)
    
    print("\n--- Evaluation Results ---")
    print(f"Percent Tier 2: {percent_tier2:.2f}%")
    print(f"Avg Inference Time: {avg_inference_time:.2f} ms")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    mlflow.log_params({
        "dynamic_threshold": args.dynamic_threshold,
        "base_threshold": static_threshold
    })
    
    mlflow.log_metrics({
        "percent_tier2": percent_tier2,
        "avg_inference_time_ms": avg_inference_time,
        **metrics
    })
    
    mlflow.end_run()

if __name__ == '__main__':
    main()

