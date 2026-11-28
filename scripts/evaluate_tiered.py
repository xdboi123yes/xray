import argparse
import json
import math
import os
import sys

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from tcrp.config import get_settings
from tcrp.data.augmentation import ClassicalAugmentation
from tcrp.data.datasets import NIHChestXrayDataset
from tcrp.evaluation.metrics import compute_all_metrics
from tcrp.models import ModelFactory, TieredSystem
from tcrp.uncertainty.calibration import calibrate_temperature
from tcrp.uncertainty.conformal import ConformalPredictor


def setup_mlflow_local(
    experiment_name: str = "chest_xray_tiered", tracking_uri: str = "experiments/mlruns"
) -> bool:
    """Configure a local MLflow tracking store (best-effort).

    Newer MLflow versions reject the filesystem tracking backend unless
    ``MLFLOW_ALLOW_FILE_STORE`` is set, so we opt in explicitly. Any failure is
    non-fatal: the evaluation still writes its result JSON without MLflow.

    Returns:
        True if MLflow tracking was initialized, False otherwise.
    """
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.makedirs(tracking_uri, exist_ok=True)
        absolute_uri = f"file://{os.path.abspath(tracking_uri)}"
        mlflow.set_tracking_uri(absolute_uri)
        mlflow.set_experiment(experiment_name)
        print(f"MLflow initialized. Tracking URI: {absolute_uri}, Experiment: {experiment_name}")
        return True
    except Exception as exc:
        print(f"Warning: MLflow tracking unavailable ({exc}); continuing without MLflow logging.")
        return False

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_model_weights(model, weights_path, device):
    """Load weights into a model, unwrapping a training checkpoint if needed.

    Training saves checkpoints as {'epoch', 'best_metric', 'model_state_dict', ...},
    but load_state_dict expects the bare state dict. Passing the whole wrapper raised
    'Missing/Unexpected key(s)' RuntimeError and made A13/A14 fail within seconds.
    """
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tiered System")
    parser.add_argument('--run-name', type=str, required=True, help="MLflow run name")
    parser.add_argument('--dynamic-threshold', type=str2bool, default=False, help="Enable dynamic thresholding")
    parser.add_argument('--tier2-backbone', type=str, default="efficientnet_b4", choices=["efficientnet_b4", "ark_plus"],
                        help="Tier 2 model backbone type")
    args = parser.parse_args()

    settings = get_settings()
    
    mlflow_on = setup_mlflow_local()
    if mlflow_on:
        try:
            mlflow.start_run(run_name=args.run_name)
        except Exception as exc:
            print(f"Warning: could not start MLflow run ({exc}); continuing without MLflow logging.")
            mlflow_on = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Models
    tier1 = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    
    print(f"Loading Tier 2 model backbone: {args.tier2_backbone}...")
    tier2 = ModelFactory.create(args.tier2_backbone, num_classes=2, pretrained=False).to(device)
    
    tier1_path = 'outputs/models/best_tier1_mobilenet.pth'
    tier2_path = f'outputs/models/best_tier2_{args.tier2_backbone}.pth'
    if args.tier2_backbone == 'ark_plus' and not os.path.exists(tier2_path):
        fallback_path = 'outputs/models/best_tier2_arkplus.pth'
        if os.path.exists(fallback_path):
            tier2_path = fallback_path
    
    if not os.path.exists(tier1_path) or not os.path.exists(tier2_path):
        raise FileNotFoundError(
            f"Evaluation requires trained weights at {tier1_path} and {tier2_path}; "
            "refusing to report metrics from randomly initialised models."
        )
        
    if os.path.exists(tier1_path):
        load_model_weights(tier1, tier1_path, device)
    if os.path.exists(tier2_path):
        load_model_weights(tier2, tier2_path, device)
        
    tier1.eval()
    tier2.eval()
    
    # Post-hoc temperature scaling strictly on Validation Split BEFORE disjoint conformal q_hat
    eval_transform = ClassicalAugmentation(
        image_size=settings.data.image_size, is_training=False
    ).pipeline
    temperature = 1.0
    temp_path = (
        "outputs/results/temperature.pt"
        if args.tier2_backbone == "efficientnet_b4"
        else f"outputs/results/temperature_{args.tier2_backbone}.pt"
    )
    loaded_temp = False
    if os.path.exists(temp_path):
        try:
            temp_state = torch.load(temp_path, map_location="cpu", weights_only=False)
            if isinstance(temp_state, dict):
                saved_backbone = temp_state.get("backbone")
                raw_t = temp_state.get("temperature")
                if (
                    saved_backbone in (None, args.tier2_backbone)
                    and not isinstance(raw_t, bool)
                    and isinstance(raw_t, (int, float))
                    and math.isfinite(float(raw_t))
                    and float(raw_t) > 0.0
                ):
                    temperature = float(raw_t)
                    loaded_temp = True
                    print(f"Loaded Validation-Split temperature (T_cal={temperature:.4f})")
        except Exception as exc:
            print(f"Warning: could not load {temp_path} ({exc}); fitting on Validation Split.")

    val_csv = "data/processed/val.csv"
    if not loaded_temp and os.path.exists(val_csv):
        print("Fitting post-hoc temperature scaling strictly on Validation Split (val.csv)...")
        val_dataset = NIHChestXrayDataset(csv_file=val_csv, transform=eval_transform)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        _, fitted_temp = calibrate_temperature(tier2, val_loader, device)
        temperature = (
            float(fitted_temp)
            if (math.isfinite(float(fitted_temp)) and float(fitted_temp) > 0.05)
            else 1.0
        )
        print(f"Fitted Validation-Split temperature (T_cal={temperature:.4f})")

    # Conformal Predictor calibrated strictly on disjoint Calibration Split with temperature scaling.
    # Scope per backbone so ark_plus runs never overwrite EfficientNet's outputs/results/q_hat.pt.
    cp_path = (
        "outputs/results/q_hat.pt"
        if args.tier2_backbone == "efficientnet_b4"
        else f"outputs/results/q_hat_{args.tier2_backbone}.pt"
    )
    cp = ConformalPredictor(alpha=1.0 - settings.evaluation.conformal_coverage)

    need_calibration = True
    if os.path.exists(cp_path):
        try:
            probe = torch.load(cp_path, map_location="cpu", weights_only=False)
            saved_backbone = probe.get("backbone") if isinstance(probe, dict) else None
            if saved_backbone in (None, args.tier2_backbone):
                cp.load(cp_path)
                temp_matches = math.isclose(
                    cp.temperature, temperature, rel_tol=1e-3, abs_tol=1e-4
                )
                need_calibration = cp.q_hat is None or not temp_matches
                if not need_calibration:
                    print(
                        f"Conformal calibration loaded (q_hat={cp.q_hat:.4f}, "
                        f"T_cal={cp.temperature:.4f})"
                    )
                elif cp.q_hat is not None:
                    print(
                        f"Loaded conformal temperature ({cp.temperature:.4f}) differs from active "
                        f"Validation-Split T_cal ({temperature:.4f}); recalibrating."
                    )
        except Exception as e:
            print(f"Warning: could not load conformal calibration {cp_path} ({e}); recalibrating.")
    if need_calibration:
        # Calibrate only on the dedicated patient-disjoint calibration split.
        print(
            "No matching conformal calibration found. Calibrating on disjoint Calibration Split "
            f"with T_cal={temperature:.4f}..."
        )
        calibration_csv = "data/processed/calibration.csv"
        if os.path.exists(calibration_csv):
            cal_dataset = NIHChestXrayDataset(csv_file=calibration_csv, transform=eval_transform)
            cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=False)
            cp.calibrate(tier2, cal_loader, device, temperature=temperature)
            os.makedirs("outputs/results", exist_ok=True)
            torch.save(
                {
                    "q_hat": float(cp.q_hat) if cp.q_hat is not None else None,
                    "alpha": float(cp.alpha),
                    "temperature": float(cp.temperature),
                    "backbone": args.tier2_backbone,
                    "temperature_split": "val",
                    "conformal_split": "calibration",
                },
                cp_path,
            )
            print(f"Conformal q_hat saved to {cp_path} (q_hat={cp.q_hat:.4f})")

    # Load threshold (robust against an empty / corrupt restored JSON file)
    threshold_path = 'outputs/models/tier1_mobilenet_threshold.json'
    static_threshold = settings.model.confidence_threshold
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path) as f:
                t_data = json.load(f)
            static_threshold = t_data.get('optimal_threshold', static_threshold)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"Warning: could not parse {threshold_path} ({e}); using default threshold {static_threshold}.")
        
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
    tiered_system = TieredSystem(
        tier1, tier2, config_dict, conformal_predictor=cp, temperature=temperature
    )
    
    # Dataset
    image_size = settings.data.image_size
    test_csv = 'data/processed/test.csv'
    
    if not os.path.exists(test_csv):
        print("Error: Test set not found. Please run preprocessing first.")
        if mlflow_on:
            mlflow.end_run()
        return
        
    test_dataset = NIHChestXrayDataset(
        csv_file=test_csv,
        transform=ClassicalAugmentation(image_size=image_size, is_training=False).pipeline
    )
    
    # DataLoader (batch size 1 for sequential routing simulation)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Evaluating {len(test_dataset)} images...")
    
    y_true = []
    y_probs = []
    conformal_probs: list[list[float]] = []
    conformal_labels: list[int] = []
    tier2_count = 0
    flagged_count = 0
    total_inference_time = 0.0
    
    for images, labels, image_ids in tqdm(test_loader, desc="Evaluating System"):
        images = images.to(device)
        label = labels.item()
        image_id = image_ids[0] if isinstance(image_ids, (list, tuple)) else image_ids
        
        result = tiered_system.route(images, image_id=image_id)
        
        y_true.append(label)
        # 'Pneumothorax' is class 1. If prediction is Pneumothorax, confidence applies to class 1.
        prob = result.confidence if result.prediction == 'Pneumothorax' else 1.0 - result.confidence
            
        y_probs.append(prob)
        
        if result.tier_used == 2:
            tier2_count += 1
            if cp.q_hat is not None:
                conformal_probs.append([1.0 - prob, prob])
                conformal_labels.append(int(label))
            if result.flagged_for_review:
                flagged_count += 1
            
        total_inference_time += result.inference_time_ms
        
    metrics = compute_all_metrics(y_true, y_probs, threshold=0.5)
    
    percent_tier2 = (tier2_count / len(test_dataset)) * 100
    avg_inference_time = total_inference_time / len(test_dataset)
    conformal_extra: dict[str, object] = {}
    if cp.q_hat is not None and conformal_probs:
        cp_report = cp.evaluate_predictions(
            conformal_probs, conformal_labels, split_name=f"{args.run_name}_test"
        )
        conformal_extra = {
            "conformal_q_hat": cp_report.q_hat,
            "conformal_empirical_coverage": cp_report.marginal_coverage,
            "conformal_avg_set_size": cp_report.avg_set_size,
            "conformal_singleton_rate": cp_report.singleton_rate,
            "conformal_two_class_rate": cp_report.two_class_abstention_rate,
            "conformal_empty_fallback_rate": cp_report.empty_fallback_rate,
            "conformal_coverage_class0": cp_report.coverage_class0,
            "conformal_coverage_class1": cp_report.coverage_class1,
        }
    
    print("\n--- Evaluation Results ---")
    print(f"Percent Tier 2: {percent_tier2:.2f}%")
    print(f"Avg Inference Time: {avg_inference_time:.2f} ms")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    if mlflow_on:
        mlflow.log_params({
            "dynamic_threshold": args.dynamic_threshold,
            "base_threshold": static_threshold,
            "temperature_val_fitted": temperature,
        })
        mlflow.log_metrics({
            "percent_tier2": percent_tier2,
            "avg_inference_time_ms": avg_inference_time,
            **metrics
        })

    # Persist headline metrics to a JSON marker. This lets the ablation runner
    # detect that this evaluation is already complete (skip-existing) and keeps
    # the final numbers around after a Colab session reset wipes the MLflow store.
    try:
        os.makedirs('outputs/results', exist_ok=True)
        results_marker = os.path.join('outputs/results', f'{args.run_name}.json')
        with open(results_marker, 'w') as rf:
            json.dump(
                {
                    'run_name': args.run_name,
                    'tier2_backbone': args.tier2_backbone,
                    'dynamic_threshold': bool(args.dynamic_threshold),
                    'temperature_val_fitted': temperature,
                    'percent_tier2': percent_tier2,
                    'flagged_for_review_rate': flagged_count / max(1, len(test_dataset)),
                    'avg_inference_time_ms': avg_inference_time,
                    **conformal_extra,
                    'metrics': metrics,
                },
                rf,
                indent=2,
            )
        print(f"Saved evaluation results marker to {results_marker}")
    except Exception as dump_err:
        print(f"Warning: could not write results marker: {dump_err}")

    if mlflow_on:
        mlflow.end_run()

if __name__ == '__main__':
    main()
