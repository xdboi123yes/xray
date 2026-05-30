"""Zero-shot out-of-distribution evaluation on CheXpert dataset.

Measures diagnostic generalizability and system throughput of tiered models
on CheXpert zero-shot, logging results to MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import get_settings
from core.augmentation.classical import ClassicalAugmentation
from core.evaluation.metrics import compute_all_metrics
from core.models.factory import ModelFactory
from core.models.tiered_system import TieredSystem
from core.uncertainty.conformal import ConformalPredictor
from infrastructure.data.chexpert_repository import CheXpertRepository


def setup_mlflow_local(experiment_name="chest_xray_tiered", tracking_uri="experiments/mlruns"):
    os.makedirs(tracking_uri, exist_ok=True)
    absolute_uri = f"file://{os.path.abspath(tracking_uri)}"
    mlflow.set_tracking_uri(absolute_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow initialized. Tracking URI: {absolute_uri}, Experiment: {experiment_name}")


def load_model_weights(model, weights_path, device):
    """Load weights into a model, unwrapping a training checkpoint if needed.

    Training saves checkpoints as {'epoch', 'best_metric', 'model_state_dict', ...},
    but load_state_dict expects the bare state dict. Passing the whole wrapper raised
    'Missing/Unexpected key(s)' RuntimeError and made A13/A14 fail within seconds.
    """
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)


def main() -> None:
    """Run zero-shot evaluation loop over CheXpert validation/test splits."""
    parser = argparse.ArgumentParser(description="Evaluate Tiered System on CheXpert Zero-Shot")
    parser.add_argument("--run-name", type=str, required=True, help="MLflow run name")
    parser.add_argument(
        "--tier2-backbone",
        type=str,
        default="efficientnet_b4",
        choices=["efficientnet_b4", "ark_plus"],
        help="Tier 2 model backbone type",
    )
    args = parser.parse_args()

    settings = get_settings()

    setup_mlflow_local()
    mlflow.start_run(run_name=args.run_name)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load Models
    tier1 = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    print(f"Loading Tier 2 model backbone: {args.tier2_backbone}...")
    tier2 = ModelFactory.create(args.tier2_backbone, num_classes=2, pretrained=False).to(device)

    tier1_path = "outputs/models/best_tier1_mobilenet.pth"
    tier2_path = f"outputs/models/best_tier2_{args.tier2_backbone}.pth"
    if args.tier2_backbone == 'ark_plus' and not os.path.exists(tier2_path):
        fallback_path = 'outputs/models/best_tier2_arkplus.pth'
        if os.path.exists(fallback_path):
            tier2_path = fallback_path

    if not os.path.exists(tier1_path) or not os.path.exists(tier2_path):
        print(
            f"Warning: Trained weights not found at '{tier1_path}' or '{tier2_path}'."
            f" Running zero-shot test using fallback initialized weights."
        )

    if os.path.exists(tier1_path):
        load_model_weights(tier1, tier1_path, device)
    if os.path.exists(tier2_path):
        load_model_weights(tier2, tier2_path, device)

    tier1.eval()
    tier2.eval()

    # Conformal Predictor
    cp_path = "outputs/results/q_hat.pt"
    cp = ConformalPredictor(alpha=1.0 - settings.evaluation.conformal_coverage)
    if os.path.exists(cp_path):
        try:
            cp.load(cp_path)
            print(f"Conformal calibration loaded (q_hat={cp.q_hat:.4f})")
        except Exception as e:
            print(f"Warning: could not load conformal calibration {cp_path} ({e}); skipping conformal coverage.")
    else:
        print("No conformal calibration found. Skipping conformal coverage.")

    # Load routing threshold (robust against an empty / corrupt restored JSON file)
    threshold_path = "outputs/models/tier1_mobilenet_threshold.json"
    static_threshold = settings.model.confidence_threshold
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path) as f:
                t_data = json.load(f)
            static_threshold = t_data.get("optimal_threshold", static_threshold)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"Warning: could not parse {threshold_path} ({e}); using default threshold {static_threshold}.")

    # Build dynamic/static config dict for TieredSystem compatibility
    config_dict = {
        "model": {
            "confidence_threshold": static_threshold,
            "threshold_window_size": settings.model.threshold_window_size,
            "threshold_delta": settings.model.threshold_delta,
            "mc_dropout_passes": settings.model.mc_dropout_passes,
            "tta_passes": settings.model.tta_passes,
        },
        "data": {
            "image_size": settings.data.image_size,
        }
    }

    # Initialize Tiered System
    tiered_system = TieredSystem(tier1, tier2, config_dict, conformal_predictor=cp)

    # Repository & Dataset Split
    chexpert_csv = "data/processed/chexpert_test.csv"
    if not os.path.exists(chexpert_csv):
        print(
            f"[evaluate_chexpert] Error: CheXpert metadata splits not found at '{chexpert_csv}'."
            f" Please run 'python scripts/download_chexpert_meta.py' first."
        )
        mlflow.end_run()
        return

    image_size = settings.data.image_size
    repository = CheXpertRepository(
        test_csv=chexpert_csv,
        transform=ClassicalAugmentation(image_size=image_size, is_training=False)._pipeline,
    )
    test_dataset = repository.get_test_dataset()

    if len(test_dataset) == 0:
        print("[evaluate_chexpert] Warning: CheXpert test dataset split is empty.")
        mlflow.end_run()
        return

    # DataLoader (batch size 1 for sequential tiered routing simulation)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Evaluating tiered system zero-shot on {len(test_dataset)} CheXpert images...")

    y_true = []
    y_probs = []
    tier2_count = 0
    total_inference_time = 0.0

    for images, labels, image_ids in tqdm(test_loader, desc="Zero-Shot Evaluating"):
        images = images.to(device)
        label = labels.item()
        image_id = image_ids[0] if isinstance(image_ids, (list, tuple)) else image_ids

        result = tiered_system.route(images, image_id=image_id)

        y_true.append(label)
        prob = result.confidence if result.prediction == "Pneumothorax" else 1.0 - result.confidence

        y_probs.append(prob)

        if result.tier_used == 2:
            tier2_count += 1

        total_inference_time += result.inference_time_ms

    # Compute diagnostics metrics
    metrics = compute_all_metrics(y_true, y_probs, threshold=0.5)

    percent_tier2 = (tier2_count / len(test_dataset)) * 100
    avg_inference_time = total_inference_time / len(test_dataset)

    print("\n--- Zero-Shot CheXpert Results ---")
    print(f"Percent Routed to Tier 2: {percent_tier2:.2f}%")
    print(f"Avg Inference Time:        {avg_inference_time:.2f} ms")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    mlflow.log_params(
        {
            "evaluation_dataset": "CheXpert",
            "tier2_backbone": args.tier2_backbone,
            "base_threshold": static_threshold,
        }
    )

    mlflow.log_metrics(
        {"percent_tier2": percent_tier2, "avg_inference_time_ms": avg_inference_time, **metrics}
    )

    # Persist headline metrics to a JSON marker. This lets the ablation runner
    # detect that this zero-shot evaluation is already complete (skip-existing) and
    # keeps the final numbers around after a Colab session reset wipes MLflow.
    try:
        os.makedirs("outputs/results", exist_ok=True)
        results_marker = os.path.join("outputs/results", f"{args.run_name}.json")
        with open(results_marker, "w") as rf:
            json.dump(
                {
                    "run_name": args.run_name,
                    "tier2_backbone": args.tier2_backbone,
                    "evaluation_dataset": "CheXpert",
                    "percent_tier2": percent_tier2,
                    "avg_inference_time_ms": avg_inference_time,
                    "metrics": metrics,
                },
                rf,
                indent=2,
            )
        print(f"Saved evaluation results marker to {results_marker}")
    except Exception as dump_err:
        print(f"Warning: could not write results marker: {dump_err}")

    mlflow.end_run()


if __name__ == "__main__":
    main()

