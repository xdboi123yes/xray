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
from core.evaluation.metrics import (
    brier_score,
    compute_all_metrics,
    expected_calibration_error,
    negative_log_likelihood,
    wilson_ci,
)
from core.evaluation.stats import bootstrap_ci
from core.models.factory import ModelFactory
from core.models.tiered_system import TieredSystem
from core.uncertainty.conformal import ConformalPredictor
from infrastructure.data.chexpert_repository import CheXpertRepository


def setup_mlflow_local(
    experiment_name: str = "chest_xray_tiered", tracking_uri: str = "experiments/mlruns"
) -> bool:
    """Configure a local MLflow tracking store (best-effort)."""
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


def load_model_weights(model, weights_path, device):
    """Load weights into a model, unwrapping a training checkpoint if needed."""
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)


def main() -> None:
    """Run zero-shot evaluation loop over CheXpert validation/large external splits."""
    parser = argparse.ArgumentParser(description="Evaluate Tiered System on CheXpert Zero-Shot")
    parser.add_argument("--run-name", type=str, required=True, help="MLflow / output JSON run name")
    parser.add_argument(
        "--tier2-backbone",
        type=str,
        default="efficientnet_b4",
        choices=["efficientnet_b4", "ark_plus"],
        help="Tier 2 model backbone type",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/processed/chexpert_test.csv",
        help="Path to CheXpert evaluation CSV (e.g. chexpert_test.csv or chexpert_large_test.csv)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap iterations for 95% confidence intervals",
    )
    args = parser.parse_args()

    settings = get_settings()

    mlflow_on = setup_mlflow_local()
    if mlflow_on:
        try:
            mlflow.start_run(run_name=args.run_name)
        except Exception as exc:
            print(f"Warning: could not start MLflow run ({exc}); continuing without MLflow logging.")
            mlflow_on = False

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    tier1 = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    print(f"Loading Tier 2 model backbone: {args.tier2_backbone}...")
    tier2 = ModelFactory.create(args.tier2_backbone, num_classes=2, pretrained=False).to(device)

    tier1_path = "outputs/models/best_tier1_mobilenet.pth"
    tier2_path = f"outputs/models/best_tier2_{args.tier2_backbone}.pth"
    if args.tier2_backbone == "ark_plus" and not os.path.exists(tier2_path):
        fallback_path = "outputs/models/best_tier2_arkplus.pth"
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

    threshold_path = "outputs/models/tier1_mobilenet_threshold.json"
    static_threshold = settings.model.confidence_threshold
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path) as f:
                t_data = json.load(f)
            static_threshold = t_data.get("optimal_threshold", static_threshold)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            print(f"Warning: could not parse {threshold_path} ({e}); using default threshold {static_threshold}.")

    config_dict = {
        "model": {
            "confidence_threshold": static_threshold,
            "threshold_window_size": settings.model.threshold_window_size,
            "threshold_delta": settings.model.threshold_delta,
            "mc_dropout_passes": settings.model.mc_dropout_passes,
            "tta_passes": settings.model.tta_passes,
        },
        "data": {"image_size": settings.data.image_size},
    }
    tiered_system = TieredSystem(tier1, tier2, config_dict, conformal_predictor=cp)

    chexpert_csv = args.csv
    if not os.path.exists(chexpert_csv):
        print(
            f"[evaluate_chexpert] Error: CheXpert metadata splits not found at '{chexpert_csv}'."
        )
        if mlflow_on:
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
        if mlflow_on:
            mlflow.end_run()
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Evaluating tiered system zero-shot on {len(test_dataset)} CheXpert images ({chexpert_csv})...")

    y_true: list[int] = []
    y_probs: list[float] = []
    patient_ids: list[str] = []
    tier2_count = 0
    total_inference_time = 0.0

    for images, labels, image_ids in tqdm(test_loader, desc="Zero-Shot Evaluating"):
        images = images.to(device)
        label = int(labels.item())
        image_id = str(image_ids[0] if isinstance(image_ids, (list, tuple)) else image_ids)
        # Extract CheXpert patient folder name (e.g. patient64541) when present
        pid = next((p for p in image_id.split("/") if p.startswith("patient")), image_id)
        result = tiered_system.route(images, image_id=image_id)

        y_true.append(label)
        prob = result.confidence if result.prediction == "Pneumothorax" else 1.0 - result.confidence
        y_probs.append(float(prob))
        patient_ids.append(pid)
        if result.tier_used == 2:
            tier2_count += 1
        total_inference_time += result.inference_time_ms

    import numpy as np

    y_true_arr = np.asarray(y_true, dtype=int)
    y_probs_arr = np.asarray(y_probs, dtype=float)
    metrics = compute_all_metrics(y_true_arr, y_probs_arr, threshold=0.5)
    metrics["ece"] = expected_calibration_error(y_true_arr, y_probs_arr, n_bins=settings.evaluation.ece_bins)
    metrics["brier"] = brier_score(y_true_arr, y_probs_arr)
    metrics["nll"] = negative_log_likelihood(y_true_arr, y_probs_arr)

    # Wilson 95% CIs for exact finite-sample sensitivity/specificity/accuracy
    n_total = len(y_true_arr)
    n_pos = int((y_true_arr == 1).sum())
    n_neg = int((y_true_arr == 0).sum())
    y_pred_05 = (y_probs_arr >= 0.5).astype(int)
    tp = int(((y_pred_05 == 1) & (y_true_arr == 1)).sum())
    tn = int(((y_pred_05 == 0) & (y_true_arr == 0)).sum())
    correct = tp + tn

    ci_summary: dict[str, list[float]] = {
        "accuracy_wilson_95ci": list(wilson_ci(correct, n_total)),
        "sensitivity_wilson_95ci": list(wilson_ci(tp, n_pos)),
        "specificity_wilson_95ci": list(wilson_ci(tn, n_neg)),
    }
    if n_pos >= 2 and n_neg >= 2:
        auc_boot = bootstrap_ci(
            y_true_arr,
            y_probs_arr,
            metric="auc",
            n_iterations=args.n_bootstrap,
            patient_ids=patient_ids,
        )
        ci_summary["auc_patient_bootstrap_95ci"] = list(auc_boot["model1_ci"])

    # Prevalence-shift analysis: evaluate operating points at higher decision thresholds
    threshold_analysis = {
        f"thr_{thr:.2f}": compute_all_metrics(y_true_arr, y_probs_arr, threshold=thr)
        for thr in (0.50, 0.70, 0.80, 0.90)
    }

    percent_tier2 = (tier2_count / max(1, n_total)) * 100.0
    avg_inference_time = total_inference_time / max(1, n_total)

    print("\n--- Zero-Shot CheXpert Results ---")
    print(f"Cohort: {n_total} images ({len(set(patient_ids))} patients; {n_pos} positive, {100.0*n_pos/max(1,n_total):.2f}% prevalence)")
    print(f"Percent Routed to Tier 2: {percent_tier2:.2f}%")
    print(f"Avg Inference Time:       {avg_inference_time:.2f} ms")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    for k, ci in ci_summary.items():
        print(f"  {k}: [{ci[0]:.4f}, {ci[1]:.4f}]")

    if mlflow_on:
        mlflow.log_params(
            {
                "evaluation_dataset": chexpert_csv,
                "tier2_backbone": args.tier2_backbone,
                "base_threshold": static_threshold,
            }
        )
        mlflow.log_metrics(
            {"percent_tier2": percent_tier2, "avg_inference_time_ms": avg_inference_time, **metrics}
        )

    try:
        os.makedirs("outputs/results", exist_ok=True)
        results_marker = os.path.join("outputs/results", f"{args.run_name}.json")
        with open(results_marker, "w") as rf:
            json.dump(
                {
                    "run_name": args.run_name,
                    "tier2_backbone": args.tier2_backbone,
                    "evaluation_dataset": chexpert_csv,
                    "n_images": n_total,
                    "n_patients": len(set(patient_ids)),
                    "n_positive": n_pos,
                    "prevalence": n_pos / max(1, n_total),
                    "percent_tier2": percent_tier2,
                    "avg_inference_time_ms": avg_inference_time,
                    "metrics": metrics,
                    "confidence_intervals": ci_summary,
                    "threshold_analysis": threshold_analysis,
                },
                rf,
                indent=2,
            )
        print(f"Saved evaluation results marker to {results_marker}")
    except Exception as dump_err:
        print(f"Warning: could not write results marker: {dump_err}")

    if mlflow_on:
        mlflow.end_run()


if __name__ == "__main__":
    main()

