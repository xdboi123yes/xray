#!/usr/bin/env python3
"""Unified evaluator for the full A1-A15 ablation matrix.

Loads the REAL trained weights for one ablation row, runs inference on the NIH
test split, computes genuine headline metrics (auc_roc / accuracy / ece + the
full confusion-derived set) and writes ``outputs/results/<run_name>.json`` --
the honest, durable source that ``build_ablation_json.py`` compiles into the
dashboard table.

It complements (does not replace) ``evaluate_tiered.py`` (A13) and
``evaluate_chexpert.py`` (A14): those two own their rows and run first; this
script fills the remaining 13 rows (A1-A12, A15) that previously had no
computed metrics at all.

Every configuration is driven by ``scripts/ablation_spec.py`` so the dashboard
metadata and the evaluation settings can never drift apart.

Usage (on Colab, after weights are trained/restored and data is present)::

    python scripts/evaluate_ablation.py --ablation A6
    python scripts/evaluate_ablation.py --all          # every non-external row
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Imports below run after the sys.path injection above so the sibling spec
# module and the project packages both resolve when run as a bare script.
from ablation_spec import TIER1_WEIGHTS, AblationSpec, get_spec

from config.settings import get_settings
from core.augmentation.classical import ClassicalAugmentation
from core.evaluation.metrics import compute_all_metrics, expected_calibration_error
from core.models.factory import ModelFactory
from core.models.tiered_system import TieredSystem
from core.uncertainty.conformal import ConformalPredictor
from infrastructure.data.dataset import NIHChestXrayDataset

TEST_CSV = "data/processed/test.csv"
VAL_CSV = "data/processed/val.csv"
RESULTS_DIR = "outputs/results"


def pick_device() -> torch.device:
    """Return CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_weights(model: torch.nn.Module, weights_path: str, device: torch.device) -> None:
    """Load weights, unwrapping a training checkpoint dict if necessary."""
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)


def resolve_tier2_weights(spec: AblationSpec) -> str:
    """Return the first existing Tier 2 weight path, or raise a clear error."""
    candidates = spec.tier2_weight_candidates()
    for path in candidates:
        if os.path.exists(path):
            return path
    raise SystemExit(
        f"[{spec.ablation_id}] Tier 2 weights not found. Tried: {candidates}. "
        "Train or restore the model before evaluating this ablation."
    )


def build_dataset(image_size: int) -> NIHChestXrayDataset:
    """Build the NIH test dataset with eval-time (non-training) transforms."""
    transform = ClassicalAugmentation(image_size=image_size, is_training=False)._pipeline
    return NIHChestXrayDataset(csv_file=TEST_CSV, transform=transform)


def evaluate_single(
    spec: AblationSpec, device: torch.device, image_size: int, limit: int | None
) -> dict[str, object]:
    """Evaluate one standalone model (Tier 1 or a single deep backbone)."""
    if spec.mode == "single_tier1":
        model = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
        load_model_weights(model, TIER1_WEIGHTS, device)
        weights_used = TIER1_WEIGHTS
    else:
        model = ModelFactory.create(spec.backbone, num_classes=2, pretrained=False).to(device)
        weights_used = resolve_tier2_weights(spec)
        load_model_weights(model, weights_used, device)
    model.eval()

    use_uncertainty = spec.mc or spec.tta
    batch_size = 1 if use_uncertainty else 64
    loader = DataLoader(build_dataset(image_size), batch_size=batch_size, shuffle=False)

    y_true: list[int] = []
    y_probs: list[float] = []
    seen = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=f"{spec.ablation_id} {spec.mode}"):
            images = images.to(device)
            if use_uncertainty:
                mean_probs, _ = model.mc_tta_forward(
                    images, T=spec.mc_passes, n_augments=spec.tta_passes
                )
                probs = mean_probs[:, 1]
            else:
                probs = torch.softmax(model(images), dim=1)[:, 1]
            y_probs.extend(probs.detach().cpu().tolist())
            y_true.extend(int(v) for v in labels.tolist())
            seen += len(labels)
            if limit is not None and seen >= limit:
                break

    return {
        "weights_used": weights_used,
        "y_true": y_true,
        "y_probs": y_probs,
    }


def calibrate_conformal(
    spec: AblationSpec, tier2: torch.nn.Module, device: torch.device, image_size: int
) -> ConformalPredictor | None:
    """Calibrate a per-ablation conformal predictor on the validation split.

    Calibration uses the exact Tier 2 weights under evaluation and is kept
    in-memory so the shared ``q_hat.pt`` (owned by A13) is never overwritten.
    """
    settings = get_settings()
    cp = ConformalPredictor(alpha=1.0 - settings.evaluation.conformal_coverage)
    if not os.path.exists(VAL_CSV):
        print(f"[{spec.ablation_id}] {VAL_CSV} missing — skipping conformal coverage.")
        return None
    transform = ClassicalAugmentation(image_size=image_size, is_training=False)._pipeline
    cal_loader = DataLoader(
        NIHChestXrayDataset(csv_file=VAL_CSV, transform=transform), batch_size=32, shuffle=False
    )
    print(f"[{spec.ablation_id}] Calibrating conformal predictor on the validation split...")
    cp.calibrate(tier2, cal_loader, device)
    return cp


def evaluate_tiered(
    spec: AblationSpec, device: torch.device, image_size: int, limit: int | None
) -> dict[str, object]:
    """Evaluate the full TieredSystem for one tiered ablation row."""
    settings = get_settings()

    tier1 = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    load_model_weights(tier1, TIER1_WEIGHTS, device)
    tier2 = ModelFactory.create(spec.backbone, num_classes=2, pretrained=False).to(device)
    weights_used = resolve_tier2_weights(spec)
    load_model_weights(tier2, weights_used, device)
    tier1.eval()
    tier2.eval()

    cp = calibrate_conformal(spec, tier2, device, image_size) if spec.conformal else None

    # Routing threshold: read the optimal Tier 1 threshold marker when available.
    static_threshold = settings.model.confidence_threshold
    threshold_path = "outputs/models/tier1_mobilenet_threshold.json"
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path) as fh:
                static_threshold = json.load(fh).get("optimal_threshold", static_threshold)
        except (json.JSONDecodeError, ValueError, OSError):
            pass

    config_dict = {
        "model": {
            "confidence_threshold": static_threshold,
            "threshold_window_size": settings.model.threshold_window_size,
            "threshold_delta": settings.model.threshold_delta if spec.dynamic else 0.0,
            "mc_dropout_passes": spec.mc_passes,
            "tta_passes": spec.tta_passes,
        },
        "data": {"image_size": image_size},
    }
    system = TieredSystem(tier1, tier2, config_dict, conformal_predictor=cp)

    loader = DataLoader(build_dataset(image_size), batch_size=1, shuffle=False)
    y_true: list[int] = []
    y_probs: list[float] = []
    tier2_count = 0
    total_time_ms = 0.0
    conformal_hits = 0
    conformal_set_sizes: list[int] = []

    for images, labels, image_ids in tqdm(loader, desc=f"{spec.ablation_id} tiered"):
        images = images.to(device)
        label = int(labels.item())
        image_id = image_ids[0] if isinstance(image_ids, (list, tuple)) else str(image_ids)
        result = system.route(images, image_id=image_id)

        prob = result.confidence if result.prediction == "Pneumothorax" else 1.0 - result.confidence
        y_true.append(label)
        y_probs.append(prob)
        if result.tier_used == 2:
            tier2_count += 1
        total_time_ms += result.inference_time_ms

        if result.conformal_set is not None:
            conformal_set_sizes.append(len(result.conformal_set))
            true_label = system.classes[label]
            if true_label in result.conformal_set:
                conformal_hits += 1

        if limit is not None and len(y_true) >= limit:
            break

    n = len(y_true)
    extra: dict[str, object] = {
        "weights_used": weights_used,
        "percent_tier2": (tier2_count / n) * 100 if n else 0.0,
        "avg_inference_time_ms": total_time_ms / n if n else 0.0,
        "y_true": y_true,
        "y_probs": y_probs,
    }
    if conformal_set_sizes:
        extra["conformal_empirical_coverage"] = conformal_hits / len(conformal_set_sizes)
        extra["conformal_avg_set_size"] = sum(conformal_set_sizes) / len(conformal_set_sizes)
    return extra


def evaluate_one(ablation_id: str, limit: int | None) -> None:
    """Evaluate a single ablation row and persist its results JSON marker."""
    spec = get_spec(ablation_id)
    if spec.mode == "external":
        raise SystemExit(
            f"[{ablation_id}] is produced by a dedicated script "
            f"({'evaluate_chexpert.py' if spec.dataset == 'chexpert' else 'evaluate_tiered.py'}); "
            "this evaluator handles A1-A12 and A15 only."
        )
    if not os.path.exists(TEST_CSV):
        raise SystemExit(f"{TEST_CSV} not found — run preprocessing first (Colab).")

    device = pick_device()
    settings = get_settings()
    image_size = settings.data.image_size
    print(f"[{spec.ablation_id}] {spec.name} | mode={spec.mode} device={device}")

    if spec.mode == "tiered":
        outcome = evaluate_tiered(spec, device, image_size, limit)
    else:
        outcome = evaluate_single(spec, device, image_size, limit)

    y_true = outcome.pop("y_true")
    y_probs = outcome.pop("y_probs")
    metrics = compute_all_metrics(y_true, y_probs, threshold=0.5)
    metrics["ece"] = expected_calibration_error(y_true, y_probs)

    print(f"\n--- {spec.ablation_id} results ---")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    payload: dict[str, object] = {
        "run_name": spec.run_name,
        "ablation_id": spec.ablation_id,
        "mode": spec.mode,
        "tier2_backbone": spec.backbone,
        "dataset": spec.dataset,
        "dynamic_threshold": spec.dynamic,
        **outcome,
        "metrics": metrics,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{spec.run_name}.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved {out_path}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate one or all A1-A15 ablation rows")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ablation", type=str, help="Ablation id, e.g. A6")
    group.add_argument(
        "--all", action="store_true", help="Evaluate every non-external row (A1-A12, A15)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate at most N images (smoke test)"
    )
    args = parser.parse_args()

    if args.all:
        from ablation_spec import ABLATION_SPECS

        for spec in ABLATION_SPECS:
            if spec.mode == "external":
                continue
            try:
                evaluate_one(spec.ablation_id, args.limit)
            except SystemExit as exc:
                print(f"[{spec.ablation_id}] skipped: {exc}")
    else:
        evaluate_one(args.ablation, args.limit)


if __name__ == "__main__":
    main()
