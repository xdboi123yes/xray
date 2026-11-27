#!/usr/bin/env python3
"""Unified, GPU-saturating evaluator for the full A1-A15 ablation matrix.

Loads the REAL trained weights for one ablation row, runs inference on the NIH
test split, computes genuine headline metrics (auc_roc / accuracy / ece + the
full confusion-derived set) and writes ``outputs/results/<run_name>.json`` --
the honest, durable source that ``build_ablation_json.py`` compiles into the
dashboard table.

Speed: MC-dropout + TTA are VECTORISED. Dropout masks are independent per batch
element, so tiling one image T times into a single forward yields T Monte-Carlo
samples in parallel; TTA views are batched the same way. The effective batch is
streamed through the model in micro-batches of ``--micro-batch`` images, which is
the single knob that controls VRAM usage (raise it on an A100, lower it on a T4).
``--amp`` runs the forwards in fp16 on CUDA. Every knob is dynamic so you can tune
throughput to the GPU without touching code.

It complements (does not replace) ``evaluate_tiered.py`` (A13) and
``evaluate_chexpert.py`` (A14). Configuration metadata is driven by
``scripts/ablation_spec.py``.

Usage (on Colab, after weights are trained/restored and data is present)::

    python scripts/evaluate_ablation.py --ablation A10 --micro-batch 512 --amp
    python scripts/evaluate_ablation.py --all --batch-size 32 --micro-batch 1024
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass

import torch
import torchvision.transforms as tv_transforms  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Imported after the sys.path injection so the sibling spec module and the
# project packages both resolve when run as a bare script.
from ablation_spec import TIER1_WEIGHTS, AblationSpec, get_spec
from tcrp.config import get_settings
from tcrp.data.augmentation import ClassicalAugmentation
from tcrp.data.datasets import NIHChestXrayDataset
from tcrp.evaluation.metrics import (
    brier_score,
    calibration_slope_intercept,
    classwise_ece,
    compute_all_metrics,
    expected_calibration_error,
    negative_log_likelihood,
)
from tcrp.models import ModelFactory
from tcrp.uncertainty.calibration import ModelWithTemperature
from tcrp.uncertainty.conformal import ConformalPredictor

VAL_CSV = "data/processed/val.csv"
TEST_CSV = "data/processed/test.csv"
CALIBRATION_CSV = "data/processed/calibration.csv"
RESULTS_DIR = "outputs/results"
CLASSES = ["No Finding", "Pneumothorax"]

# Same test-time augmentations the model classes use, so TTA stays consistent.
_TTA_TRANSFORM = tv_transforms.Compose(
    [
        tv_transforms.RandomHorizontalFlip(p=0.5),
        tv_transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    ]
)


@dataclass
class EvalConfig:
    """Dynamic, GPU-tunable evaluation knobs (all overridable from the CLI)."""

    batch_size: int = 16
    micro_batch: int = 256
    num_workers: int = 2
    amp: bool = True
    mc_passes: int | None = None
    tta_passes: int | None = None
    limit: int | None = None


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
    transform = ClassicalAugmentation(image_size=image_size, is_training=False).pipeline
    return NIHChestXrayDataset(csv_file=TEST_CSV, transform=transform)


def _enable_mc_dropout(model: torch.nn.Module) -> None:
    """Activate dropout for MC sampling while keeping BatchNorm in eval mode.

    This is the correct MC-dropout setup: dropout is stochastic but normalization
    uses the trained running statistics (not per-batch statistics).
    """
    model.eval()
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def _autocast_enabled(cfg: EvalConfig, device: torch.device) -> bool:
    """AMP is used for fp16 forwards on CUDA only."""
    return cfg.amp and device.type == "cuda"


def _forward_ctx(cfg: EvalConfig, device: torch.device) -> contextlib.AbstractContextManager[None]:
    """Return an fp16 autocast context on CUDA when AMP is on, else a no-op."""
    if _autocast_enabled(cfg, device):
        return torch.autocast("cuda")
    return contextlib.nullcontext()


def predict_probs_plain(
    model: torch.nn.Module,
    images: torch.Tensor,
    cfg: EvalConfig,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return batched softmax probabilities [B, 2] with a single forward."""
    with torch.no_grad(), _forward_ctx(cfg, device):
        logits = model(images.to(device))
    return torch.softmax(logits.float() / max(temperature, 1e-6), dim=1).cpu()


def predict_probs_uncertain_decomposed(
    model: torch.nn.Module,
    images: torch.Tensor,
    mc_passes: int,
    tta_passes: int,
    cfg: EvalConfig,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean probabilities [B, 2] plus Law-of-Total-Variance decomposition.

    Decomposes total variance across K TTA views and T MC passes into:
      epistemic_var = E_k[Var_t(p | a_k)]   (within-view MC-Dropout variance)
      tta_var       = Var_k[E_t(p | a_k)]   (between-view TTA sensitivity)
      total_var     = epistemic_var + tta_var
    """
    batch = images.shape[0]
    if mc_passes > 1:
        _enable_mc_dropout(model)
    reps_per_forward = max(1, cfg.micro_batch // batch)
    temp = max(temperature, 1e-6)

    views = [images] + [_TTA_TRANSFORM(images) for _ in range(max(0, tta_passes - 1))]
    per_view_tensors: list[torch.Tensor] = []
    with torch.no_grad():
        for view in views:
            view_dev = view.to(device)
            view_chunks: list[torch.Tensor] = []
            done = 0
            while done < mc_passes:
                k = min(reps_per_forward, mc_passes - done)
                chunk = view_dev.repeat(k, 1, 1, 1)
                with _forward_ctx(cfg, device):
                    logits = model(chunk)
                probs = torch.softmax(logits.float() / temp, dim=1).reshape(k, batch, 2).cpu()
                view_chunks.append(probs)
                done += k
            per_view_tensors.append(torch.cat(view_chunks, dim=0))  # [T, B, 2]
    model.eval()

    # Shape: [K, T, B, 2]
    stacked = torch.stack(per_view_tensors, dim=0)
    view_means = stacked.mean(dim=1)  # [K, B, 2]
    view_vars = stacked.var(dim=1, unbiased=False)  # [K, B, 2]

    mean_probs = view_means.mean(dim=0)  # [B, 2]
    epistemic_var = view_vars.mean(dim=0)  # [B, 2]
    tta_var = view_means.var(dim=0, unbiased=False)  # [B, 2]
    total_var = epistemic_var + tta_var
    return mean_probs, total_var, epistemic_var, tta_var


def predict_probs_uncertain(
    model: torch.nn.Module,
    images: torch.Tensor,
    mc_passes: int,
    tta_passes: int,
    cfg: EvalConfig,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return MC-dropout + TTA mean probabilities [B, 2] using batched forwards."""
    mean_probs, _, _, _ = predict_probs_uncertain_decomposed(
        model, images, mc_passes, tta_passes, cfg, device, temperature=temperature
    )
    return mean_probs


def _resolve_passes(spec: AblationSpec, cfg: EvalConfig) -> tuple[int, int]:
    """Resolve effective (mc_passes, tta_passes), honoring CLI overrides."""
    mc = cfg.mc_passes if cfg.mc_passes is not None else spec.mc_passes
    tta = cfg.tta_passes if cfg.tta_passes is not None else spec.tta_passes
    return mc, tta


def fit_temperature_on_val(
    model: torch.nn.Module, device: torch.device, image_size: int, cfg: EvalConfig
) -> float:
    """Fit post-hoc temperature scalar strictly on the validation split (val.csv)."""
    if not os.path.exists(VAL_CSV):
        return 1.0
    transform = ClassicalAugmentation(image_size=image_size, is_training=False).pipeline
    val_loader = DataLoader(
        NIHChestXrayDataset(csv_file=VAL_CSV, transform=transform),
        batch_size=max(cfg.batch_size, 64),
        num_workers=cfg.num_workers,
    )
    wrapper = ModelWithTemperature(model)
    wrapper.set_temperature(val_loader, device)
    temp = float(wrapper.temperature.item())
    return temp if temp > 0.05 else 1.0


def evaluate_single(
    spec: AblationSpec, device: torch.device, image_size: int, cfg: EvalConfig
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
    mc_passes, tta_passes = _resolve_passes(spec, cfg)
    batch_size = cfg.batch_size if use_uncertainty else max(cfg.batch_size, 64)
    loader = DataLoader(
        build_dataset(image_size), batch_size=batch_size, num_workers=cfg.num_workers
    )

    y_true: list[int] = []
    y_probs: list[float] = []
    epistemic_vars: list[float] = []
    tta_vars: list[float] = []
    seen = 0
    start = time.time()
    for images, labels, _ in tqdm(loader, desc=f"{spec.ablation_id} {spec.mode}"):
        if use_uncertainty:
            mean_probs, _, epi_var, tta_var = predict_probs_uncertain_decomposed(
                model, images, mc_passes, tta_passes, cfg, device
            )
            epistemic_vars.extend(epi_var[:, 1].tolist())
            tta_vars.extend(tta_var[:, 1].tolist())
        else:
            mean_probs = predict_probs_plain(model, images, cfg, device)
        y_probs.extend(mean_probs[:, 1].tolist())
        y_true.extend(int(v) for v in labels.tolist())
        seen += len(labels)
        if cfg.limit is not None and seen >= cfg.limit:
            break
    elapsed_ms = (time.time() - start) * 1000.0

    result: dict[str, object] = {
        "weights_used": weights_used,
        "avg_inference_time_ms": elapsed_ms / max(1, len(y_true)),
        "y_true": y_true,
        "y_probs": y_probs,
    }
    if epistemic_vars:
        result["mean_epistemic_var"] = float(sum(epistemic_vars) / len(epistemic_vars))
        result["mean_tta_var"] = float(sum(tta_vars) / len(tta_vars))
    return result


def calibrate_conformal(
    spec: AblationSpec,
    tier2: torch.nn.Module,
    device: torch.device,
    image_size: int,
    cfg: EvalConfig,
    temperature: float = 1.0,
) -> ConformalPredictor | None:
    """Calibrate split-conformal q_hat strictly on the disjoint calibration split.

    Applies ``temperature`` (fitted on the separate validation split) to Tier-2 logits
    before computing nonconformity scores so validation and conformal calibration data
    remain completely disjoint (Reviewer 2 Comment 5).
    """
    settings = get_settings()
    cp = ConformalPredictor(alpha=1.0 - settings.evaluation.conformal_coverage)
    if not os.path.exists(CALIBRATION_CSV):
        print(f"[{spec.ablation_id}] {CALIBRATION_CSV} missing - skipping conformal coverage.")
        return None
    transform = ClassicalAugmentation(image_size=image_size, is_training=False).pipeline
    cal_loader = DataLoader(
        NIHChestXrayDataset(csv_file=CALIBRATION_CSV, transform=transform),
        batch_size=max(cfg.batch_size, 64),
        num_workers=cfg.num_workers,
    )
    print(
        f"[{spec.ablation_id}] Calibrating conformal predictor strictly on disjoint "
        f"{CALIBRATION_CSV} (temperature={temperature:.3f} from val split)..."
    )

    class _ScaledModel(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module, temp: float) -> None:
            super().__init__()
            self.inner = inner
            self.temp = max(temp, 1e-6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x) / self.temp

    cp.calibrate(_ScaledModel(tier2, temperature).to(device), cal_loader, device)
    return cp


def _routing_decisions(
    tier1_confidences: list[float], static_threshold: float, window: int, delta: float
) -> list[bool]:
    """Replicate TieredSystem's (optionally dynamic) routing as a cheap CPU walk.

    Returns a per-image mask that is True when the case escalates to Tier 2.
    """
    threshold = static_threshold
    recent: list[float] = []
    escalated: list[bool] = []
    for conf in tier1_confidences:
        recent.append(conf)
        if len(recent) > window:
            recent.pop(0)
            mean_conf = sum(recent) / window
            if mean_conf < 0.65:
                threshold = max(0.5, threshold - delta)
            elif mean_conf > 0.85:
                threshold = min(0.95, threshold + delta)
        escalated.append(conf < threshold)
    return escalated


def evaluate_tiered(
    spec: AblationSpec, device: torch.device, image_size: int, cfg: EvalConfig
) -> dict[str, object]:
    """Evaluate a tiered ablation row: batched Tier 1, then vectorised Tier 2."""
    settings = get_settings()

    tier1 = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    load_model_weights(tier1, TIER1_WEIGHTS, device)
    tier2 = ModelFactory.create(spec.backbone, num_classes=2, pretrained=False).to(device)
    weights_used = resolve_tier2_weights(spec)
    load_model_weights(tier2, weights_used, device)
    tier1.eval()
    tier2.eval()

    # Fit post-hoc temperature strictly on val.csv before disjoint conformal calibration
    temperature = fit_temperature_on_val(tier2, device, image_size, cfg) if spec.conformal else 1.0
    cp = (
        calibrate_conformal(spec, tier2, device, image_size, cfg, temperature=temperature)
        if spec.conformal
        else None
    )

    static_threshold = settings.model.confidence_threshold
    threshold_path = "outputs/models/tier1_mobilenet_threshold.json"
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path) as fh:
                static_threshold = json.load(fh).get("optimal_threshold", static_threshold)
        except (json.JSONDecodeError, ValueError, OSError):
            pass

    mc_passes, tta_passes = _resolve_passes(spec, cfg)
    dataset: NIHChestXrayDataset | Subset = build_dataset(image_size)
    if cfg.limit is not None:
        dataset = Subset(dataset, range(min(cfg.limit, len(dataset))))

    start = time.time()

    # Pass 1: Tier 1 screening over every image (batched, cheap).
    t1_loader = DataLoader(
        dataset, batch_size=max(cfg.batch_size, 64), num_workers=cfg.num_workers
    )
    t1_prob1: list[float] = []
    t1_conf: list[float] = []
    y_true: list[int] = []
    for images, labels, _ in tqdm(t1_loader, desc=f"{spec.ablation_id} tier1"):
        probs = predict_probs_plain(tier1, images, cfg, device)
        t1_prob1.extend(probs[:, 1].tolist())
        t1_conf.extend(probs.max(dim=1).values.tolist())
        y_true.extend(int(v) for v in labels.tolist())

    total = len(y_true)
    delta = settings.model.threshold_delta if spec.dynamic else 0.0
    escalated = _routing_decisions(
        t1_conf, static_threshold, settings.model.threshold_window_size, delta
    )
    escalated_idx = [i for i, esc in enumerate(escalated) if esc]

    # Pass 2: Tier 2 deep model (MC + TTA) on the escalated subset only.
    final_prob1 = list(t1_prob1)
    conformal_hits = 0
    conformal_hits_c0 = 0
    conformal_n_c0 = 0
    conformal_hits_c1 = 0
    conformal_n_c1 = 0
    conformal_empty_fallbacks = 0
    conformal_set_sizes: list[int] = []
    epistemic_vars: list[float] = []
    tta_vars: list[float] = []
    flagged_count = 0
    if escalated_idx:
        t2_loader = DataLoader(
            Subset(dataset, escalated_idx),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        pos = 0
        for images, _, _ in tqdm(t2_loader, desc=f"{spec.ablation_id} tier2"):
            mean_probs, total_var, epi_var, tta_var = predict_probs_uncertain_decomposed(
                tier2, images, mc_passes, tta_passes, cfg, device, temperature=temperature
            )
            for b in range(mean_probs.shape[0]):
                global_idx = escalated_idx[pos]
                pos += 1
                final_prob1[global_idx] = float(mean_probs[b, 1])
                epistemic_vars.append(float(epi_var[b, 1]))
                tta_vars.append(float(tta_var[b, 1]))
                pred_cls = int(mean_probs[b].argmax().item())
                pred_var = float(total_var[b, pred_cls])
                pred_set: list[str] | None = None
                if cp is not None and cp.q_hat is not None:
                    raw_threshold = 1.0 - cp.q_hat
                    if float(mean_probs[b].max().item()) < raw_threshold:
                        conformal_empty_fallbacks += 1
                    pred_set = cp.predict_set(mean_probs[b : b + 1])
                    conformal_set_sizes.append(len(pred_set))
                    true_lbl = y_true[global_idx]
                    hit = CLASSES[true_lbl] in pred_set
                    if hit:
                        conformal_hits += 1
                    if true_lbl == 0:
                        conformal_n_c0 += 1
                        if hit:
                            conformal_hits_c0 += 1
                    else:
                        conformal_n_c1 += 1
                        if hit:
                            conformal_hits_c1 += 1
                if (pred_var > 0.1) or (pred_set is not None and len(pred_set) > 1):
                    flagged_count += 1

    elapsed_ms = (time.time() - start) * 1000.0
    extra: dict[str, object] = {
        "weights_used": weights_used,
        "temperature_val_fitted": temperature,
        "percent_tier2": (len(escalated_idx) / total) * 100 if total else 0.0,
        "avg_inference_time_ms": elapsed_ms / max(1, total),
        "flagged_for_review_rate": (flagged_count / total) if total else 0.0,
        "y_true": y_true,
        "y_probs": final_prob1,
    }
    if epistemic_vars:
        extra["mean_epistemic_var"] = float(sum(epistemic_vars) / len(epistemic_vars))
        extra["mean_tta_var"] = float(sum(tta_vars) / len(tta_vars))
    if conformal_set_sizes:
        n_cp = len(conformal_set_sizes)
        extra["conformal_q_hat"] = cp.q_hat if cp is not None else None
        extra["conformal_empirical_coverage"] = conformal_hits / n_cp
        extra["conformal_avg_set_size"] = sum(conformal_set_sizes) / n_cp
        extra["conformal_singleton_rate"] = sum(1 for s in conformal_set_sizes if s == 1) / n_cp
        extra["conformal_two_class_rate"] = sum(1 for s in conformal_set_sizes if s == 2) / n_cp
        extra["conformal_empty_fallback_rate"] = conformal_empty_fallbacks / n_cp
        extra["conformal_coverage_class0"] = (
            conformal_hits_c0 / conformal_n_c0 if conformal_n_c0 else float("nan")
        )
        extra["conformal_coverage_class1"] = (
            conformal_hits_c1 / conformal_n_c1 if conformal_n_c1 else float("nan")
        )
    return extra


def evaluate_one(ablation_id: str, cfg: EvalConfig) -> None:
    """Evaluate a single ablation row and persist its results JSON marker."""
    spec = get_spec(ablation_id)
    if spec.mode == "external":
        raise SystemExit(
            f"[{ablation_id}] is produced by a dedicated script "
            f"({'evaluate_chexpert.py' if spec.dataset == 'chexpert' else 'evaluate_tiered.py'}); "
            "this evaluator handles A1-A12 and A15 only."
        )
    if not os.path.exists(TEST_CSV):
        raise SystemExit(f"{TEST_CSV} not found - run preprocessing first (Colab).")

    device = pick_device()
    settings = get_settings()
    image_size = settings.data.image_size
    mc_passes, tta_passes = _resolve_passes(spec, cfg)
    print(
        f"[{spec.ablation_id}] {spec.name} | mode={spec.mode} device={device} "
        f"batch={cfg.batch_size} micro={cfg.micro_batch} mc={mc_passes} tta={tta_passes} "
        f"amp={_autocast_enabled(cfg, device)}"
    )

    if spec.mode == "tiered":
        outcome = evaluate_tiered(spec, device, image_size, cfg)
    else:
        outcome = evaluate_single(spec, device, image_size, cfg)

    y_true = outcome.pop("y_true")
    y_probs = outcome.pop("y_probs")
    metrics = compute_all_metrics(y_true, y_probs, threshold=0.5)
    metrics["ece"] = expected_calibration_error(y_true, y_probs, n_bins=settings.evaluation.ece_bins)
    metrics.update(classwise_ece(y_true, y_probs, n_bins=settings.evaluation.ece_bins))
    metrics["brier"] = brier_score(y_true, y_probs)
    metrics["nll"] = negative_log_likelihood(y_true, y_probs)
    cal_slope, cal_intercept = calibration_slope_intercept(y_true, y_probs)
    metrics["calibration_slope"] = cal_slope
    metrics["calibration_intercept"] = cal_intercept

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


def build_config(args: argparse.Namespace) -> EvalConfig:
    """Build the dynamic EvalConfig from parsed CLI arguments."""
    return EvalConfig(
        batch_size=args.batch_size,
        micro_batch=args.micro_batch,
        num_workers=args.num_workers,
        amp=args.amp,
        mc_passes=args.mc_passes,
        tta_passes=args.tta_passes,
        limit=args.limit,
    )


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
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Images per dataset batch (outer loop)"
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=256,
        help="Max images per GPU forward (the VRAM knob; raise on A100, lower on T4)",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fp16 autocast on CUDA (--no-amp to disable)",
    )
    parser.add_argument(
        "--mc-passes", type=int, default=None, help="Override MC-dropout passes for every row"
    )
    parser.add_argument(
        "--tta-passes", type=int, default=None, help="Override TTA passes for every row"
    )
    args = parser.parse_args()
    cfg = build_config(args)

    if args.all:
        from ablation_spec import ABLATION_SPECS

        for spec in ABLATION_SPECS:
            if spec.mode == "external":
                continue
            try:
                evaluate_one(spec.ablation_id, cfg)
            except SystemExit as exc:
                print(f"[{spec.ablation_id}] skipped: {exc}")
    else:
        evaluate_one(args.ablation, cfg)


if __name__ == "__main__":
    main()
