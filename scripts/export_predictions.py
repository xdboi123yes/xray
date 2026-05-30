"""Export REAL per-image predictions for the analysis notebooks.

Runs the trained Tier 1 (MobileNetV2) and Tier 2 (EfficientNet / Ark+) models over
the held-out test split and writes ``outputs/results/tiered_predictions.csv`` with
genuine, per-image columns:

    image_id, y_true,
    tier1_prob, tier1_uncertainty,      # MobileNet softmax + MC-dropout variance
    tier2_prob,                         # deep backbone softmax
    escalated, tier_used, tiered_prob,  # routing decision + routed probability
    prob_uncal, prob_cal,               # raw vs temperature-scaled (fit on val)
    Patient Gender, Patient Age, View Position   # joined from NIH metadata

This is the single, honest data source the analysis notebooks load — they never
fabricate data. Run it on Colab after the models are trained/restored:

    python scripts/export_predictions.py --tier2-backbone ark_plus
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import get_settings
from core.augmentation.classical import ClassicalAugmentation
from core.models.factory import ModelFactory
from infrastructure.data.dataset import NIHChestXrayDataset


def _load_weights(model: torch.nn.Module, path: str, device: torch.device) -> bool:
    """Load a checkpoint, unwrapping a training dict if needed. Returns True on success."""
    if not os.path.exists(path):
        return False
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    return True


def _pneumo_prob(logits: torch.Tensor) -> np.ndarray:
    """Softmax probability of the Pneumothorax class (index 1)."""
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def _fit_temperature(logit_diff: np.ndarray, y: np.ndarray) -> float:
    """Fit a 1-parameter temperature on binary logits (log p/(1-p)) by NLL grid search."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 5.0, 91):
        p = 1.0 / (1.0 + np.exp(-logit_diff / t))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def _run_split(
    csv_file: str,
    tier1: torch.nn.Module,
    tier2: torch.nn.Module,
    device: torch.device,
    image_size: int,
    mc_passes: int,
    esc_threshold: float,
) -> pd.DataFrame:
    """Run both tiers over a split and return a per-image dataframe."""
    transform = ClassicalAugmentation(image_size=image_size, is_training=False)._pipeline
    loader = DataLoader(
        NIHChestXrayDataset(csv_file=csv_file, transform=transform),
        batch_size=1,
        shuffle=False,
    )

    rows: list[dict[str, object]] = []
    tier1.eval()
    tier2.eval()
    with torch.no_grad():
        for images, labels, image_ids in loader:
            images = images.to(device)
            image_id = image_ids[0] if isinstance(image_ids, (list, tuple)) else str(image_ids)

            t1_logits = tier1(images)
            t1_prob = float(_pneumo_prob(t1_logits)[0])

            # MC-dropout epistemic uncertainty for Tier 1
            if hasattr(tier1, "mc_forward"):
                _, var = tier1.mc_forward(images, T=mc_passes)
                t1_unc = float(var.mean().detach().cpu().item())
            else:
                t1_unc = float("nan")

            t2_prob = float(_pneumo_prob(tier2(images))[0])

            # Routing: escalate to Tier 2 when Tier 1 is not confident enough, i.e. its top-class
            # probability is below the configured confidence threshold (the real router's rule).
            escalated = max(t1_prob, 1.0 - t1_prob) < esc_threshold
            tiered_prob = t2_prob if escalated else t1_prob

            rows.append(
                {
                    "image_id": str(image_id),
                    "y_true": int(labels.item()),
                    "tier1_prob": t1_prob,
                    "tier1_uncertainty": t1_unc,
                    "tier2_prob": t2_prob,
                    "escalated": bool(escalated),
                    "tier_used": 2 if escalated else 1,
                    "tiered_prob": tiered_prob,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entrypoint for exporting real per-image predictions."""
    parser = argparse.ArgumentParser(description="Export real per-image tiered predictions")
    parser.add_argument(
        "--tier2-backbone",
        default="ark_plus",
        choices=["efficientnet_b4", "ark_plus"],
    )
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Tier 1 confidence below which a case escalates to Tier 2 "
        "(defaults to config model.confidence_threshold).",
    )
    parser.add_argument("--output", default="outputs/results/tiered_predictions.csv")
    args = parser.parse_args()

    settings = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = settings.data.image_size
    esc_threshold = (
        args.confidence_threshold
        if args.confidence_threshold is not None
        else settings.model.confidence_threshold
    )

    tier1 = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    tier2 = ModelFactory.create(args.tier2_backbone, num_classes=2, pretrained=False).to(device)

    t1_path = "outputs/models/best_tier1_mobilenet.pth"
    t2_path = f"outputs/models/best_tier2_{args.tier2_backbone}.pth"
    if args.tier2_backbone == "ark_plus" and not os.path.exists(t2_path):
        alt = "outputs/models/best_tier2_arkplus.pth"
        if os.path.exists(alt):
            t2_path = alt
    if not _load_weights(tier1, t1_path, device) or not _load_weights(tier2, t2_path, device):
        raise SystemExit(
            f"Trained weights not found ({t1_path} / {t2_path}). "
            "Train or restore the models before exporting real predictions."
        )

    test_csv = "data/processed/test.csv"
    val_csv = "data/processed/val.csv"
    if not os.path.exists(test_csv):
        raise SystemExit(f"{test_csv} not found — run preprocessing first.")

    print("Running both tiers over the test split...")
    df = _run_split(test_csv, tier1, tier2, device, image_size, args.mc_passes, esc_threshold)

    # Temperature scaling: fit T on the validation split, apply to the test tiered probs.
    if os.path.exists(val_csv):
        print("Fitting temperature on the validation split...")
        val_df = _run_split(
            val_csv, tier1, tier2, device, image_size, args.mc_passes, esc_threshold
        )
        eps = 1e-6
        v_logit = np.log(np.clip(val_df["tiered_prob"], eps, 1 - eps)) - np.log(
            np.clip(1 - val_df["tiered_prob"], eps, 1 - eps)
        )
        temperature = _fit_temperature(v_logit.to_numpy(), val_df["y_true"].to_numpy())
    else:
        temperature = 1.0
    print(f"Temperature = {temperature:.3f}")

    eps = 1e-6
    t_logit = np.log(np.clip(df["tiered_prob"], eps, 1 - eps)) - np.log(
        np.clip(1 - df["tiered_prob"], eps, 1 - eps)
    )
    df["prob_uncal"] = df["tiered_prob"]
    df["prob_cal"] = 1.0 / (1.0 + np.exp(-t_logit / temperature))

    # Join NIH demographics (Patient Gender / Age / View Position) by image id.
    meta_path = "data/raw/Data_Entry_2017.csv"
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        keep = [
            c
            for c in ["Image Index", "Patient Gender", "Patient Age", "View Position"]
            if c in meta.columns
        ]
        df = df.merge(
            meta[keep].rename(columns={"Image Index": "image_id"}), on="image_id", how="left"
        )
    else:
        print(f"Note: {meta_path} not found — demographic columns will be empty.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} real per-image predictions to {args.output}")
    print(f"  Escalated to Tier 2: {int(df['escalated'].sum())} / {len(df)}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
