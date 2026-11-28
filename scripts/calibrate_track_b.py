"""Disjoint Track B calibration: Validation-Split temperature, then Calibration-Split conformal q_hat.

``temperature``: fit post-hoc temperature scaling on ``data/processed/val.csv`` only and write
``outputs/results/temperature.pt`` plus a 15-bin reliability diagram.
``conformal``: verify val/calibration patient disjointness, load the Validation-Split temperature,
and fit split-conformal ``q_hat`` on temperature-scaled ``data/processed/calibration.csv``
probabilities, logging marginal/per-class coverage and singleton, two-class, and empty-fallback rates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tcrp.config import get_settings
from tcrp.data.augmentation import ClassicalAugmentation
from tcrp.data.datasets import NIHChestXrayDataset
from tcrp.data.splits import verify_patient_disjointness
from tcrp.models import ModelFactory
from tcrp.uncertainty.calibration import (
    calibrate_temperature,
    compute_ece,
    plot_reliability_diagram,
)
from tcrp.uncertainty.conformal import ConformalPredictor

PROTOCOL_VERSION = "patient-disjoint-v1"
BACKBONE = "efficientnet_b4"
WEIGHTS = "outputs/models/best_tier2_efficientnet.pth"
VAL_CSV = "data/processed/val.csv"
CAL_CSV = "data/processed/calibration.csv"
TEMPERATURE_PATH = "outputs/results/temperature.pt"
Q_HAT_PATH = "outputs/results/q_hat.pt"
SUMMARY_PATH = "outputs/results/conformal_calibration_summary.json"


def _load_model(backbone: str, weights: str, device: torch.device) -> torch.nn.Module:
    model = ModelFactory.create(backbone, num_classes=2, pretrained=False)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) else state)
    return model.to(device).eval()


def _loader(csv: str, num_workers: int) -> DataLoader:
    settings = get_settings()
    tf = ClassicalAugmentation(image_size=settings.data.image_size, is_training=False).pipeline
    return DataLoader(NIHChestXrayDataset(csv_file=csv, transform=tf), batch_size=32, shuffle=False,
                      num_workers=num_workers)


def fit_temperature(device: torch.device, num_workers: int) -> None:
    model = _load_model(BACKBONE, WEIGHTS, device)
    loader = _loader(VAL_CSV, num_workers)
    _, raw_t = calibrate_temperature(model, loader, device)
    if not (math.isfinite(float(raw_t)) and float(raw_t) > 0.05):
        raise RuntimeError(f"Temperature fit diverged (T={raw_t}); refusing to write temperature.pt")
    temperature = float(raw_t)

    probs_pre, probs_post, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            logits = model(x)
            probs_pre.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            probs_post.append(torch.softmax(logits / temperature, dim=1)[:, 1].cpu().numpy())
            labels.append(np.asarray(y))
    pre, post, lab = np.concatenate(probs_pre), np.concatenate(probs_post), np.concatenate(labels)
    ece_pre, ece_post = compute_ece(pre, lab, n_bins=15), compute_ece(post, lab, n_bins=15)

    os.makedirs(os.path.dirname(TEMPERATURE_PATH), exist_ok=True)
    torch.save(
        {"temperature": temperature, "ece": float(ece_post), "ece_before": float(ece_pre), "split": "val",
         "n_samples": int(len(lab)), "backbone": BACKBONE, "weights_path": WEIGHTS, "protocol": PROTOCOL_VERSION},
        TEMPERATURE_PATH,
    )
    figures = get_settings().paths.figures
    plot_reliability_diagram(post, lab, n_bins=15, save_path=os.path.join(figures, "reliability_tier2_eff.png"))
    print(f"Validation-Split temperature T={temperature:.4f} (n={len(lab)}): "
          f"ECE {ece_pre:.4f} -> {ece_post:.4f}")


def fit_conformal(device: torch.device, num_workers: int) -> None:
    verify_patient_disjointness({"val": pd.read_csv(VAL_CSV), "calibration": pd.read_csv(CAL_CSV)})
    temp_state = torch.load(TEMPERATURE_PATH, map_location="cpu", weights_only=False)
    if not isinstance(temp_state, dict) or temp_state.get("split") != "val":
        raise RuntimeError("temperature.pt must be the Validation-Split temperature (split='val')")
    temperature = float(temp_state["temperature"])
    backbone = str(temp_state["backbone"])
    model = _load_model(backbone, str(temp_state["weights_path"]), device)

    cp = ConformalPredictor(alpha=1.0 - get_settings().evaluation.conformal_coverage)
    q_hat = cp.calibrate(model, _loader(CAL_CSV, num_workers), device, temperature=temperature)
    torch.save(
        {"q_hat": float(q_hat), "alpha": float(cp.alpha), "temperature": temperature, "backbone": backbone,
         "temperature_split": "val", "conformal_split": "calibration", "protocol": PROTOCOL_VERSION},
        Q_HAT_PATH,
    )
    report = cp.last_calibration_report
    if report is None:
        raise RuntimeError("ConformalPredictor.calibrate produced no calibration report")
    summary = {"protocol": PROTOCOL_VERSION, "backbone": backbone, "temperature_split": "val",
               "conformal_split": "calibration", "temperature_before_conformal": True, **report.to_dict()}
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Calibration-Split conformal q_hat={q_hat:.4f} (T={temperature:.4f}): "
          f"coverage={report.marginal_coverage:.4f} [c0={report.coverage_class0:.4f}, c1={report.coverage_class1:.4f}], "
          f"singleton={report.singleton_rate:.4f}, two_class={report.two_class_abstention_rate:.4f}, "
          f"empty_fallback={report.empty_fallback_rate:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["temperature", "conformal"])
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (fit_temperature if args.stage == "temperature" else fit_conformal)(device, args.num_workers)


if __name__ == "__main__":
    main()
