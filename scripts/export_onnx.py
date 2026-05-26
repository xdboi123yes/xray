"""Compile PyTorch diagnostic weights to standard ONNX.

Command-line script to trace and export Tier 1 or Tier 2 model graphs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path to support standalone execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from core.models.factory import ModelFactory
from infrastructure.export.onnx_exporter import ONNXModelExporter
import core.models.tier1_mobilenet  # registers mobilenet_v2
import core.models.tier2_efficientnet  # registers efficientnet_b4
import core.models.tier2_ark  # registers ark_plus


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tier1", "tier2"],
        help="Model tier to export (tier1 or tier2).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply post-training quantization.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output filename path.",
    )
    args = parser.parse_args()

    # Load appropriate model backbone registry key
    if args.model == "tier1":
        backbone_key = "mobilenet_v2"
        weights_filename = "best_tier1_mobilenet.pth"
        default_out = "outputs/models/tier1_mobilenet.onnx"
    else:
        backbone_key = "efficientnet_b4"
        weights_filename = "best_tier2_efficientnet.pth"
        default_out = "outputs/models/tier2_efficientnet.onnx"

    weights_path = Path("outputs/models") / weights_filename
    output_path = args.output or default_out

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Instantiating model backbone: {backbone_key}...")
    model = ModelFactory.create(backbone_key)

    if weights_path.exists():
        print(f"Loading trained weights from: {weights_path}...")
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    else:
        print(f"Weights file not found at {weights_path}. Exporting random-initialized model.")

    print(f"Exporting model to ONNX format: {output_path}...")
    try:
        ONNXModelExporter.export(model, output_path)
        print("Model successfully compiled and saved to disk.")
    except Exception as exc:
        print(f"Export failed: {exc}")
        raise


if __name__ == "__main__":
    main()
