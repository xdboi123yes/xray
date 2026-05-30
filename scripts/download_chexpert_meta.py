"""Helper script to set up CheXpert metadata splits for zero-shot testing.

Simulates/downloads high-fidelity mock demographic and disease labels for out-of-distribution
generalization evaluation, enabling instant dry-runs without requiring local 400GB downloads.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.integrity import guard_mock


def main() -> None:
    """Prepare/simulate CheXpert validation and test database metadata records."""
    parser = argparse.ArgumentParser(description="Set up CheXpert Metadata splits")
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/chexpert_test.csv",
        help="Path where the processed CheXpert validation metadata will be saved.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of mock CheXpert instances to simulate.",
    )
    args = parser.parse_args()

    print("Preparing CheXpert metadata splits...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if os.path.exists(args.output_path):
        print(f"CheXpert metadata already exists at '{args.output_path}'. Skipping setup.")
        return

    # This writes SIMULATED metadata with RANDOM labels — dry-run scaffolding only. For a real
    # zero-shot CheXpert evaluation use the notebook's "CheXpert Zero-Shot Test Set" cell
    # (Kaggle ashery/chexpert). Refuse to fabricate unless mock data is explicitly allowed.
    guard_mock("[download_chexpert_meta] simulated CheXpert metadata with random labels")

    # Simulate high-fidelity clinical and demographic labels
    np.random.seed(42)
    sample_ids = [f"patient{i:05d}/study1/view1_frontal.png" for i in range(1, args.n_samples + 1)]
    
    # 40% positive Pneumothorax rate, 40% clean No Finding, 20% other (which will get filtered)
    pneumo_labels = (np.random.rand(args.n_samples) < 0.40).astype(float)
    no_finding_labels = np.zeros(args.n_samples)
    
    # Ensure mutually exclusive labels for clean binary evaluation
    for idx, is_p in enumerate(pneumo_labels):
        if is_p == 0.0 and np.random.rand() < 0.66:
            no_finding_labels[idx] = 1.0

    genders = np.random.choice(["Male", "Female"], size=args.n_samples)
    ages = np.random.randint(18, 90, size=args.n_samples)

    chexpert_df = pd.DataFrame({
        "Path": [f"CheXpert-v1.0-small/valid/{sid}" for sid in sample_ids],
        "Sex": genders,
        "Age": ages,
        "Frontal/Lateral": "Frontal",
        "AP/PA": np.random.choice(["AP", "PA"], size=args.n_samples),
        "No Finding": no_finding_labels,
        "Cardiomegaly": np.nan,
        "Edema": np.nan,
        "Consolidation": np.nan,
        "Pneumonia": np.nan,
        "Atelectasis": np.nan,
        "Pneumothorax": pneumo_labels,
        "Pleural Effusion": np.nan,
        "Support Devices": np.nan,
    })

    # Save to the target output directory
    chexpert_df.to_csv(args.output_path, index=False)
    print(f"Successfully simulated {args.n_samples} CheXpert metadata records at: {args.output_path}")


if __name__ == "__main__":
    main()
