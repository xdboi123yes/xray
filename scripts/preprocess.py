"""CLI entrypoint for building four-way Patient-Disjoint Partition splits for NIH ChestX-ray14."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tcrp.config import get_settings
from tcrp.data.splits import build_patient_disjoint_splits


def split_data(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
    max_ratio: int = 5,
    min_images: int | None = None,
    min_positives: int | None = None,
) -> None:
    """Build four-way Patient-Disjoint Partition splits and print summary counts."""
    summary = build_patient_disjoint_splits(
        csv_path=csv_path,
        image_dir=image_dir,
        output_dir=output_dir,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        max_ratio=max_ratio,
    )
    total_samples = sum(c.samples for c in summary.splits.values())
    total_positives = sum(c.pneumothorax for c in summary.splits.values())
    if min_images is not None and total_samples < min_images:
        raise ValueError(
            f"Incomplete dataset in '{image_dir}': matched {total_samples} images, "
            f"expected at least {min_images}."
        )
    if min_positives is not None and total_positives < min_positives:
        raise ValueError(
            f"Incomplete dataset in '{image_dir}': matched {total_positives} pneumothorax "
            f"images, expected at least {min_positives}."
        )
    for name, counts in summary.splits.items():
        print(
            f"  {name:11s}: {counts.samples:6d} samples ({counts.patients:5d} patients) | "
            f"Pneumo: {counts.pneumothorax} | Normal: {counts.no_finding}"
        )
    print(f"Verified zero patient overlap: {summary.pairwise_patient_overlaps}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess NIH ChestX-ray14 data")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing all images (default: data/raw/images)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=None,
        help="Fail fast if total matched images across splits fall below this floor.",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=None,
        help="Fail fast if total matched Pneumothorax images fall below this floor.",
    )
    args = parser.parse_args()

    config = get_settings()
    raw_csv = "data/raw/Data_Entry_2017.csv"
    processed_dir = "data/processed/"
    image_dir = args.image_dir or config.paths.image_dir

    if not os.path.exists(raw_csv):
        print(f"Error: {raw_csv} not found.")
        print(
            "Please download the NIH ChestX-ray14 dataset and place 'Data_Entry_2017.csv' in 'data/raw/'."
        )
    else:
        split_data(
            csv_path=raw_csv,
            image_dir=image_dir,
            output_dir=processed_dir,
            train_split=config.data.train_split,
            val_split=config.data.val_split,
            test_split=config.data.test_split,
            seed=config.training.seed,
            min_images=args.min_images,
            min_positives=args.min_positives,
        )


if __name__ == "__main__":
    main()
