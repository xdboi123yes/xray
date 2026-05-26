"""Data Preprocessing Script.

Splits NIH ChestX-ray14 data into train/val/test sets.
Filters for Pneumothorax and No Finding classes.
Caps No Finding at 5:1 ratio relative to Pneumothorax.
Saves image_dir.txt for downstream use.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import get_settings


def split_data(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
    max_ratio: int = 5,
) -> None:
    """Split NIH data into train/val/test with class balancing.

    Args:
        csv_path: Path to Data_Entry_2017.csv.
        image_dir: Single directory containing all images.
        output_dir: Where to save train.csv, val.csv, test.csv.
        train_split: Train ratio.
        val_split: Val ratio.
        test_split: Test ratio.
        seed: Random seed.
        max_ratio: Max ratio of No Finding to Pneumothorax.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter: Pneumothorax (str.contains for multi-label) + No Finding (exact)
    mask_pneumo = df["Finding Labels"].str.contains("Pneumothorax", na=False)
    mask_normal = df["Finding Labels"] == "No Finding"
    df_filtered = df[mask_pneumo | mask_normal].copy()

    # Create binary label
    df_filtered["Label"] = df_filtered["Finding Labels"].apply(
        lambda x: 1 if "Pneumothorax" in str(x) else 0
    )

    print(
        f"Before balancing: {len(df_filtered)} total | "
        f"Pneumothorax: {(df_filtered['Label']==1).sum()} | "
        f"No Finding: {(df_filtered['Label']==0).sum()}"
    )

    # Cap No Finding at max_ratio:1 relative to Pneumothorax
    n_pneumo = (df_filtered["Label"] == 1).sum()
    max_normal = n_pneumo * max_ratio

    df_pneumo = df_filtered[df_filtered["Label"] == 1]
    df_normal = df_filtered[df_filtered["Label"] == 0]

    if len(df_normal) > max_normal:
        df_normal = df_normal.sample(n=max_normal, random_state=seed)
        print(f"Capping No Finding to {max_normal} (5:1 ratio)")

    df_filtered = pd.concat([df_pneumo, df_normal], ignore_index=True)

    # Check image existence
    if os.path.exists(image_dir):
        print(f"Checking image existence in {image_dir}...")
        existing_files = set(os.listdir(image_dir))
        df_filtered["exists"] = df_filtered["Image Index"].isin(existing_files)
        n_missing = (~df_filtered["exists"]).sum()
        if n_missing > 0:
            print(f"Warning: {n_missing} images not found in {image_dir}, skipping them.")
        df_filtered = df_filtered[df_filtered["exists"]].drop(columns=["exists"])
        print(f"Found {len(df_filtered)} images with verified existence.")
    else:
        print(f"Warning: {image_dir} not found. Proceeding without verifying image existence.")

    if len(df_filtered) == 0:
        print("Error: No images found to process.")
        return

    print(
        f"\nAfter balancing: {len(df_filtered)} total | "
        f"Pneumothorax: {(df_filtered['Label']==1).sum()} | "
        f"No Finding: {(df_filtered['Label']==0).sum()}"
    )

    # Stratified split: train vs (val + test)
    val_test_ratio = val_split + test_split
    train_df, val_test_df = train_test_split(
        df_filtered,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=df_filtered["Label"],
    )

    # val vs test
    test_ratio_of_val_test = test_split / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_ratio_of_val_test,
        random_state=seed,
        stratify=val_test_df["Label"],
    )

    # Save CSVs with required columns
    os.makedirs(output_dir, exist_ok=True)
    cols_to_save = ["Image Index", "Finding Labels", "Label"]

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(output_dir, f"{name}.csv")
        split_df[cols_to_save].to_csv(out_path, index=False)
        n_p = (split_df["Label"] == 1).sum()
        n_n = (split_df["Label"] == 0).sum()
        print(
            f"  {name:5s}: {len(split_df):6d} samples | Pneumo: {n_p} | Normal: {n_n} → {out_path}"
        )

    # Save image_dir to txt for downstream use
    image_dir_txt = os.path.join(output_dir, "image_dir.txt")
    with open(image_dir_txt, "w") as f:
        f.write(image_dir)
    print(f"\nImage directory saved to {image_dir_txt}: {image_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess NIH ChestX-ray14 data")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing all images (default: data/raw/images)",
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
        )


if __name__ == "__main__":
    main()
