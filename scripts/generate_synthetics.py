"""CLI script to generate synthetic Pneumothorax images.

Generates high-fidelity chest radiographs using Stable Diffusion conditioning, 
evaluating batch quality using Frechet Inception Distance (FID) threshold gates.
"""

from __future__ import annotations

import os
import shutil

import pandas as pd
from tqdm import tqdm

from config.settings import get_settings
from infrastructure.data.fid_eval import FIDEvaluator
from infrastructure.data.synthetic_gen import SyntheticGenerator


def main() -> None:
    config = get_settings()

    if not config.data.synthetic_augmentation:
        print("Synthetic augmentation is disabled in configuration settings.")
        return

    train_csv = "data/processed/train.csv"
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(train_csv)
    # Filter for Pneumothorax class to use as conditioning images
    df_pneumo = df[df["Finding Labels"] == "Pneumothorax"]

    if len(df_pneumo) == 0:
        print("No Pneumothorax images found in training set.")
        return

    print(f"Found {len(df_pneumo)} real Pneumothorax images for conditioning.")

    # We will process a small test batch to avoid taking hours on a local machine
    batch_size = 5  # Number of real images to process in this run
    num_variations = 3  # Synthetics per real image

    # Read image_dir from image_dir.txt if available
    image_dir_txt = "data/processed/image_dir.txt"
    if os.path.exists(image_dir_txt):
        with open(image_dir_txt) as f:
            real_images_dir = f.read().strip()
    else:
        real_images_dir = config.paths.image_dir

    synthetic_base_dir = "data/synthetic"

    temp_batch_dir = os.path.join(synthetic_base_dir, "temp_batch")
    accepted_dir = os.path.join(synthetic_base_dir, "accepted")
    rejected_dir = os.path.join(synthetic_base_dir, "rejected")

    os.makedirs(temp_batch_dir, exist_ok=True)
    os.makedirs(accepted_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    # Initialize Generator and Evaluator
    print("Initializing Stable Diffusion...")
    try:
        generator = SyntheticGenerator(config)
        evaluator = FIDEvaluator(config)
    except Exception as e:
        print(f"Error initializing models: {e}")
        return

    synthetic_csv_path = os.path.join(synthetic_base_dir, "synthetic_metadata.csv")
    accepted_records = []
    if os.path.exists(synthetic_csv_path):
        accepted_records = pd.read_csv(synthetic_csv_path).to_dict("records")

    total_to_process = min(batch_size, len(df_pneumo))
    print(f"Generating synthetic variations for {total_to_process} images...")

    for i in tqdm(range(total_to_process), desc="Generating"):
        row = df_pneumo.iloc[i]
        real_img_path = os.path.join(real_images_dir, row["Image Index"])

        if not os.path.exists(real_img_path):
            continue

        synthetics = generator.generate(real_img_path, num_variations=num_variations)

        for j, synth_img in enumerate(synthetics):
            base_name = os.path.splitext(row["Image Index"])[0]
            synth_filename = f"synth_{base_name}_{j}.png"
            synth_path = os.path.join(temp_batch_dir, synth_filename)
            synth_img.save(synth_path)

    # Evaluate Batch
    print("\nEvaluating generated batch with FID...")
    # Create a temporary dir of real pneumothorax images for fair FID comparison
    temp_real_dir = os.path.join(synthetic_base_dir, "temp_real")
    os.makedirs(temp_real_dir, exist_ok=True)

    # Use up to 50 real images for the FID baseline comparison
    for _, row in df_pneumo.head(50).iterrows():
        src = os.path.join(real_images_dir, row["Image Index"])
        if os.path.exists(src):
            shutil.copy(src, os.path.join(temp_real_dir, row["Image Index"]))

    accepted, fid_score, moved_files = evaluator.evaluate_and_filter_batch(
        temp_real_dir, temp_batch_dir, accepted_dir, rejected_dir
    )

    if accepted:
        print(f"--> Batch ACCEPTED! FID: {fid_score:.2f} <= {evaluator.fid_threshold}")
        for filepath in moved_files:
            filename = os.path.basename(filepath)
            accepted_records.append({"Image Index": filename, "Finding Labels": "Pneumothorax"})
    else:
        print(f"--> Batch REJECTED. FID: {fid_score:.2f} > {evaluator.fid_threshold}")

    # Save metadata
    if accepted_records:
        pd.DataFrame(accepted_records).to_csv(synthetic_csv_path, index=False)
        print(f"Total accepted synthetic images tracked: {len(accepted_records)}")

    # Cleanup temp dirs
    shutil.rmtree(temp_real_dir)
    if os.path.exists(temp_batch_dir) and not os.listdir(temp_batch_dir):
        os.rmdir(temp_batch_dir)


if __name__ == "__main__":
    main()
