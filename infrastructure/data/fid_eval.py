"""Fréchet Inception Distance (FID) evaluator for synthetic X-rays.

Computes fidelity scores between real and generated image sets to filter poor samples.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import structlog
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths  # type: ignore[import-untyped]

from config.settings import Settings, get_settings

log = structlog.get_logger(__name__)


class FIDEvaluator:
    """Computes FID values and filters synthetic batches against threshold gates."""

    def __init__(
        self,
        config: dict[str, Any] | Settings | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initializes the FIDEvaluator.

        Args:
            config: Project settings (Pydantic Settings or raw dict).
            device: Active computation device (CUDA/MPS/CPU).
        """
        if config is None:
            self.settings = get_settings()
        elif isinstance(config, dict):
            self.settings = Settings(**config)
        else:
            self.settings = config

        self.device = device or torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fid_threshold = self.settings.augmentation.fid_threshold

    def compute_fid(
        self,
        real_dir: str,
        synthetic_dir: str,
        batch_size: int = 32,
    ) -> float:
        """Computes Fréchet Inception Distance (FID) between real and synthetic folders.

        Args:
            real_dir: Path to directory containing real clinical X-Rays.
            synthetic_dir: Path to directory containing generated synthetic X-Rays.
            batch_size: Evaluation batch size.

        Returns:
            The calculated FID score float (lower is more realistic).
        """
        log.info("Computing FID between real images and synthetic batch...")

        if not os.path.exists(real_dir):
            raise FileNotFoundError(f"Real image directory not found: {real_dir}")
        if not os.path.exists(synthetic_dir):
            raise FileNotFoundError(f"Synthetic image directory not found: {synthetic_dir}")

        device_str = str(self.device)

        # Ensure enough files exist for covariance matrix calculations
        num_synthetic = len(
            [
                f
                for f in os.listdir(synthetic_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        if num_synthetic < 2:
            log.warning(
                "Warning: FID requires more than 1 image to compute covariance. "
                "Skipping FID calculation."
            )
            return float("inf")

        fid_value: float = calculate_fid_given_paths(
            paths=[real_dir, synthetic_dir],
            batch_size=min(batch_size, num_synthetic),
            device=device_str,
            dims=2048,
            num_workers=0,
        )
        return fid_value

    def evaluate_and_filter_batch(
        self,
        real_dir: str,
        synthetic_batch_dir: str,
        accepted_dir: str,
        rejected_dir: str,
    ) -> tuple[bool, float, list[str]]:
        """Filters a synthetic batch by evaluating its FID against standard threshold.

        Args:
            real_dir: Path to directory containing real clinical images.
            synthetic_batch_dir: Temporary directory containing the synthetic batch.
            accepted_dir: Target directory to move accepted synthetic images.
            rejected_dir: Target directory to move rejected synthetic images.

        Returns:
            A tuple of (True if accepted, calculated FID score float, list of moved file paths).
        """
        os.makedirs(accepted_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

        fid_score = self.compute_fid(real_dir, synthetic_batch_dir)
        log.info(
            f"Batch FID Score: {fid_score:.4f} "
            f"(Threshold: {self.fid_threshold})"
        )

        accepted = fid_score <= self.fid_threshold
        target_dir = accepted_dir if accepted else rejected_dir

        moved_files: list[str] = []
        for filename in os.listdir(synthetic_batch_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                src = os.path.join(synthetic_batch_dir, filename)
                dst = os.path.join(target_dir, filename)
                shutil.move(src, dst)
                moved_files.append(dst)

        return accepted, fid_score, moved_files
