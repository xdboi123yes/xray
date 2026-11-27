"""CLI script to generate multi-batch FID-gated Synthetic Augmentation Batches.

Iterates across Patient-Disjoint Partition training pneumothorax seeds (both
isolated and comorbid positive cases by default), generates Stable Diffusion
img2img variations, evaluates each batch with Inception-v3 FID against real
training pneumothorax radiographs, logs batch metrics, routes images into
accepted/rejected directories, persists accepted metadata and a qualitative
comparison grid, and supports crash-safe checkpoint/resume.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tcrp.config import Settings, get_settings
from tcrp.data.synthetic import (
    MultiBatchGenerationSummary,
    SyntheticBatchPipeline,
    SyntheticGenerationConfig,
)


def _resolve_default_image_dir(settings: Settings) -> str:
    image_dir_txt = os.path.join(settings.paths.data_processed, "image_dir.txt")
    if os.path.exists(image_dir_txt):
        with open(image_dir_txt, encoding="utf-8") as fh:
            resolved = fh.read().strip()
            if resolved:
                return resolved
    return settings.paths.image_dir


def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    """Build argument parser for configurable multi-batch synthetic generation."""
    cfg = settings or get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Generate multi-batch FID-gated Synthetic Augmentation Batches "
            "from Patient-Disjoint Partition training pneumothorax seeds."
        )
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=os.path.join(cfg.paths.data_processed, "train.csv"),
        help="Path to Patient-Disjoint Partition training split CSV.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing real NIH radiographs (defaults to image_dir.txt / config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=cfg.paths.data_synthetic,
        help="Output directory for accepted/, rejected/, fid_log.csv, metadata, and resume state.",
    )
    parser.add_argument(
        "--seed-scope",
        type=str,
        choices=["all", "isolated"],
        default="all",
        help="Seed scope: 'all' (isolated + comorbid Pneumothorax) or 'isolated' only.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of real training pneumothorax seeds per FID-evaluated batch.",
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=3,
        help="Number of synthetic variations generated per real seed radiograph.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.training.seed,
        help="Base random seed for deterministic variation generation.",
    )
    parser.add_argument(
        "--fid-threshold",
        type=float,
        default=cfg.augmentation.fid_threshold,
        help="Maximum Inception-v3 FID score to accept a generated batch (default: 50.0).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.3,
        help="Stable Diffusion img2img denoising strength.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Optional cap on total real seeds processed (default: None = all training seeds).",
    )
    parser.add_argument(
        "--fid-reference-count",
        type=int,
        default=None,
        help="Optional cap on real training pneumothorax reference images used for FID.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from last completed batch in resume_state.json (use --no-resume for fresh run).",
    )
    parser.add_argument(
        "--qualitative-grid-path",
        type=str,
        default=None,
        help="Optional output path for the real-vs-synthetic qualitative comparison grid PNG.",
    )
    parser.add_argument(
        "--qualitative-grid-samples",
        type=int,
        default=4,
        help="Maximum number of real/synthetic pairs shown in the qualitative comparison grid.",
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments for synthetic generation."""
    return build_parser(settings=settings).parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    pipeline: SyntheticBatchPipeline | None = None,
    settings: Settings | None = None,
) -> MultiBatchGenerationSummary | None:
    """Execute multi-batch FID-gated synthetic generation from CLI arguments."""
    cfg = settings or get_settings()
    args = parse_args(argv, settings=cfg)

    if not cfg.data.synthetic_augmentation:
        print("Synthetic augmentation is disabled in configuration settings.")
        return None

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"{args.train_csv} not found. Run preprocess.py first.")

    real_images_dir = args.image_dir or _resolve_default_image_dir(cfg)
    active_pipeline = pipeline or SyntheticBatchPipeline(
        fid_threshold=args.fid_threshold,
        settings=cfg,
    )

    gen_config = SyntheticGenerationConfig(
        train_csv=args.train_csv,
        real_images_dir=real_images_dir,
        output_dir=args.output_dir,
        seed_scope=args.seed_scope,
        batch_size=args.batch_size,
        num_variations=args.num_variations,
        seed=args.seed,
        fid_threshold=args.fid_threshold,
        strength=args.strength,
        max_seeds=args.max_seeds,
        fid_reference_count=args.fid_reference_count,
        resume=args.resume,
        qualitative_grid_path=args.qualitative_grid_path,
        qualitative_grid_samples=args.qualitative_grid_samples,
    )

    summary = active_pipeline.run_multi_batch(gen_config)
    print(
        f"Synthetic generation complete: seeds={summary.processed_seeds}/{summary.total_seeds} "
        f"(skipped={summary.skipped_seeds}), batches={summary.batches_executed}/{summary.total_batches} "
        f"(accepted={summary.batches_accepted}, rejected={summary.batches_rejected}), "
        f"images accepted={summary.accepted_images}, rejected={summary.rejected_images}"
    )
    print(f"  FID log:  {summary.fid_log_path}")
    print(f"  Metadata: {summary.metadata_csv_path}")
    if summary.qualitative_grid_path:
        print(f"  Grid:     {summary.qualitative_grid_path}")
    return summary


if __name__ == "__main__":
    main()
