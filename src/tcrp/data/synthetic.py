"""FID-gated Synthetic Augmentation Batch pipeline with injectable adapters.

Consolidates Stable Diffusion img2img radiograph synthesis and Inception-v3 FID
quality filtering behind two narrow true-external seams (:class:`GeneratorAdapter`
and :class:`FIDEvaluatorAdapter`), supporting multi-batch iteration across all
Patient-Disjoint Partition training pneumothorax seeds, crash-safe checkpoint
and resume, FID logging, accepted metadata tracking, and qualitative comparison
grid generation without requiring GPU weights in unit tests.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import structlog
import torch
from PIL import Image

from tcrp.config import Settings, get_settings, guard_mock

log = structlog.get_logger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

FID_LOG_COLUMNS = [
    "batch",
    "seed_count",
    "generated_count",
    "fid",
    "threshold",
    "accepted",
    "decision",
]
METADATA_COLUMNS = [
    "Image Index",
    "Finding Labels",
    "Seed Image",
    "Patient ID",
    "Variation",
    "Batch",
]


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _atomic_write_text(path: str, content: str) -> None:
    """Write text atomically via a temporary sibling file and ``os.replace``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    os.replace(tmp_path, path)


def _atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV atomically via ``os.replace``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _stage_reference_image(src: str, dst: str) -> None:
    """Stage a real reference image into a temp folder via symlink with copy fallback."""
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


class GeneratorAdapter(Protocol):
    """Seam for generating synthetic pneumothorax radiograph variations."""

    def generate(
        self,
        init_image_path: str,
        num_variations: int = 3,
        strength: float = 0.3,
        severity: str | None = None,
        location: str | None = None,
    ) -> list[Image.Image]:
        """Generate ``num_variations`` synthetic images from a real seed radiograph."""
        ...


class FIDEvaluatorAdapter(Protocol):
    """Seam for computing Inception-v3 Fréchet Inception Distance between image folders."""

    def compute_fid(self, real_dir: str, synthetic_dir: str, batch_size: int = 32) -> float:
        """Return FID score (lower is closer to real radiographs; +inf on failure)."""
        ...


@dataclass(frozen=True)
class BatchFIDRecord:
    """Outcome of FID quality gating on one Synthetic Augmentation Batch."""

    accepted: bool
    fid_score: float
    threshold: float
    moved_files: list[str]


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    """Configuration for multi-batch FID-gated synthetic generation with resume."""

    train_csv: str
    real_images_dir: str
    output_dir: str = "data/synthetic"
    seed_scope: Literal["all", "isolated"] = "all"
    batch_size: int = 50
    num_variations: int = 3
    seed: int = 42
    fid_threshold: float | None = None
    strength: float = 0.3
    max_seeds: int | None = None
    fid_reference_count: int | None = None
    resume: bool = True
    qualitative_grid_path: str | None = None
    qualitative_grid_samples: int = 4

    def __post_init__(self) -> None:
        if self.seed_scope not in ("all", "isolated"):
            raise ValueError(
                f"Invalid seed_scope '{self.seed_scope}'. Expected 'all' or 'isolated'."
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_variations < 1:
            raise ValueError(f"num_variations must be >= 1, got {self.num_variations}")
        if self.max_seeds is not None and self.max_seeds < 1:
            raise ValueError(f"max_seeds must be >= 1, got {self.max_seeds}")
        if self.fid_reference_count is not None and self.fid_reference_count < 2:
            raise ValueError(
                f"fid_reference_count must be >= 2, got {self.fid_reference_count}"
            )


@dataclass(frozen=True)
class MultiBatchGenerationSummary:
    """Aggregate summary of a multi-batch Synthetic Augmentation Batch run."""

    total_seeds: int
    processed_seeds: int
    skipped_seeds: int
    total_batches: int
    batches_executed: int
    batches_accepted: int
    batches_rejected: int
    accepted_images: int
    rejected_images: int
    fid_log_path: str
    metadata_csv_path: str
    resume_state_path: str
    qualitative_grid_path: str | None


def select_training_pneumothorax_seeds(
    train_df: pd.DataFrame,
    seed_scope: Literal["all", "isolated"] = "all",
    max_seeds: int | None = None,
) -> pd.DataFrame:
    """Select real pneumothorax training seeds in deterministic order.

    Args:
        train_df: Training split DataFrame from the Patient-Disjoint Partition.
        seed_scope: ``"all"`` includes both isolated and comorbid pneumothorax
            cases; ``"isolated"`` restricts to exact ``"Pneumothorax"`` labels.
        max_seeds: Optional cap on the number of seeds returned.

    Returns:
        Deduplicated, deterministically sorted DataFrame of selected seed rows.
    """
    if seed_scope not in ("all", "isolated"):
        raise ValueError(f"Invalid seed_scope '{seed_scope}'. Expected 'all' or 'isolated'.")

    if seed_scope == "isolated":
        if "Finding Labels" not in train_df.columns:
            raise ValueError("seed_scope='isolated' requires a 'Finding Labels' column.")
        mask = train_df["Finding Labels"].astype(str).str.strip() == "Pneumothorax"
    else:
        if "Finding Labels" in train_df.columns:
            mask = train_df["Finding Labels"].astype(str).str.contains("Pneumothorax", na=False)
        elif "Label" in train_df.columns:
            mask = train_df["Label"] == 1
        else:
            raise ValueError(
                "Training CSV must contain 'Finding Labels' or 'Label' column to select seeds."
            )

    selected = (
        train_df[mask]
        .drop_duplicates(subset=["Image Index"])
        .sort_values("Image Index", kind="mergesort")
        .reset_index(drop=True)
    )
    if max_seeds is not None:
        selected = selected.iloc[:max_seeds].reset_index(drop=True)
    return selected


def build_qualitative_comparison_grid(
    metadata_df: pd.DataFrame,
    real_images_dir: str,
    accepted_dir: str,
    output_path: str,
    max_pairs: int = 4,
) -> str | None:
    """Save a side-by-side real vs. accepted synthetic qualitative comparison grid."""
    if metadata_df.empty or max_pairs < 1:
        return None

    pairs: list[tuple[str, str, str, str]] = []
    seen_seeds: set[str] = set()
    for _, row in metadata_df.iterrows():
        synth_name = str(row["Image Index"])
        seed_name = str(row.get("Seed Image", ""))
        if not seed_name or seed_name in seen_seeds:
            continue
        real_path = os.path.join(real_images_dir, seed_name)
        synth_path = os.path.join(accepted_dir, synth_name)
        if os.path.exists(real_path) and os.path.exists(synth_path):
            pairs.append((real_path, synth_path, seed_name, synth_name))
            seen_seeds.add(seed_name)
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.0, 3.2 * n_rows))
    axes_2d = np.asarray(axes).reshape(n_rows, 2)
    for row_idx, (real_path, synth_path, seed_name, synth_name) in enumerate(pairs):
        real_img = Image.open(real_path).convert("L")
        synth_img = Image.open(synth_path).convert("L")
        axes_2d[row_idx, 0].imshow(real_img, cmap="gray")
        axes_2d[row_idx, 0].set_title(f"Real Seed: {seed_name}", fontsize=9)
        axes_2d[row_idx, 0].axis("off")

        axes_2d[row_idx, 1].imshow(synth_img, cmap="gray")
        axes_2d[row_idx, 1].set_title(f"Accepted Synthetic: {synth_name}", fontsize=9)
        axes_2d[row_idx, 1].axis("off")

    fig.suptitle(
        "Real Training Pneumothorax Seeds vs. FID-Accepted Synthetics",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


class DiffusersGeneratorAdapter:
    """Production Stable Diffusion v1.5 img2img adapter for pneumothorax synthesis."""

    BASE_PROMPT = (
        "frontal chest x-ray radiograph, pneumothorax, collapsed lung, "
        "pleural air, high quality medical imaging, grayscale"
    )
    NEGATIVE_PROMPT = "color, artifacts, text, labels, watermark, cartoon, drawing"

    def __init__(
        self,
        settings: Settings | None = None,
        device: torch.device | None = None,
        model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ) -> None:
        self.settings = settings or get_settings()
        self.device = device or _default_device()
        self.model_id = model_id
        self._pipe: Any = None
        self._loaded = False

    def _load_pipeline(self) -> Any:
        if self._loaded:
            return self._pipe
        from diffusers import StableDiffusionImg2ImgPipeline

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        log.info("Loading Stable Diffusion img2img pipeline", model_id=self.model_id)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id, torch_dtype=dtype, safety_checker=None
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        # Keep attention slicing only on constrained GPUs (<12 GB VRAM) or MPS;
        # higher-VRAM GPUs (e.g. L4 24 GB) run faster with native PyTorch SDPA.
        if self.device.type == "mps":
            pipe.enable_attention_slicing()
        elif self.device.type == "cuda" and torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            if total_vram_gb < 12.0:
                pipe.enable_attention_slicing()
        self._pipe = pipe
        self._loaded = True
        return self._pipe

    def generate(
        self,
        init_image_path: str,
        num_variations: int = 3,
        strength: float = 0.3,
        severity: str | None = None,
        location: str | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Generate synthetic variations conditioned on ``init_image_path``."""
        if not os.path.exists(init_image_path):
            raise FileNotFoundError(f"Reference radiograph not found: {init_image_path}")

        prompt = self.BASE_PROMPT
        if severity:
            prompt = f"{severity} pneumothorax, {prompt}"
        if location:
            prompt = f"{prompt}, located in the {location} chest cavity"

        init_img = Image.open(init_image_path).convert("RGB").resize((512, 512))
        image_size = self.settings.data.image_size
        base_seed = self.settings.training.seed if seed is None else int(seed)

        pipe = self._load_pipeline()
        generators = [
            torch.Generator(device=self.device).manual_seed(base_seed + i)
            for i in range(num_variations)
        ]

        with torch.inference_mode():
            outputs = pipe(
                prompt=[prompt] * num_variations,
                negative_prompt=[self.NEGATIVE_PROMPT] * num_variations,
                image=[init_img] * num_variations,
                strength=strength,
                guidance_scale=self.settings.augmentation.sd_guidance_scale,
                num_inference_steps=self.settings.augmentation.sd_num_inference_steps,
                generator=generators,
            ).images

        return [out.resize((image_size, image_size)) for out in outputs]


class PyTorchFIDAdapter:
    """Production Inception-v3 FID evaluator backed by ``pytorch-fid`` with real-stats caching."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or _default_device()
        self._inception_model: Any = None
        self._real_stats_cache: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _get_inception_model(self, dims: int = 2048) -> Any:
        if self._inception_model is None:
            from pytorch_fid.inception import InceptionV3

            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            self._inception_model = InceptionV3([block_idx]).to(self.device)
        return self._inception_model

    def compute_fid(self, real_dir: str, synthetic_dir: str, batch_size: int = 32) -> float:
        """Compute Inception-v3 FID between real and synthetic image directories."""
        try:
            from pytorch_fid.fid_score import (
                calculate_frechet_distance,
                compute_statistics_of_path,
            )
        except ImportError:
            guard_mock("[PyTorchFIDAdapter] pytorch-fid is not installed")
            log.warning("MOCK MODE: pytorch-fid unavailable; returning +inf.")
            return float("inf")

        if not os.path.exists(real_dir) or not os.path.exists(synthetic_dir):
            return float("inf")

        real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
        synth_files = [
            f for f in os.listdir(synthetic_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if len(real_files) < 2 or len(synth_files) < 2:
            return float("inf")

        dims = 2048
        num_workers = 4 if self.device.type == "cuda" else 0
        model = self._get_inception_model(dims=dims)

        cache_key = (os.path.realpath(real_dir), len(real_files), dims)
        if cache_key not in self._real_stats_cache:
            m1, s1 = compute_statistics_of_path(
                real_dir,
                model,
                min(batch_size, len(real_files)),
                dims,
                self.device,
                num_workers,
            )
            self._real_stats_cache[cache_key] = (m1, s1)
        m1, s1 = self._real_stats_cache[cache_key]

        m2, s2 = compute_statistics_of_path(
            synthetic_dir,
            model,
            min(batch_size, len(synth_files)),
            dims,
            self.device,
            num_workers,
        )
        fid_val = calculate_frechet_distance(m1, s1, m2, s2)
        return float(fid_val)


class SyntheticBatchPipeline:
    """Orchestrates Synthetic Augmentation Batch generation and FID quality gating."""

    def __init__(
        self,
        generator: GeneratorAdapter | None = None,
        fid_evaluator: FIDEvaluatorAdapter | None = None,
        fid_threshold: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.generator: GeneratorAdapter = generator or DiffusersGeneratorAdapter(self.settings)
        self.fid_evaluator: FIDEvaluatorAdapter = fid_evaluator or PyTorchFIDAdapter()
        self.fid_threshold = (
            float(fid_threshold)
            if fid_threshold is not None
            else float(self.settings.augmentation.fid_threshold)
        )
        self._generator_accepts_seed = self._check_generator_accepts_seed()

    def _check_generator_accepts_seed(self) -> bool:
        try:
            sig = inspect.signature(self.generator.generate)
        except (TypeError, ValueError):
            return False
        for param in sig.parameters.values():
            if param.name == "seed" or param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return False

    def generate_variations(
        self,
        init_image_path: str,
        num_variations: int = 3,
        strength: float = 0.3,
        severity: str | None = None,
        location: str | None = None,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """Delegate synthetic variation synthesis to the configured generator adapter."""
        if not os.path.exists(init_image_path):
            log.error("Seed radiograph not found", path=init_image_path)
            return []
        kwargs: dict[str, Any] = {
            "init_image_path": init_image_path,
            "num_variations": num_variations,
            "strength": strength,
            "severity": severity,
            "location": location,
        }
        if seed is not None and self._generator_accepts_seed:
            kwargs["seed"] = seed
        return self.generator.generate(**kwargs)

    def evaluate_and_filter_batch(
        self,
        real_dir: str,
        synthetic_batch_dir: str,
        accepted_dir: str,
        rejected_dir: str,
        fid_threshold: float | None = None,
    ) -> BatchFIDRecord:
        """Evaluate batch FID and route synthetic images into accepted or rejected folder."""
        os.makedirs(accepted_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

        threshold = self.fid_threshold if fid_threshold is None else float(fid_threshold)
        fid_score = float(self.fid_evaluator.compute_fid(real_dir, synthetic_batch_dir))
        accepted = fid_score <= threshold
        target_dir = accepted_dir if accepted else rejected_dir

        moved_files: list[str] = []
        if os.path.exists(synthetic_batch_dir):
            for filename in sorted(os.listdir(synthetic_batch_dir)):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    src = os.path.join(synthetic_batch_dir, filename)
                    dst = os.path.join(target_dir, filename)
                    shutil.move(src, dst)
                    moved_files.append(dst)

        log.info(
            "Evaluated Synthetic Augmentation Batch",
            fid_score=fid_score,
            threshold=threshold,
            accepted=accepted,
            num_files=len(moved_files),
        )
        return BatchFIDRecord(
            accepted=accepted,
            fid_score=fid_score,
            threshold=threshold,
            moved_files=moved_files,
        )

    @staticmethod
    def _compute_plan_hash(
        seed_names: list[str],
        config: SyntheticGenerationConfig,
        active_threshold: float,
    ) -> str:
        payload = {
            "seed_scope": config.seed_scope,
            "batch_size": config.batch_size,
            "num_variations": config.num_variations,
            "seed": config.seed,
            "strength": config.strength,
            "fid_threshold": active_threshold,
            "seeds": seed_names,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def run_multi_batch(
        self,
        config: SyntheticGenerationConfig,
        on_batch_complete: Callable[[int, int, BatchFIDRecord], None] | None = None,
    ) -> MultiBatchGenerationSummary:
        """Run multi-batch FID-gated synthetic generation with crash-safe resume.

        Iterates deterministically across all qualifying pneumothorax training
        seeds from ``config.train_csv``, evaluates each generated batch against
        real training pneumothorax radiographs via Inception-v3 FID, routes
        images to ``accepted/`` or ``rejected/``, and atomically persists
        ``fid_log.csv``, ``synthetic_metadata.csv``, ``resume_state.json``, and
        a qualitative real-vs-synthetic comparison grid.
        """
        if not os.path.exists(config.train_csv):
            raise FileNotFoundError(f"Training split CSV not found: {config.train_csv}")

        active_threshold = (
            self.fid_threshold if config.fid_threshold is None else float(config.fid_threshold)
        )

        train_df = pd.read_csv(config.train_csv)
        candidate_df = select_training_pneumothorax_seeds(
            train_df, seed_scope=config.seed_scope, max_seeds=config.max_seeds
        )

        # Keep only seeds whose real radiograph exists on disk
        available_rows: list[dict[str, Any]] = []
        for _, row in candidate_df.iterrows():
            img_name = str(row["Image Index"])
            if os.path.exists(os.path.join(config.real_images_dir, img_name)):
                patient_id = (
                    str(row["Patient ID"])
                    if "Patient ID" in candidate_df.columns and pd.notna(row["Patient ID"])
                    else img_name.split("_")[0]
                )
                available_rows.append({"Image Index": img_name, "Patient ID": patient_id})
            else:
                log.warning("Skipping missing training seed radiograph", image_index=img_name)

        output_dir = config.output_dir
        accepted_dir = os.path.join(output_dir, "accepted")
        rejected_dir = os.path.join(output_dir, "rejected")
        temp_batch_dir = os.path.join(output_dir, "temp_batch")
        temp_real_dir = os.path.join(output_dir, "temp_real")
        fid_log_path = os.path.join(output_dir, "fid_log.csv")
        metadata_csv_path = os.path.join(output_dir, "synthetic_metadata.csv")
        resume_state_path = os.path.join(output_dir, "resume_state.json")
        complete_marker_path = os.path.join(output_dir, "generation_complete.json")
        grid_target = (
            config.qualitative_grid_path
            if config.qualitative_grid_path is not None
            else os.path.join(output_dir, "qualitative_grid.png")
        )

        if not config.resume:
            for path in (accepted_dir, rejected_dir, temp_batch_dir, temp_real_dir):
                shutil.rmtree(path, ignore_errors=True)
            for path in (fid_log_path, metadata_csv_path, resume_state_path, complete_marker_path):
                if os.path.exists(path):
                    os.remove(path)

        os.makedirs(accepted_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

        seed_names = [r["Image Index"] for r in available_rows]
        plan_hash = self._compute_plan_hash(seed_names, config, active_threshold)

        batches = [
            available_rows[i : i + config.batch_size]
            for i in range(0, len(available_rows), config.batch_size)
        ]

        completed_batches: set[int] = set()
        processed_seeds: set[str] = set()
        fid_by_batch: dict[int, dict[str, Any]] = {}
        accepted_by_image: dict[str, dict[str, Any]] = {}

        if config.resume and os.path.exists(resume_state_path):
            with open(resume_state_path, encoding="utf-8") as fh:
                saved_state = json.load(fh)
            if saved_state.get("plan_hash") != plan_hash:
                raise ValueError(
                    "Existing resume_state.json does not match the current training seed "
                    "plan or batch configuration; pass resume=False / --no-resume to "
                    "regenerate cleanly."
                )
            completed_batches = {int(b) for b in saved_state.get("completed_batches", [])}
            processed_seeds = {str(s) for s in saved_state.get("processed_seeds", [])}

            if os.path.exists(fid_log_path):
                existing_fid_df = pd.read_csv(fid_log_path)
                for rec in existing_fid_df.to_dict("records"):
                    b_num = int(rec["batch"])
                    if b_num in completed_batches:
                        fid_by_batch[b_num] = rec

            if os.path.exists(metadata_csv_path):
                existing_meta_df = pd.read_csv(metadata_csv_path)
                has_batch_col = "Batch" in existing_meta_df.columns
                for rec in existing_meta_df.to_dict("records"):
                    if has_batch_col and int(rec["Batch"]) not in completed_batches:
                        continue
                    accepted_by_image[str(rec["Image Index"])] = rec

        if not available_rows:
            _atomic_write_csv(pd.DataFrame(columns=FID_LOG_COLUMNS), fid_log_path)
            _atomic_write_csv(pd.DataFrame(columns=METADATA_COLUMNS), metadata_csv_path)
            _atomic_write_text(
                resume_state_path,
                json.dumps(
                    {
                        "plan_hash": plan_hash,
                        "completed_batches": [],
                        "processed_seeds": [],
                    },
                    indent=2,
                ),
            )
            return MultiBatchGenerationSummary(
                total_seeds=0,
                processed_seeds=0,
                skipped_seeds=0,
                total_batches=0,
                batches_executed=0,
                batches_accepted=0,
                batches_rejected=0,
                accepted_images=0,
                rejected_images=0,
                fid_log_path=fid_log_path,
                metadata_csv_path=metadata_csv_path,
                resume_state_path=resume_state_path,
                qualitative_grid_path=None,
            )

        batches_executed = 0
        skipped_seeds = 0

        # Stage real training pneumothorax reference images once for the run
        if os.path.exists(temp_real_dir):
            shutil.rmtree(temp_real_dir)
        os.makedirs(temp_real_dir, exist_ok=True)
        try:
            ref_rows = available_rows
            if (
                config.fid_reference_count is not None
                and len(available_rows) > config.fid_reference_count
            ):
                rng = np.random.default_rng(config.seed)
                chosen_idx = sorted(
                    rng.choice(
                        len(available_rows),
                        size=config.fid_reference_count,
                        replace=False,
                    ).tolist()
                )
                ref_rows = [available_rows[i] for i in chosen_idx]

            for ref_row in ref_rows:
                fname = ref_row["Image Index"]
                _stage_reference_image(
                    os.path.join(config.real_images_dir, fname),
                    os.path.join(temp_real_dir, fname),
                )

            for batch_idx, batch_seeds in enumerate(batches, start=1):
                if batch_idx in completed_batches:
                    skipped_seeds += len(batch_seeds)
                    continue

                if os.path.exists(complete_marker_path):
                    os.remove(complete_marker_path)

                # Clean any partial files left by a prior mid-batch interruption
                if os.path.exists(temp_batch_dir):
                    shutil.rmtree(temp_batch_dir)
                os.makedirs(temp_batch_dir, exist_ok=True)

                pending_meta_by_name: dict[str, dict[str, Any]] = {}
                batch_seed_names: list[str] = []
                for local_idx, seed_row in enumerate(batch_seeds):
                    seed_name = seed_row["Image Index"]
                    if seed_name in processed_seeds:
                        skipped_seeds += 1
                        continue
                    batch_seed_names.append(seed_name)
                    base_stem = os.path.splitext(seed_name)[0]
                    for var_idx in range(config.num_variations):
                        stale_name = f"synth_{base_stem}_{var_idx}.png"
                        for target in (accepted_dir, rejected_dir):
                            stale_path = os.path.join(target, stale_name)
                            if os.path.exists(stale_path):
                                os.remove(stale_path)
                    global_seed_idx = (batch_idx - 1) * config.batch_size + local_idx
                    image_seed = config.seed + global_seed_idx * config.num_variations
                    real_img_path = os.path.join(config.real_images_dir, seed_name)

                    synthetics = self.generate_variations(
                        init_image_path=real_img_path,
                        num_variations=config.num_variations,
                        strength=config.strength,
                        seed=image_seed,
                    )
                    for var_idx, synth_img in enumerate(synthetics):
                        synth_filename = f"synth_{base_stem}_{var_idx}.png"
                        synth_img.save(os.path.join(temp_batch_dir, synth_filename))
                        pending_meta_by_name[synth_filename] = {
                            "Image Index": synth_filename,
                            "Finding Labels": "Pneumothorax",
                            "Seed Image": seed_name,
                            "Patient ID": seed_row["Patient ID"],
                            "Variation": var_idx,
                            "Batch": batch_idx,
                        }

                batch_record = self.evaluate_and_filter_batch(
                    real_dir=temp_real_dir,
                    synthetic_batch_dir=temp_batch_dir,
                    accepted_dir=accepted_dir,
                    rejected_dir=rejected_dir,
                    fid_threshold=active_threshold,
                )
                batches_executed += 1

                fid_by_batch[batch_idx] = {
                    "batch": batch_idx,
                    "seed_count": len(batch_seed_names),
                    "generated_count": len(pending_meta_by_name),
                    "fid": float(batch_record.fid_score),
                    "threshold": float(active_threshold),
                    "accepted": bool(batch_record.accepted),
                    "decision": "accepted" if batch_record.accepted else "rejected",
                }
                if batch_record.accepted:
                    for filepath in batch_record.moved_files:
                        fname = os.path.basename(filepath)
                        meta = pending_meta_by_name.get(
                            fname,
                            {
                                "Image Index": fname,
                                "Finding Labels": "Pneumothorax",
                                "Seed Image": "",
                                "Patient ID": "",
                                "Variation": 0,
                                "Batch": batch_idx,
                            },
                        )
                        accepted_by_image[fname] = meta

                completed_batches.add(batch_idx)
                processed_seeds.update(batch_seed_names)

                # Atomically persist logs, metadata, and resume state after each batch
                fid_df = pd.DataFrame(
                    [fid_by_batch[b] for b in sorted(fid_by_batch)],
                    columns=FID_LOG_COLUMNS,
                )
                _atomic_write_csv(fid_df, fid_log_path)

                meta_rows = [accepted_by_image[k] for k in sorted(accepted_by_image)]
                meta_df = pd.DataFrame(meta_rows, columns=METADATA_COLUMNS)
                _atomic_write_csv(meta_df, metadata_csv_path)

                _atomic_write_text(
                    resume_state_path,
                    json.dumps(
                        {
                            "plan_hash": plan_hash,
                            "completed_batches": sorted(completed_batches),
                            "processed_seeds": sorted(processed_seeds),
                        },
                        indent=2,
                    ),
                )
                log.info(
                    "Completed synthetic batch",
                    batch=batch_idx,
                    total_batches=len(batches),
                    fid=round(float(batch_record.fid_score), 3),
                    accepted=bool(batch_record.accepted),
                    total_accepted_images=len(accepted_by_image),
                )
                if on_batch_complete is not None:
                    on_batch_complete(batch_idx, len(batches), batch_record)
        finally:
            shutil.rmtree(temp_real_dir, ignore_errors=True)
            if os.path.exists(temp_batch_dir) and not os.listdir(temp_batch_dir):
                os.rmdir(temp_batch_dir)

        # Ensure metadata and FID log exist even on a zero-work resume
        if not os.path.exists(fid_log_path):
            _atomic_write_csv(pd.DataFrame(columns=FID_LOG_COLUMNS), fid_log_path)
        if not os.path.exists(metadata_csv_path):
            _atomic_write_csv(pd.DataFrame(columns=METADATA_COLUMNS), metadata_csv_path)

        final_meta_df = pd.read_csv(metadata_csv_path)
        saved_grid = build_qualitative_comparison_grid(
            metadata_df=final_meta_df,
            real_images_dir=config.real_images_dir,
            accepted_dir=accepted_dir,
            output_path=grid_target,
            max_pairs=config.qualitative_grid_samples,
        )

        batches_accepted = sum(1 for rec in fid_by_batch.values() if bool(rec["accepted"]))
        batches_rejected = sum(1 for rec in fid_by_batch.values() if not bool(rec["accepted"]))
        accepted_images = len(accepted_by_image)
        rejected_images = sum(
            int(rec["generated_count"])
            for rec in fid_by_batch.values()
            if not bool(rec["accepted"])
        )

        _atomic_write_text(
            complete_marker_path,
            json.dumps(
                {
                    "plan_hash": plan_hash,
                    "total_seeds": len(available_rows),
                    "total_batches": len(batches),
                    "batches_accepted": batches_accepted,
                    "batches_rejected": batches_rejected,
                    "accepted_images": accepted_images,
                    "rejected_images": rejected_images,
                },
                indent=2,
            ),
        )

        return MultiBatchGenerationSummary(
            total_seeds=len(available_rows),
            processed_seeds=len(processed_seeds),
            skipped_seeds=skipped_seeds,
            total_batches=len(batches),
            batches_executed=batches_executed,
            batches_accepted=batches_accepted,
            batches_rejected=batches_rejected,
            accepted_images=accepted_images,
            rejected_images=rejected_images,
            fid_log_path=fid_log_path,
            metadata_csv_path=metadata_csv_path,
            resume_state_path=resume_state_path,
            qualitative_grid_path=saved_grid,
        )
