"""FID-gated Synthetic Augmentation Batch pipeline with injectable adapters.

Consolidates Stable Diffusion img2img radiograph synthesis and Inception-v3 FID
quality filtering behind two narrow true-external seams (:class:`GeneratorAdapter`
and :class:`FIDEvaluatorAdapter`), enabling deterministic CPU tests without real
diffusion or Inception weights.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Protocol

import structlog
import torch
from PIL import Image

from tcrp.config import Settings, get_settings, guard_mock

log = structlog.get_logger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        model_id: str = "runwayml/stable-diffusion-v1-5",
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
        if self.device.type in ("cuda", "mps"):
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
        seed = self.settings.training.seed

        pipe = self._load_pipeline()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        generated: list[Image.Image] = []
        for i in range(num_variations):
            generator.manual_seed(seed + i)
            out = pipe(
                prompt=prompt,
                negative_prompt=self.NEGATIVE_PROMPT,
                image=init_img,
                strength=strength,
                guidance_scale=self.settings.augmentation.sd_guidance_scale,
                num_inference_steps=self.settings.augmentation.sd_num_inference_steps,
                generator=generator,
            ).images[0]
            generated.append(out.resize((image_size, image_size)))
        return generated


class PyTorchFIDAdapter:
    """Production Inception-v3 FID evaluator backed by ``pytorch-fid``."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or _default_device()

    def compute_fid(self, real_dir: str, synthetic_dir: str, batch_size: int = 32) -> float:
        """Compute Inception-v3 FID between real and synthetic image directories."""
        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths
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

        fid_val = calculate_fid_given_paths(
            paths=[real_dir, synthetic_dir],
            batch_size=min(batch_size, len(synth_files)),
            device=str(self.device),
            dims=2048,
            num_workers=0,
        )
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

    def generate_variations(
        self,
        init_image_path: str,
        num_variations: int = 3,
        strength: float = 0.3,
        severity: str | None = None,
        location: str | None = None,
    ) -> list[Image.Image]:
        """Delegate synthetic variation synthesis to the configured generator adapter."""
        if not os.path.exists(init_image_path):
            log.error("Seed radiograph not found", path=init_image_path)
            return []
        return self.generator.generate(
            init_image_path=init_image_path,
            num_variations=num_variations,
            strength=strength,
            severity=severity,
            location=location,
        )

    def evaluate_and_filter_batch(
        self,
        real_dir: str,
        synthetic_batch_dir: str,
        accepted_dir: str,
        rejected_dir: str,
    ) -> BatchFIDRecord:
        """Evaluate batch FID and route synthetic images into accepted or rejected folder."""
        os.makedirs(accepted_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

        fid_score = float(self.fid_evaluator.compute_fid(real_dir, synthetic_batch_dir))
        accepted = fid_score <= self.fid_threshold
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
            threshold=self.fid_threshold,
            accepted=accepted,
            num_files=len(moved_files),
        )
        return BatchFIDRecord(
            accepted=accepted,
            fid_score=fid_score,
            threshold=self.fid_threshold,
            moved_files=moved_files,
        )
