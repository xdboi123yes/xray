"""Stable Diffusion based synthetic X-ray generator.

Uses Image-to-Image pipeline to create Pneumothorax variations from raw X-rays.
"""

from __future__ import annotations

import os
from typing import Any

import structlog
import torch
from diffusers import StableDiffusionImg2ImgPipeline  # type: ignore[import-untyped]
from PIL import Image

from config.settings import Settings, get_settings

log = structlog.get_logger(__name__)


class SyntheticGenerator:
    """Generates synthetic variations of Pneumothorax chest X-Rays using SD img2img."""

    def __init__(
        self,
        config: dict[str, Any] | Settings | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initializes the SyntheticGenerator.

        Args:
            config: Project settings (Pydantic Settings or raw dict).
            device: Active computation device (CUDA/MPS/CPU).
        """
        # Load settings or default
        if config is None:
            self.settings = get_settings()
        elif isinstance(config, dict):
            # Parse dict into Settings model
            self.settings = Settings(**config)
        else:
            self.settings = config

        self.device = device or torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Standard SD v1.5 backbone
        model_id = "runwayml/stable-diffusion-v1-5"
        log.info(f"Loading Stable Diffusion pipeline ({model_id}) on {self.device}...")

        # Mac MPS performs best with float32; CUDA supports float16
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,  # Disable safety checker for medical images
        )
        self.pipe = self.pipe.to(self.device)

        # Enable MPS/CUDA attention slicing to save memory on unified architectures
        if self.device.type in ("mps", "cuda"):
            self.pipe.enable_attention_slicing()

        # Configured clinical prompts
        self.prompt = (
            "frontal chest x-ray radiograph, pneumothorax, collapsed lung, "
            "pleural air, high quality medical imaging, grayscale"
        )
        self.negative_prompt = "color, artifacts, text, labels, watermark, cartoon, drawing"

        self.num_inference_steps = self.settings.augmentation.sd_num_inference_steps
        self.guidance_scale = self.settings.augmentation.sd_guidance_scale

    def generate(
        self,
        init_image_path: str,
        num_variations: int = 3,
        strength: float = 0.3,
    ) -> list[Image.Image]:
        """Generates realistic variations of a reference Pneumothorax chest X-Ray.

        Args:
            init_image_path: Absolute or relative path to the base image.
            num_variations: Number of modified images to generate.
            strength: Degree of alteration from the original image (0.0 to 1.0).

        Returns:
            A list of PIL Images containing the generated synthetic variations.
        """
        if not os.path.exists(init_image_path):
            raise FileNotFoundError(f"Reference image not found at {init_image_path}")

        init_image = Image.open(init_image_path).convert("RGB")
        # Stable Diffusion 1.5 standard internal size
        init_image = init_image.resize((512, 512))

        seed = self.settings.training.seed
        generator = torch.Generator(device=self.device).manual_seed(seed)

        generated_images: list[Image.Image] = []
        for i in range(num_variations):
            # Advance seed to ensure diversity across variations
            generator.manual_seed(seed + i)

            result = self.pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=generator,
            ).images[0]

            # Resize back to target dataset resolution
            image_size = self.settings.data.image_size
            resized_result = result.resize((image_size, image_size))
            generated_images.append(resized_result)

        return generated_images
