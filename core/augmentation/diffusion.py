"""Diffusion-based image augmentation strategy for Chest X-Ray classification.

Uses Stable Diffusion Image-to-Image pipeline to produce synthetic variations
of chest radiographs while preserving overall anatomical structure.
"""

from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from core.interfaces.base_augmentation import BaseAugmentation


class DiffusionAugmentation(BaseAugmentation):
    """Generative medical image augmentation strategy using Stable Diffusion.

    Injects pneumatic structural anomalies into input chest radiographs
    to regularize deep classifiers during training.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device | None = None,
        pipeline: Any | None = None,
    ) -> None:
        """Initialize the diffusion augmentation strategy.

        Args:
            config: Configuration dictionary with augmentation parameters.
            device: Active computation device (CUDA/MPS/CPU).
            pipeline: Pre-loaded StableDiffusionImg2ImgPipeline instance (optional).
        """
        self._config = config
        self._device = device or torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._pipeline = pipeline

        # Prompt engineering optimized for medical imaging consistency
        self._prompt = (
            "frontal chest x-ray radiograph, pneumothorax, collapsed lung, "
            "pleural air, high quality medical imaging, grayscale"
        )
        self._negative_prompt = (
            "color, artifacts, text, labels, watermark, cartoon, drawing"
        )

        self._num_inference_steps = config.get("augmentation", {}).get(
            "sd_num_inference_steps", 50
        )
        self._guidance_scale = config.get("augmentation", {}).get(
            "sd_guidance_scale", 7.5
        )
        self._strength = config.get("augmentation", {}).get("sd_strength", 0.3)

    def _lazy_load_pipeline(self) -> Any:
        """Lazy load the Stable Diffusion img2img pipeline.

        Returns:
            The loaded StableDiffusionImg2ImgPipeline instance.
        """
        if self._pipeline is not None:
            return self._pipeline

        from diffusers import StableDiffusionImg2ImgPipeline

        model_id = "runwayml/stable-diffusion-v1-5"
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None
        )
        pipe = pipe.to(self._device)

        if self._device.type in ("mps", "cuda"):
            pipe.enable_attention_slicing()

        self._pipeline = pipe
        return pipe

    def apply(
        self, image: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Apply Stable Diffusion image-to-image augmentation.

        Converts input tensor or array to a PIL Image, passes it through the
        img2img pipeline with medical prompts, and converts it back to the
        original input type.

        Args:
            image: Input image as a numpy array or PyTorch tensor.

        Returns:
            The generated synthetic image of the same type.
        """
        # Save original type and shape information
        is_tensor = isinstance(image, torch.Tensor)
        original_device = image.device if isinstance(image, torch.Tensor) else None

        # Convert to PIL Image [512x512] for optimal Stable Diffusion execution
        if is_tensor:
            # Assumes [C, H, W]
            tensor_cpu = (
                cast(torch.Tensor, image).detach().cpu().clamp(0, 1).numpy()
            )
            arr = (tensor_cpu.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        else:
            arr = cast(np.ndarray[Any, Any], image)
            if arr.dtype != np.uint8:
                arr = (arr * 255.0).astype(np.uint8)

        # Make sure image has 3 channels for SD
        if len(arr.shape) == 2:
            arr = np.repeat(np.expand_dims(arr, 2), 3, axis=2)
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        pil_image = Image.fromarray(arr).resize((512, 512))

        # Run pipeline
        pipe = self._lazy_load_pipeline()
        seed = self._config.get("training", {}).get("seed", 42)
        generator = torch.Generator(device=self._device).manual_seed(seed)

        result_pil = pipe(
            prompt=self._prompt,
            negative_prompt=self._negative_prompt,
            image=pil_image,
            strength=self._strength,
            guidance_scale=self._guidance_scale,
            num_inference_steps=self._num_inference_steps,
            generator=generator,
        ).images[0]

        # Resize back to original target resolution
        image_size = self._config.get("data", {}).get("image_size", 224)
        result_pil = result_pil.resize((image_size, image_size))
        result_arr = np.array(result_pil).astype(np.float32) / 255.0

        if is_tensor:
            # Convert back to torch tensor [C, H, W] on the original device
            result_tensor = (
                torch.from_numpy(result_arr.transpose(2, 0, 1))
                .float()
                .to(original_device)
            )
            return result_tensor
        else:
            return result_arr
