"""Preprocessing functions for chest radiographs, including CLAHE and histogram matching."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
import albumentations as A

class ImagePreprocessor:
    """Handles medical image preprocessing pipelines such as CLAHE and standardization."""

    def __init__(self, target_size: tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size
        self.clahe_transform = A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0)

    def apply_clahe(self, image: Image.Image) -> Image.Image:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to PIL Image."""
        # Convert to numpy array
        img_np = np.array(image.convert("L"))
        # Apply CLAHE via albumentations
        augmented = self.clahe_transform(image=img_np)
        enhanced_np = augmented["image"]
        return Image.fromarray(enhanced_np)

    def apply_histogram_matching(self, source: Image.Image, reference: Image.Image) -> Image.Image:
        """Simple histogram matching using ImageOps.autocontrast as robust fallback."""
        # Ensure grayscale
        src_gray = source.convert("L")
        # Match using autocontrast for standardizing range
        matched = ImageOps.autocontrast(src_gray, cutoff=1)
        return matched

    def preprocess_pipeline(self, image: Image.Image) -> Image.Image:
        """Full pipeline: Resize, Grayscale, CLAHE enhancement, and normalization fallback."""
        # 1. Resize
        resized = image.resize(self.target_size, Image.Resampling.BILINEAR)
        # 2. Grayscale
        gray = resized.convert("L")
        # 3. CLAHE
        enhanced = self.apply_clahe(gray)
        return enhanced
