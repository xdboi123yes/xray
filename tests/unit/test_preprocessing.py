"""Unit tests for ImagePreprocessor (CLAHE & Preprocessing)."""

from __future__ import annotations

from PIL import Image
import numpy as np
from infrastructure.data.preprocessing import ImagePreprocessor

def test_image_preprocessor_clahe() -> None:
    """Verify that CLAHE is applied successfully and dimensions are maintained."""
    preprocessor = ImagePreprocessor(target_size=(100, 100))
    
    # Create dummy grayscaled image
    dummy_np = (np.random.rand(100, 100) * 255).astype(np.uint8)
    dummy_img = Image.fromarray(dummy_np)
    
    enhanced = preprocessor.apply_clahe(dummy_img)
    assert enhanced.size == (100, 100)
    assert enhanced.mode == "L"

def test_image_preprocessor_pipeline() -> None:
    """Verify that full preprocessing pipeline reshapes and enhances dummy input."""
    preprocessor = ImagePreprocessor(target_size=(128, 128))
    
    # Create RGB dummy image
    dummy_np = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    dummy_img = Image.fromarray(dummy_np)
    
    processed = preprocessor.preprocess_pipeline(dummy_img)
    assert processed.size == (128, 128)
    assert processed.mode == "L"
