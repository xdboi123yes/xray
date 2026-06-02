"""Unit tests for API inference preprocessing (train/serve parity).

The serving path must apply the same ImageNet normalization the models were
trained with; feeding un-normalized [0, 1] images is out-of-distribution and
was the root cause of border-focused Grad-CAM maps.
"""

from __future__ import annotations

import io

from PIL import Image

from application.services.inference_service import InferenceService

# ImageNet normalization constants the models were trained with.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _png_bytes(color: tuple[int, int, int], size: int = 64) -> bytes:
    """Encode a solid-color RGB PNG and return its bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocess_applies_imagenet_normalization() -> None:
    service = InferenceService()
    tensor = service.preprocess_image(_png_bytes((0, 0, 0)))

    assert tensor.shape == (1, 3, 224, 224)
    # A pure-black pixel (0) must map to (0 - mean) / std -- clearly negative --
    # matching training. The old pipeline left black pixels at 0.0.
    assert tensor.min().item() < -1.5
    expected_red_min = -_MEAN[0] / _STD[0]  # ~ -2.11
    assert abs(tensor[0, 0].min().item() - expected_red_min) < 0.05


def test_display_image_stays_unnormalized() -> None:
    service = InferenceService()
    display = service.display_image(_png_bytes((128, 128, 128)))

    assert display.shape == (224, 224, 3)
    # The Grad-CAM overlay base is the human-visible image, strictly within [0, 1].
    assert display.min() >= 0.0
    assert display.max() <= 1.0
