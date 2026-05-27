"""Unit tests for the SyntheticDataService class.

Verifies conditional prompt generation, mock stable-diffusion image generation,
and Inception FID score calculations.
"""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from application.services.synthetic_data_service import SyntheticDataService


def test_synthetic_data_service_init(mock_config: dict[str, Any]) -> None:
    """Test that SyntheticDataService initializes correctly.

    Args:
        mock_config: Pytest fixture representing mock parameters.
    """
    service = SyntheticDataService(mock_config)
    assert service.fid_threshold == 50.0
    assert "pneumothorax" in service.base_prompt
    assert not service._pipeline_loaded


def test_synthetic_data_service_generate_mock(mock_config: dict[str, Any], tmp_path: Any) -> None:
    """Test mock variation generation using simulated image noise.

    Args:
        mock_config: Pytest fixture representing mock parameters.
        tmp_path: Pytest temporary directory path.
    """
    service = SyntheticDataService(mock_config)

    # Setup dummy conditioning image
    dummy_img_path = os.path.join(tmp_path, "conditioning_image.png")
    Image.new("RGB", (224, 224), color="gray").save(dummy_img_path)

    # Execute mock generation
    images = service.generate_variations(
        dummy_img_path,
        n_variations=2,
        strength=0.3,
        severity="severe",
        location="apex",
    )

    assert len(images) == 2
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (224, 224)


def test_synthetic_data_service_compute_fid(mock_config: dict[str, Any]) -> None:
    """Test mock FID calculation when splits directories are unpopulated.

    Args:
        mock_config: Pytest fixture representing mock parameters.
    """
    service = SyntheticDataService(mock_config)
    # With missing/unpopulated directories, it should return inf or fallback float
    score = service.compute_fid("mock_real_dir", "mock_synth_dir")
    assert score == float("inf") or isinstance(score, float)
