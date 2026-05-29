"""Unit tests for tier comparison visualization utilities."""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from core.explainability.tier_comparison import combined_overlay, side_by_side


@pytest.fixture()
def sample_arrays() -> tuple[
    np.ndarray[object, np.dtype[np.float32]],
    np.ndarray[object, np.dtype[np.float32]],
    np.ndarray[object, np.dtype[np.float32]],
]:
    rng = np.random.default_rng(42)
    image = rng.random((64, 64, 3)).astype(np.float32)
    cam1 = rng.random((64, 64, 3)).astype(np.float32)
    cam2 = rng.random((64, 64, 3)).astype(np.float32)
    return image, cam1, cam2


def test_side_by_side_returns_figure(sample_arrays: tuple) -> None:
    image, cam1, cam2 = sample_arrays
    fig = side_by_side(image, cam1, cam2)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close("all")


def test_side_by_side_saves_file(
    sample_arrays: tuple, tmp_path: pytest.TempPathFactory
) -> None:
    image, cam1, cam2 = sample_arrays
    save_path = str(tmp_path / "comparison.png")  # type: ignore[operator]
    fig = side_by_side(image, cam1, cam2, save_path=save_path)
    assert fig is not None
    assert os.path.exists(save_path)
    import matplotlib.pyplot as plt

    plt.close("all")


def test_combined_overlay_shape(sample_arrays: tuple) -> None:
    image, _, _ = sample_arrays
    rng = np.random.default_rng(0)
    cam1_gray = rng.random((64, 64)).astype(np.float32)
    cam2_gray = rng.random((64, 64)).astype(np.float32)
    result = combined_overlay(image, cam1_gray, cam2_gray)
    assert result.shape == (64, 64, 3)


def test_combined_overlay_value_range(sample_arrays: tuple) -> None:
    image, _, _ = sample_arrays
    rng = np.random.default_rng(1)
    cam1_gray = rng.random((64, 64)).astype(np.float32)
    cam2_gray = rng.random((64, 64)).astype(np.float32)
    result = combined_overlay(image, cam1_gray, cam2_gray)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
