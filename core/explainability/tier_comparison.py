"""Tier Escalation Attention Comparison Visualization.

Provides side-by-side diagnostic maps comparing Tier 1 vs Tier 2 spatial attention,
and attention divergence overlays showing focal variance.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def side_by_side(
    image: np.ndarray[Any, Any],
    cam1_overlay: np.ndarray[Any, Any],
    cam2_overlay: np.ndarray[Any, Any],
    save_path: str | None = None,
) -> Any:
    """Creates a 3-panel figure showing original radiograph and attention overlays.

    Args:
        image: Original radiograph float array in [0, 1].
        cam1_overlay: Tier 1 Grad-CAM overlay array in [0, 255].
        cam2_overlay: Tier 2 Grad-CAM overlay array in [0, 255].
        save_path: Optional output path to save diagram figure.

    Returns:
        Matplotlib figure instance.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(cam1_overlay)
    axes[1].set_title("Tier 1 (MobileNetV2) Focus")
    axes[1].axis("off")

    axes[2].imshow(cam2_overlay)
    axes[2].set_title("Tier 2 (EfficientNetB4) Focus")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def combined_overlay(
    image: np.ndarray[Any, Any],
    grayscale_cam1: np.ndarray[Any, Any],
    grayscale_cam2: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Combines Tier 1 (Red) and Tier 2 (Blue) heatmaps over grayscale background.

    Args:
        image: Original radiograph image.
        grayscale_cam1: Grayscale Tier 1 CAM.
        grayscale_cam2: Grayscale Tier 2 CAM.

    Returns:
        Combined overlay float array in [0, 1].
    """
    rgb_heatmap = np.zeros_like(image)
    rgb_heatmap[:, :, 0] = grayscale_cam1  # Red channel for Tier 1
    rgb_heatmap[:, :, 2] = grayscale_cam2  # Blue channel for Tier 2

    # Standardize radiograph under grayscale
    gray_image = np.mean(image, axis=2, keepdims=True)
    gray_image = np.repeat(gray_image, 3, axis=2)

    # Apply blend overlay
    alpha = 0.4
    overlay = (1.0 - alpha) * gray_image + alpha * rgb_heatmap
    overlay = np.clip(overlay, 0, 1)

    return np.asarray(overlay)
