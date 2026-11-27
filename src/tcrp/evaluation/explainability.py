"""HiResCAM attribution and Tier 1 vs. Tier 2 spatial attention comparisons."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from tcrp.config import guard_mock

try:
    from pytorch_grad_cam import HiResCAM  # type: ignore[import-untyped]
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore[import-untyped]

    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

ReshapeTransform = Callable[[torch.Tensor], torch.Tensor]


def _swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape Swin token grid/sequence into ``[B, C, H, W]`` for CAM pooling."""
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2).contiguous()
    batch, length, channels = tensor.shape
    side = int(round(length**0.5))
    return tensor.reshape(batch, side, side, channels).permute(0, 3, 1, 2).contiguous()


def resolve_target_layers(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Module], ReshapeTransform | None]:
    """Resolve the CAM target layer(s) and optional reshape transform for a model."""
    backbone = getattr(model, "backbone", model)

    features = getattr(backbone, "features", None)
    if features is not None:
        return [features[-1]], None

    layers = getattr(backbone, "layers", None)
    if layers is not None:
        return [layers[-1].blocks[-1].norm2], _swin_reshape_transform

    norm = getattr(backbone, "norm", None)
    if norm is not None:
        return [norm], _swin_reshape_transform

    raise ValueError(f"Cannot resolve a CAM target layer for {type(model).__name__}")


def suppress_background(
    cam: np.ndarray[Any, Any],
    rgb_img: np.ndarray[Any, Any],
    threshold: float = 0.05,
    blur_ksize: int = 15,
) -> np.ndarray[Any, Any]:
    """Zero out CAM attribution over near-black collimation/background pixels."""
    gray = rgb_img.mean(axis=2) if rgb_img.ndim == 3 else rgb_img
    mask = (gray > threshold).astype(np.float32)
    if blur_ksize > 1:
        mask = np.asarray(cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0), dtype=np.float32)
    masked = cam * mask
    peak = float(masked.max())
    if peak > 1e-8:
        masked = masked / peak
    return np.asarray(masked)


class XRayHiResCAM:
    """Computes and overlays HiResCAM attributions on input chest radiographs."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: list[torch.nn.Module],
        reshape_transform: ReshapeTransform | None = None,
    ) -> None:
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        if HAS_GRAD_CAM:
            self.cam = HiResCAM(
                model=model,
                target_layers=target_layers,
                reshape_transform=reshape_transform,
            )

    def generate(
        self, input_tensor: torch.Tensor, target_category: Any = None
    ) -> np.ndarray[Any, Any]:
        """Generate a 2D grayscale HiResCAM attribution map in ``[0, 1]``."""
        if HAS_GRAD_CAM:
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=target_category)
            return np.asarray(grayscale_cam[0, :])
        guard_mock("[HiResCAM] grad-cam package is not installed")
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        return np.asarray(np.exp(-(x * x + y * y)))

    def overlay(
        self, rgb_img: np.ndarray[Any, Any], grayscale_cam: np.ndarray[Any, Any], alpha: float = 0.5
    ) -> np.ndarray[Any, Any]:
        """Blend the CAM onto an un-normalized ``[H, W, 3]`` RGB display image in ``[0, 1]``."""
        if HAS_GRAD_CAM:
            visualization = show_cam_on_image(
                rgb_img, grayscale_cam, use_rgb=True, image_weight=1.0 - alpha
            )
            return np.asarray(visualization)
        cam_img = (grayscale_cam * 255).astype(np.uint8)
        heatmap = cv2.cvtColor(cv2.applyColorMap(cam_img, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        rgb_255 = (rgb_img * 255).astype(np.uint8)
        return np.asarray(cv2.addWeighted(rgb_255, 1.0 - alpha, heatmap, alpha, 0))


def side_by_side(
    image: np.ndarray[Any, Any],
    cam1_overlay: np.ndarray[Any, Any],
    cam2_overlay: np.ndarray[Any, Any],
    save_path: str | None = None,
) -> Any:
    """Create a 3-panel figure comparing original radiograph, Tier 1 focus, and Tier 2 focus."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(cam1_overlay)
    axes[1].set_title("Tier 1 (MobileNetV2) Focus")
    axes[1].axis("off")

    axes[2].imshow(cam2_overlay)
    axes[2].set_title("Tier 2 Specialist Focus")
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
    """Blend Tier 1 (red) and Tier 2 (blue) heatmaps over a grayscale radiograph."""
    rgb_heatmap = np.zeros_like(image)
    rgb_heatmap[:, :, 0] = grayscale_cam1
    rgb_heatmap[:, :, 2] = grayscale_cam2

    gray_image = np.repeat(np.mean(image, axis=2, keepdims=True), 3, axis=2)
    alpha = 0.4
    overlay = np.clip((1.0 - alpha) * gray_image + alpha * rgb_heatmap, 0, 1)
    return np.asarray(overlay)
