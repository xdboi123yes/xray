"""HiResCAM attribution for chest X-ray classifiers.

HiResCAM (Draelos & Carin, 2020) multiplies the target-layer activations by the
element-wise gradients *without* Grad-CAM's channel-wise gradient averaging,
producing a map that is provably faithful to the regions the model actually
used. This module also resolves the correct target layer for each supported
architecture, including a reshape transform for Swin/Ark+ transformer backbones
whose feature maps are token grids rather than ``[B, C, H, W]`` tensors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch

from core.integrity import guard_mock

try:
    from pytorch_grad_cam import HiResCAM  # type: ignore[import-untyped]
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore[import-untyped]

    HAS_GRAD_CAM = True
except ImportError:
    HAS_GRAD_CAM = False

ReshapeTransform = Callable[[torch.Tensor], torch.Tensor]


def _swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape a Swin token grid/sequence into ``[B, C, H, W]`` for CAM pooling.

    Recent timm Swin stages emit ``[B, H, W, C]``; older ones emit ``[B, L, C]``.
    Both are converted to channel-first spatial maps so the CAM can upsample them.
    """
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2).contiguous()
    batch, length, channels = tensor.shape
    side = int(round(length**0.5))
    return tensor.reshape(batch, side, side, channels).permute(0, 3, 1, 2).contiguous()


def resolve_target_layers(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Module], ReshapeTransform | None]:
    """Resolve the CAM target layer(s) and optional reshape transform for a model.

    Convolutional backbones (MobileNetV2 / EfficientNet-B4) expose
    ``backbone.features``; their final feature block is the standard CAM target
    and needs no reshape. Swin/Ark+ transformer backbones expose ``backbone.layers``
    and need the final stage's last norm layer plus a reshape transform.

    Args:
        model: A tiered classifier exposing a ``.backbone`` attribute.

    Returns:
        A tuple of (target_layers, reshape_transform). ``reshape_transform`` is
        None for convolutional backbones.

    Raises:
        ValueError: If no CAM-compatible target layer can be found.
    """
    backbone = getattr(model, "backbone", model)

    features = getattr(backbone, "features", None)
    if features is not None:
        # torchvision conv backbones: last feature block, already [B, C, H, W].
        return [features[-1]], None

    # timm Swin transformer (Ark+): final stage, last block's second norm.
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
    """Zero out CAM attribution over near-black (collimation/background) pixels.

    Chest radiographs frequently carry black collimation borders that hold no
    diagnostic signal. Without this, an imperfect model can place its highest
    attribution on those corners. Masking by image content keeps the heatmap on
    anatomy, then renormalizes so the colormap spans the in-body attribution.

    Args:
        cam: Grayscale CAM array [H, W] in [0, 1].
        rgb_img: Display image [H, W, 3] in [0, 1] at the same spatial size as ``cam``.
        threshold: Pixels with mean intensity at or below this are treated as background.
        blur_ksize: Odd kernel size used to soften the mask edges (1 disables blur).

    Returns:
        The masked, renormalized CAM array [H, W] in [0, 1].
    """
    gray = rgb_img.mean(axis=2) if rgb_img.ndim == 3 else rgb_img
    mask = (gray > threshold).astype(np.float32)
    if blur_ksize > 1:
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    masked = cam * mask
    peak = float(masked.max())
    if peak > 1e-8:
        masked = masked / peak
    return np.asarray(masked)


class XRayHiResCAM:
    """Computes and overlays HiResCAM attributions on input radiographs."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: list[torch.nn.Module],
        reshape_transform: ReshapeTransform | None = None,
    ) -> None:
        """Initialize XRayHiResCAM.

        Args:
            model: PyTorch classification model.
            target_layers: Target layers to capture activations and gradients.
            reshape_transform: Optional transform mapping a transformer feature
                map to ``[B, C, H, W]`` (required for Swin/Ark+ backbones).
        """
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
        """Generate a raw 2D grayscale CAM in ``[0, 1]``.

        Args:
            input_tensor: Normalized image tensor of shape [1, 3, H, W].
            target_category: Target class (defaults to the predicted class).

        Returns:
            Grayscale CAM array of shape [H, W] normalized in [0, 1].
        """
        if HAS_GRAD_CAM:
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=target_category)
            return np.asarray(grayscale_cam[0, :])
        guard_mock("[HiResCAM] grad-cam package is not installed")
        # Dry-run only (XRAY_ALLOW_MOCK=1): a centred gaussian, NOT a real attribution.
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        return np.asarray(np.exp(-(x * x + y * y)))

    def overlay(
        self, rgb_img: np.ndarray[Any, Any], grayscale_cam: np.ndarray[Any, Any], alpha: float = 0.5
    ) -> np.ndarray[Any, Any]:
        """Blend the CAM onto the (un-normalized) display image.

        Args:
            rgb_img: Display image of shape [H, W, 3] in range [0, 1].
            grayscale_cam: Grayscale CAM of shape [H, W] in range [0, 1].
            alpha: Opacity weight of the heatmap layer.

        Returns:
            Blended RGB image of shape [H, W, 3] as uint8 in range [0, 255].
        """
        if HAS_GRAD_CAM:
            visualization = show_cam_on_image(
                rgb_img, grayscale_cam, use_rgb=True, image_weight=1.0 - alpha
            )
            return np.asarray(visualization)
        cam_img = (grayscale_cam * 255).astype(np.uint8)
        heatmap = cv2.cvtColor(cv2.applyColorMap(cam_img, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        rgb_255 = (rgb_img * 255).astype(np.uint8)
        return np.asarray(cv2.addWeighted(rgb_255, 1.0 - alpha, heatmap, alpha, 0))
