"""Inference Service coordinating tiered radiograph classification pipelines.

Takes raw image bytes, runs rapid screening or escalated diagnostics,
and returns structured Pydantic DTO models.
"""

from __future__ import annotations

import base64
import io
import time
import uuid
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms  # type: ignore[import-untyped]
from PIL import Image

from application.dto.prediction_dto import PredictionDTO
from core.explainability.hirescam import (
    XRayHiResCAM,
    resolve_target_layers,
    suppress_background,
)
from core.interfaces.base_model import BaseClassifier
from core.interfaces.base_router import BaseRouter


class InferenceService:
    """Manages full execution pipeline for tiered chest radiograph classification."""

    def __init__(self) -> None:
        """Initialize InferenceService."""
        self.image_size = 224
        # Model input MUST match the training/eval pipeline, which applies ImageNet
        # normalization (see ClassicalAugmentation). Feeding un-normalized [0, 1]
        # images is out-of-distribution for the models and produces unreliable
        # predictions and border-focused Grad-CAM maps.
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        # Un-normalized image (in [0, 1]) used only as the Grad-CAM overlay base.
        self.display_preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def preprocess_image(self, file_bytes: bytes) -> torch.Tensor:
        """Preprocesses raw uploaded image bytes into a normalized PyTorch tensor.

        Args:
            file_bytes: Raw bytes from radiograph upload.

        Returns:
            An ImageNet-normalized float tensor of shape [1, 3, 224, 224].
        """
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return self.preprocess(image).unsqueeze(0)

    def display_image(self, file_bytes: bytes) -> np.ndarray[Any, Any]:
        """Return the resized, un-normalized RGB image in [0, 1] for CAM overlays.

        Args:
            file_bytes: Raw bytes from radiograph upload.

        Returns:
            Float array of shape [H, W, 3] in range [0, 1].
        """
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.asarray(self.display_preprocess(image).permute(1, 2, 0).numpy())

    def _generate_gradcam_b64(
        self,
        model: BaseClassifier,
        tensor: torch.Tensor,
        display_img: np.ndarray[Any, Any],
        device: torch.device,
    ) -> str | None:
        """Generates a base64-encoded HiResCAM overlay safely.

        Args:
            model: Target model classifier.
            tensor: Normalized image tensor of shape [1, 3, 224, 224].
            display_img: Un-normalized RGB image [H, W, 3] in [0, 1] for the overlay.
            device: Active computing device.

        Returns:
            Base64 PNG data URL or None if generation failed.
        """
        try:
            # Target layers (and a reshape transform for Swin/Ark+) are resolved
            # from the model's own backbone, never from a hardcoded backbone name.
            target_layers, reshape_transform = resolve_target_layers(model)
            cam = XRayHiResCAM(model, target_layers, reshape_transform=reshape_transform)
            heatmap = cam.generate(tensor.to(device))

            # Overlay onto the un-normalized image so the heatmap aligns with what
            # the clinician sees, not the mean-subtracted model input.
            rgb_img = np.clip(display_img, 0.0, 1.0).astype(np.float32)
            # Keep attribution off the black collimation border (no diagnostic signal).
            heatmap = suppress_background(heatmap, rgb_img)
            visualization = cam.overlay(rgb_img, heatmap, alpha=0.45)

            vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".png", vis_bgr)
            b64_str = base64.b64encode(buffer).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"
        except Exception:
            # Fallback to prevent complete request failure if the CAM library throws.
            return None

    def predict(
        self,
        image_bytes: bytes,
        tier1: BaseClassifier,
        tier2: BaseClassifier,
        router: BaseRouter,
        device: torch.device,
        return_gradcam: bool = True,
        conformal_predictor: Any = None,
    ) -> PredictionDTO:
        """Runs the complete tiered diagnostics pipeline.

        Args:
            image_bytes: Uploaded radiograph file contents in bytes.
            tier1: Lightweight MobileNetV2 screening model.
            tier2: Heavyweight specialist classifier.
            router: Confidence routing model wrapper.
            device: Compute acceleration target.
            return_gradcam: Whether to synthesize Grad-CAM attention overlays.
            conformal_predictor: Conformal intervals lookup utility.

        Returns:
            Structured PredictionDTO containing prediction decisions and analytics.
        """
        start_time = time.perf_counter()
        tensor = self.preprocess_image(image_bytes)
        display_img = self.display_image(image_bytes)
        tensor_device = tensor.to(device)

        # 1. Tier 1 Inference (Lightweight screening)
        with torch.no_grad():
            t1_out = tier1(tensor_device)
            t1_conf_t, t1_pred_t = tier1.get_confidence(t1_out)
            t1_conf = float(t1_conf_t[0].item())
            t1_pred = int(t1_pred_t[0].item())

        # 2. Threshold Routing decision
        routed_tier = router.route(t1_conf)
        router.update(t1_conf)

        prediction_idx = t1_pred
        confidence = t1_conf
        mc_variance = None
        flagged = False
        gcam_t1 = None
        gcam_t2 = None

        if routed_tier == 2:
            # 3. Escalated deep diagnostics via specialized Tier 2 model
            # Uses MC Dropout T=10 to determine predictive uncertainty variance
            mean_probs, variances = tier2.mc_forward(tensor_device, T=10)
            t2_prob = float(mean_probs[0, 1].item())  # Pneumothorax positive probability
            mc_variance = float(variances[0, 1].item())

            prediction_idx = 1 if t2_prob >= 0.5 else 0
            confidence = t2_prob if prediction_idx == 1 else (1.0 - t2_prob)

            # High uncertainty triggers clinical warning flags
            if mc_variance > 0.08:
                flagged = True

            if return_gradcam:
                gcam_t2 = self._generate_gradcam_b64(tier2, tensor, display_img, device)
        else:
            # Bypassed escalation
            if return_gradcam:
                gcam_t1 = self._generate_gradcam_b64(tier1, tensor, display_img, device)

        # Label Mapping
        prediction = "Pneumothorax" if prediction_idx == 1 else "No Finding"

        # Conformal prediction set. Only a real, calibrated ConformalPredictor provides the
        # coverage guarantee; without one we fall back to a plain uncertainty-based set and
        # report NO coverage figure, so the UI never claims a 95% guarantee it cannot back.
        if conformal_predictor and getattr(conformal_predictor, "q_hat", None) is not None:
            with torch.no_grad():
                probs_val = (
                    torch.softmax(tier2(tensor_device), dim=1)
                    if routed_tier == 2
                    else torch.softmax(t1_out, dim=1)
                )
                conformal_set = conformal_predictor.predict_set(probs_val)
                conformal_coverage = 1.0 - conformal_predictor.alpha
        else:
            if mc_variance is not None and mc_variance > 0.12:
                conformal_set = ["Pneumothorax", "No Finding"]
            else:
                conformal_set = [prediction]
            conformal_coverage = None

        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000.0

        req_id = str(uuid.uuid4())
        now_iso = datetime.now().isoformat() + "Z"

        # Human-readable name of the model that actually produced the decision (the UI labels
        # this "Active Specialist Model"). Derived from the real class, never a hardcoded string.
        friendly_names = {
            "Tier1MobileNet": "MobileNetV2 (Tier 1)",
            "Tier2EfficientNet": "EfficientNet-B4 (Tier 2)",
            "Tier2ArkPlus": "Ark+ Swin (Tier 2)",
        }
        active = tier2 if routed_tier == 2 else tier1
        active_model_name = friendly_names.get(type(active).__name__, type(active).__name__)

        return PredictionDTO(
            request_id=req_id,
            prediction=prediction,
            confidence=confidence,
            tier_used=routed_tier,
            mc_variance=mc_variance,
            mc_passes=10 if routed_tier == 2 else None,
            tta_passes=None,
            conformal_set=conformal_set,
            conformal_coverage=conformal_coverage,
            flagged_for_review=flagged,
            inference_time_ms=inference_time,
            gradcam_tier1_b64=gcam_t1,
            gradcam_tier2_b64=gcam_t2,
            model_version=active_model_name,
            timestamp=now_iso,
        )
