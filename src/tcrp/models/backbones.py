"""Tier 1 and Tier 2 backbone classifiers and registry-driven ModelFactory.

Tier 1 is MobileNetV2. Tier 2 is a single selected specialist backbone—either
EfficientNet-B4 or Ark+ Swin (never combined into an ensemble).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import structlog
import timm
import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B4_Weights,
    MobileNet_V2_Weights,
    efficientnet_b4,
    mobilenet_v2,
)

from tcrp.models.base import BaseClassifier

log = structlog.get_logger(__name__)

ArkVariant = Literal["base", "small", "tiny"]


class ModelFactory:
    """Registry-driven factory for instantiating Tier 1 and Tier 2 classifiers."""

    _registry: ClassVar[dict[str, type[BaseClassifier]]] = {}

    @classmethod
    def register(cls, key: str) -> Callable[[type[BaseClassifier]], type[BaseClassifier]]:
        """Register a :class:`BaseClassifier` subclass under ``key``."""

        def decorator(model_cls: type[BaseClassifier]) -> type[BaseClassifier]:
            if key in cls._registry:
                raise ValueError(f"Model with key '{key}' is already registered")
            cls._registry[key] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, model_type: str, **kwargs: Any) -> BaseClassifier:
        """Instantiate a registered classifier by key."""
        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model_type='{model_type}'. "
                f"Available registered models: {cls.list_models()}"
            )
        return cls._registry[model_type](**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """Return sorted keys of registered classifiers."""
        return sorted(cls._registry.keys())


@ModelFactory.register("mobilenet_v2")
class Tier1MobileNet(BaseClassifier):
    """Tier 1 lightweight MobileNetV2 screening classifier."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.backbone(x))


@ModelFactory.register("efficientnet_b4")
class Tier2EfficientNet(BaseClassifier):
    """Tier 2 specialist EfficientNet-B4 classifier."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b4(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.backbone(x))


@ModelFactory.register("ark_plus")
class Tier2ArkPlus(BaseClassifier):
    """Tier 2 specialist Ark+ Swin Transformer classifier with progressive unfreezing."""

    def __init__(
        self,
        num_classes: int = 2,
        variant: ArkVariant = "base",
        pretrained: bool = True,
        freeze_epochs: int = 5,
        ark_checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self._variant = variant
        self._freeze_epochs = freeze_epochs
        self.backbone, feat_dim = self._build_backbone(variant, pretrained, ark_checkpoint_path)

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, num_classes),
        )

        self._frozen = False
        if freeze_epochs > 0:
            self._freeze_backbone()

    def _build_backbone(
        self, variant: str, pretrained: bool, ark_path: str | None
    ) -> tuple[nn.Module, int]:
        model_name = f"swin_{variant}_patch4_window7_224"
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        if pretrained:
            target_path = (
                ark_path
                or os.environ.get("ARK_CHECKPOINT_PATH")
                or "outputs/models/ark_plus_swin_base.pth"
            )
            if target_path and Path(target_path).exists():
                state = torch.load(target_path, map_location="cpu", weights_only=False)
                if isinstance(state, dict):
                    for key in ("teacher", "state_dict", "model", "model_state_dict"):
                        if key in state and isinstance(state[key], dict):
                            state = state[key]
                            break
                if not isinstance(state, dict):
                    raise RuntimeError(f"Ark+ checkpoint at '{target_path}' is not a state dict.")
                incompatible = backbone.load_state_dict(state, strict=False)
                expected_keys = len(backbone.state_dict())
                matched_keys = expected_keys - len(incompatible.missing_keys)
                if matched_keys == 0:
                    raise RuntimeError(
                        f"Ark+ checkpoint '{target_path}' matched 0/{expected_keys} backbone keys "
                        f"(unexpected={len(incompatible.unexpected_keys)})."
                    )
                log.info(
                    "Loaded Ark+ checkpoint weights",
                    path=target_path,
                    matched=matched_keys,
                    missing=len(incompatible.missing_keys),
                    unexpected=len(incompatible.unexpected_keys),
                )
            else:
                log.info("Using ImageNet-pretrained Swin backbone", variant=variant)
        else:
            log.info("Initialized random-weight Swin backbone", variant=variant)

        feat_dim = int(getattr(backbone, "num_features", 1024))
        return backbone, feat_dim

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze_at_epoch(self, current_epoch: int) -> None:
        """Unfreeze backbone parameters once ``current_epoch >= freeze_epochs``."""
        if self._frozen and current_epoch >= self._freeze_epochs:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self._frozen = False
            log.info("Unfroze Ark+ backbone", epoch=current_epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return cast(torch.Tensor, self.classifier(feats))
