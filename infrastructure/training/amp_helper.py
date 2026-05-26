"""AMP (Automatic Mixed Precision) helper module for clean context management."""

from __future__ import annotations

from typing import Any
import torch

class MixedPrecisionContext:
    """Context manager wrapping PyTorch's automatic mixed precision (AMP) safely."""

    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        """Initialize MixedPrecisionContext.

        Args:
            device: Target hardware device (e.g. cuda, cpu).
            enabled: If True, uses autocast context.
        """
        self.device = device
        self.enabled = enabled
        # PyTorch autocast supports device_type parameter ('cuda' or 'cpu')
        device_type = "cuda" if device.type == "cuda" else "cpu"
        self.autocast = torch.autocast(
            device_type=device_type,
            enabled=enabled,
            dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
        )

    def __enter__(self) -> Any:
        return self.autocast.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        return self.autocast.__exit__(exc_type, exc_val, exc_tb)
