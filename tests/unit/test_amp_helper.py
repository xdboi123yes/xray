"""Unit tests for MixedPrecisionContext manager."""

from __future__ import annotations

import torch

from infrastructure.training.amp_helper import MixedPrecisionContext


def test_mixed_precision_context_cpu() -> None:
    """Verify that MixedPrecisionContext executes without errors on CPU."""
    device = torch.device("cpu")
    context = MixedPrecisionContext(device=device, enabled=True)
    
    with context:
        # Perform a dummy tensor operation
        x = torch.randn(2, 2)
        y = x * 2.0
        assert y.shape == (2, 2)
