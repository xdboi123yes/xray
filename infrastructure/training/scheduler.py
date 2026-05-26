"""Learning rate schedulers for neural network optimization.

Provides configurations for scheduling plateau learning rate drops.
"""

from __future__ import annotations

from typing import Any

import torch

from config.settings import Settings, get_settings


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any] | Settings | None = None,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Creates a learning rate scheduler monitoring validation metric performance.

    Defaults to ReduceLROnPlateau scheduler.

    Args:
        optimizer: The PyTorch optimizer instance to govern.
        config: Project settings (Pydantic Settings or raw dict).

    Returns:
        The configured ReduceLROnPlateau learning rate scheduler.
    """
    if config is None:
        settings = get_settings()
    elif isinstance(config, dict):
        settings = Settings(**config)
    else:
        settings = config

    # Retrieve patient epochs from configuration or fall back to default
    patience = getattr(settings.training, "early_stopping_patience", 7) // 2
    if patience < 1:
        patience = 3

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=patience,
        min_lr=1e-6,
    )
    return scheduler
