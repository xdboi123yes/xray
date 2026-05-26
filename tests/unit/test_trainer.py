"""Unit tests for the Trainer engine.

Verifies creation, custom optimizer and criteria configurations, and observer
registrations for the Trainer class.
"""

from typing import Any

import torch
import torch.nn as nn

from core.interfaces.base_observer import TrainingObserver
from core.models.tier1_mobilenet import Tier1MobileNet
from infrastructure.training.trainer import Trainer


class MockObserver(TrainingObserver):
    """Mock observer to verify event registrations."""

    def __init__(self) -> None:
        self.start_called = False

    def on_train_start(self, trainer: Any) -> None:
        self.start_called = True


def test_trainer_init(mock_config: dict[str, Any]) -> None:
    """Test Trainer initialization and state setups.

    Args:
        mock_config: Pytest fixture representing mock parameters.
    """
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=mock_config,
        use_amp=False,
        accumulate_grad_batches=2,
        gradient_clip_val=1.5,
    )

    assert trainer.model is model
    assert trainer.optimizer is optimizer
    assert trainer.criterion is criterion
    assert trainer.device == device
    assert trainer.accumulate_grad_batches == 2
    assert trainer.gradient_clip_val == 1.5
    assert not trainer.use_amp


def test_trainer_observer_registration(mock_config: dict[str, Any]) -> None:
    """Test that observers can be registered cleanly to the Trainer engine.

    Args:
        mock_config: Pytest fixture representing mock parameters.
    """
    model = Tier1MobileNet(num_classes=2, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=mock_config,
    )

    obs = MockObserver()
    trainer.add_observer(obs)

    assert obs in trainer._observers
