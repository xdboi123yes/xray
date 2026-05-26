"""Unit tests for decoupled MC Dropout and Test-Time Augmentation (TTA)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from core.uncertainty.mc_dropout import compute_predictive_entropy, compute_mutual_information
from core.uncertainty.tta import TestTimeAugmenter

class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.fc(x)

def test_compute_predictive_entropy() -> None:
    """Verify entropy calculations for highly certain vs uncertain probabilities."""
    certain = np.array([1.0, 0.0])
    uncertain = np.array([0.5, 0.5])

    h_certain = compute_predictive_entropy(certain)
    h_uncertain = compute_predictive_entropy(uncertain)

    assert h_certain < 1e-5
    assert abs(h_uncertain - np.log(2)) < 1e-5

def test_compute_mutual_information() -> None:
    """Verify mutual information breakdown yields positive values."""
    mc_probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
    ])
    pred_ent, exp_ent, mi = compute_mutual_information(mc_probs)
    assert pred_ent >= 0.0
    assert exp_ent >= 0.0
    assert mi >= 0.0
    assert pred_ent >= exp_ent

def test_tta_prediction_pipeline() -> None:
    """Verify TestTimeAugmenter transforms and averages predictions correctly."""
    augmenter = TestTimeAugmenter(n_augments=5)
    model = DummyModel()
    dummy_input = torch.randn(1, 3, 224, 224)

    mean_probs, variance = augmenter.predict_with_tta(model, dummy_input)
    assert mean_probs.shape == (1, 2)
    assert variance.shape == (1,)
    assert (mean_probs >= 0.0).all() and (mean_probs <= 1.0).all()
    assert (variance >= 0.0).all()
