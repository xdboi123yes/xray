"""End-to-End integration test for the research TCR-P tiered classification pipeline.

Verifies the data flow from dummy dataset loaders through Tier 1 and Tier 2 training,
checkpoint serialization and reload, split-conformal calibration, and routed TCR-P
inference including Escalation and Human Deferral checks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.models.factory import ModelFactory
from core.models.tiered_system import TieredSystem
from core.uncertainty.conformal import ConformalPredictor
from infrastructure.training.observers.checkpoint_observer import CheckpointObserver
from infrastructure.training.trainer import Trainer


class E2EDummyDataset(Dataset[Any]):
    """Dummy dataset simulating preprocessed chest radiographs for E2E testing."""

    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img = torch.rand(3, 224, 224)
        label = idx % 2
        return img, label, f"img_e2e_{idx}"


def test_e2e_full_pipeline(tmp_path: Path) -> None:
    """Trains mini Tier 1/Tier 2 models, calibrates conformal q_hat, and routes predictions."""
    device = torch.device("cpu")

    t1_ckpt_path = str(tmp_path / "e2e_tier1_best.pth")
    t2_ckpt_path = str(tmp_path / "e2e_tier2_best.pth")
    cp_path = str(tmp_path / "e2e_q_hat.pt")

    train_loader = DataLoader(E2EDummyDataset(num_samples=4), batch_size=2)
    val_loader = DataLoader(E2EDummyDataset(num_samples=2), batch_size=2)

    # 1. Train mini Tier 1 (MobileNetV2)
    t1_model = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    t1_trainer = Trainer(
        model=t1_model,
        optimizer=torch.optim.Adam(t1_model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config={"training": {"epochs": 1, "seed": 42}},
        use_amp=False,
    )
    t1_trainer.add_observer(CheckpointObserver(checkpoint_path=t1_ckpt_path))
    t1_trainer.train(train_loader, val_loader)
    assert os.path.exists(t1_ckpt_path)

    # 2. Train mini Tier 2 (EfficientNet-B4)
    t2_model = ModelFactory.create("efficientnet_b4", num_classes=2, pretrained=False).to(device)
    t2_trainer = Trainer(
        model=t2_model,
        optimizer=torch.optim.Adam(t2_model.parameters(), lr=1e-3),
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config={"training": {"epochs": 1, "seed": 42}},
        use_amp=False,
    )
    t2_trainer.add_observer(CheckpointObserver(checkpoint_path=t2_ckpt_path))
    t2_trainer.train(train_loader, val_loader)
    assert os.path.exists(t2_ckpt_path)

    # 3. Calibrate and reload split-conformal predictor
    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(t2_model, val_loader, device)
    cp.save(cp_path)
    assert os.path.exists(cp_path)

    reloaded_cp = ConformalPredictor(alpha=0.1)
    reloaded_cp.load(cp_path)
    assert reloaded_cp.q_hat is not None

    # 4. Reload checkpoints into TCR-P and exercise Escalation to Tier 2
    t1_reloaded = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    t1_state = torch.load(t1_ckpt_path, map_location=device, weights_only=False)
    t1_reloaded.load_state_dict(t1_state.get("model_state_dict", t1_state))
    t1_reloaded.eval()

    t2_reloaded = ModelFactory.create("efficientnet_b4", num_classes=2, pretrained=False).to(device)
    t2_state = torch.load(t2_ckpt_path, map_location=device, weights_only=False)
    t2_reloaded.load_state_dict(t2_state.get("model_state_dict", t2_state))
    t2_reloaded.eval()

    system = TieredSystem(
        tier1_model=t1_reloaded,
        tier2_model=t2_reloaded,
        config={
            "model": {
                "confidence_threshold": 0.999,
                "threshold_window_size": 10,
                "threshold_delta": 0.05,
                "mc_dropout_passes": 2,
                "tta_passes": 2,
            }
        },
        conformal_predictor=reloaded_cp,
    )

    sample_x = torch.rand(1, 3, 224, 224)
    result = system.route(sample_x, image_id="e2e_sample_0")

    assert result.prediction in ("Pneumothorax", "No Finding")
    assert 0.0 <= result.confidence <= 1.0
    assert result.tier_used in (1, 2)
    assert result.inference_time_ms >= 0.0
    if result.tier_used == 2:
        assert result.mc_variance is not None and result.mc_variance >= 0.0
        assert result.conformal_set is not None and len(result.conformal_set) >= 1
        assert isinstance(result.flagged_for_review, bool)
