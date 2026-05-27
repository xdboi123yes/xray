"""End-to-End integration tests for the full tiered classification pipeline.

Verifies the entire data flow from dummy/mini dataset loaders, Trainer executions,
model checkpoint serialization, and dynamic model route inference via the FastAPI client.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from torch.utils.data import DataLoader, Dataset

from core.models.factory import ModelFactory
from core.uncertainty.conformal import ConformalPredictor
from infrastructure.training.observers.checkpoint_observer import CheckpointObserver
from infrastructure.training.trainer import Trainer
from web.backend.app import app
from web.backend.deps import get_system_state


class E2EDummyDataset(Dataset[Any]):
    """Dummy dataset simulating raw radiographs and label loaders for E2E testing."""

    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        # Grayscale dummy image tensor [3, 224, 224]
        img = torch.rand(3, 224, 224)
        label = idx % 2
        return img, label, f"img_e2e_{idx}"


def test_e2e_full_pipeline(tmp_path: Any) -> None:
    """Trains mini models, serializes checkpoints, and queries the REST API for routed predictions."""
    device = torch.device("cpu")

    # 1. Setup paths under temporary folder to prevent workspace pollution
    t1_ckpt_path = os.path.join(tmp_path, "e2e_tier1_best.pth")
    t2_ckpt_path = os.path.join(tmp_path, "e2e_tier2_best.pth")
    cp_path = os.path.join(tmp_path, "e2e_q_hat.pt")
    test_db_path = os.path.join(tmp_path, "e2e_history.db")

    # 2. Train a mini Tier 1 MobileNetV2 for 1 epoch
    t1_model = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False).to(device)
    t1_optimizer = torch.optim.Adam(t1_model.parameters(), lr=1e-3)
    t1_criterion = nn.CrossEntropyLoss()

    t1_trainer = Trainer(
        model=t1_model,
        optimizer=t1_optimizer,
        criterion=t1_criterion,
        device=device,
        config={"training": {"epochs": 1, "seed": 42}},
        use_amp=False,
    )
    t1_trainer.add_observer(CheckpointObserver(checkpoint_path=t1_ckpt_path))

    train_loader = DataLoader(E2EDummyDataset(num_samples=4), batch_size=2)
    val_loader = DataLoader(E2EDummyDataset(num_samples=2), batch_size=2)

    t1_trainer.train(train_loader, val_loader)
    assert os.path.exists(t1_ckpt_path)

    # 3. Train a mini Tier 2 EfficientNetB4 for 1 epoch
    t2_model = ModelFactory.create("efficientnet_b4", num_classes=2, pretrained=False).to(device)
    t2_optimizer = torch.optim.Adam(t2_model.parameters(), lr=1e-3)
    t2_criterion = nn.CrossEntropyLoss()

    t2_trainer = Trainer(
        model=t2_model,
        optimizer=t2_optimizer,
        criterion=t2_criterion,
        device=device,
        config={"training": {"epochs": 1, "seed": 42}},
        use_amp=False,
    )
    t2_trainer.add_observer(CheckpointObserver(checkpoint_path=t2_ckpt_path))

    t2_trainer.train(train_loader, val_loader)
    assert os.path.exists(t2_ckpt_path)

    # 4. Calibrate a Conformal Predictor using the trained Tier 2 model
    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(t2_model, val_loader, device)
    cp.save(cp_path)
    assert os.path.exists(cp_path)

    # 5. Initialize dynamic state of the FastAPI application with these mini weights
    from infrastructure.persistence.prediction_log import HistoryDatabaseManager

    state = get_system_state()
    state.db_manager = HistoryDatabaseManager(test_db_path)
    state.initialize(
        db_path=test_db_path,
        t1_path=t1_ckpt_path,
        t2_path=t2_ckpt_path,
    )

    # Force reload models in state to use these new weights
    state.reload_tier(1, "mobilenet_v2", t1_ckpt_path)
    state.reload_tier(2, "efficientnet_b4", t2_ckpt_path)

    # Load conformal calibrations in state
    state.conformal_predictor = cp

    # 6. Run E2E client inference through FastAPI client
    with TestClient(app) as client:
        # Create a mock image in bytes
        import io

        from PIL import Image

        img_pil = Image.new("RGB", (224, 224), color="gray")
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Query single predict REST endpoint
        files = {"file": ("e2e_test.png", img_bytes, "image/png")}
        response = client.post(
            "/api/v1/predict",
            files=files,
            params={"return_gradcam": True},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify complete DTO validation
        assert "request_id" in data
        assert data["prediction"] in ("Pneumothorax", "No Finding")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["tier_used"] in (1, 2)
        assert "inference_time_ms" in data
        assert isinstance(data["conformal_set"], list)
        assert len(data["conformal_set"]) > 0

        # Query dynamic threshold configurations
        response_th = client.get("/api/v1/threshold")
        assert response_th.status_code == 200
        assert response_th.json()["value"] == 0.75

        # Query history list
        response_hist = client.get("/api/v1/history")
        assert response_hist.status_code == 200
        assert len(response_hist.json()) == 1
        assert response_hist.json()[0]["request_id"] == data["request_id"]

    # 7. Clean up E2E generated artifacts
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
