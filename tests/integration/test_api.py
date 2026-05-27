"""Integration tests for the chest X-ray backend FastAPI server.

Verifies health check, single image prediction, batch prediction, threshold updates,
SQLite history tracking, record deletes, and WebSocket diagnostic progress streams.
"""

from __future__ import annotations

import base64
import io
import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from infrastructure.persistence.prediction_log import HistoryDatabaseManager
from web.backend.app import app
from web.backend.deps import get_system_state


@pytest.fixture(scope="module")
def mock_png_bytes() -> bytes:
    """Generates a gray 224x224 PNG image in memory for uploading."""
    img = Image.new("RGB", (224, 224), color="gray")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Establishes clean testing folder and database locations for isolation."""
    test_db = "outputs/test_history.db"
    test_t1 = "outputs/models/best_tier1_mobilenet.pth"
    test_t2 = "outputs/models/best_tier2_efficientnet.pth"

    # Override state db to point to test_history.db, bypassing cached singleton pollution
    state = get_system_state()
    state.db_manager = HistoryDatabaseManager(test_db)
    state.initialize(db_path=test_db, t1_path=test_t1, t2_path=test_t2)

    yield

    # Clean up test database after tests finish
    if os.path.exists(test_db):
        os.remove(test_db)


def test_health_endpoint() -> None:
    """Verifies the health check endpoint returns 200 and loads models."""
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "gpu" in data
        assert "models_loaded" in data
        assert "uptime_s" in data
        assert isinstance(data["models_loaded"], list)


def test_predict_single_endpoint(mock_png_bytes: bytes) -> None:
    """Verifies that uploading a PNG returns a valid PredictionDTO and saves history."""
    with TestClient(app) as client:
        files = {"file": ("test.png", mock_png_bytes, "image/png")}
        response = client.post(
            "/api/v1/predict",
            files=files,
            params={"return_gradcam": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert data["prediction"] in ("Pneumothorax", "No Finding")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["tier_used"] in (1, 2)
        assert "inference_time_ms" in data
        assert "timestamp" in data
        assert "gradcam_tier1_b64" in data or "gradcam_tier2_b64" in data


def test_predict_invalid_file_type() -> None:
    """Verifies uploading an invalid file type (e.g. text) returns 400 Bad Request."""
    with TestClient(app) as client:
        files = {"file": ("test.txt", b"plain text", "text/plain")}
        response = client.post("/api/v1/predict", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["code"] == "INVALID_INPUT"


def test_predict_batch_endpoint(mock_png_bytes: bytes) -> None:
    """Verifies batch diagnostic endpoint under batch capacity limits."""
    with TestClient(app) as client:
        files = [
            ("files", ("test1.png", mock_png_bytes, "image/png")),
            ("files", ("test2.png", mock_png_bytes, "image/png")),
        ]
        response = client.post("/api/v1/predict/batch", files=files)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert "request_id" in data[0]
        assert "request_id" in data[1]


def test_threshold_get_and_put() -> None:
    """Verifies runtime threshold updates and router modifications."""
    with TestClient(app) as client:
        # 1. Get initial threshold settings
        response = client.get("/api/v1/threshold")
        assert response.status_code == 200
        data = response.json()
        assert data["value"] == 0.75
        assert data["mode"] == "static"

        # 2. Put updated threshold settings
        update_payload = {"value": 0.88, "mode": "dynamic"}
        response = client.put("/api/v1/threshold", json=update_payload)
        assert response.status_code == 200
        updated_data = response.json()
        assert updated_data["value"] == 0.88
        assert updated_data["mode"] == "dynamic"

        # 3. Verify changes were applied on state singleton
        state = get_system_state()
        assert state.threshold_value == 0.88
        assert state.threshold_mode == "dynamic"

        # 4. Restore defaults
        client.put("/api/v1/threshold", json={"value": 0.75, "mode": "static"})


def test_history_and_delete_endpoints(mock_png_bytes: bytes) -> None:
    """Verifies paginated history lists and record deletion operations."""
    with TestClient(app) as client:
        # 1. Clear database before run
        state = get_system_state()
        db = state.db_manager
        assert db is not None
        db.clear_history()

        # 2. Add single prediction entry to populate DB
        files = {"file": ("test.png", mock_png_bytes, "image/png")}
        response = client.post("/api/v1/predict", files=files)
        pred_data = response.json()
        req_id = pred_data["request_id"]

        # 3. Query history logs
        hist_response = client.get("/api/v1/history", params={"limit": 10})
        assert hist_response.status_code == 200
        hist_data = hist_response.json()
        assert len(hist_data) == 1
        assert hist_data[0]["request_id"] == req_id

        # 4. Delete the logged entry
        del_response = client.delete(f"/api/v1/history/{req_id}")
        assert del_response.status_code == 200
        assert del_response.json()["status"] == "deleted"

        # 5. Verify history is empty
        hist_response_empty = client.get("/api/v1/history")
        assert len(hist_response_empty.json()) == 0


def test_ablation_and_models_endpoints() -> None:
    """Verifies static ablation results and active model layers details."""
    with TestClient(app) as client:
        # Ablations checklist
        response_abl = client.get("/api/v1/ablation")
        assert response_abl.status_code == 200
        data_abl = response_abl.json()
        assert isinstance(data_abl, list)
        assert data_abl[0]["ablation_id"] == "A1"

        # Model configs metadata
        response_mod = client.get("/api/v1/models")
        assert response_mod.status_code == 200
        data_mod = response_mod.json()
        assert "tier1" in data_mod
        assert "tier2" in data_mod


def test_websocket_diagnostic_stream(mock_png_bytes: bytes) -> None:
    """Verifies real-time WebSocket connection diagnostic steps sequence."""
    client = TestClient(app)
    # Convert mock image bytes to base64 string
    image_b64 = base64.b64encode(mock_png_bytes).decode("utf-8")
    payload = {
        "type": "upload",
        "image_b64": f"data:image/png;base64,{image_b64}",
    }

    with client.websocket_connect("/api/v1/ws/predict") as websocket:
        # 1. Client uploads base64 radiograph
        websocket.send_json(payload)

        # 2. Server streams progress: preprocessing
        frame_proc = websocket.receive_json()
        assert frame_proc["type"] == "progress"
        assert frame_proc["step"] == "preprocessing"
        assert frame_proc["percent"] == 20

        # 3. Server streams progress: tier 1 inference
        frame_t1 = websocket.receive_json()
        assert frame_t1["type"] == "progress"
        assert frame_t1["step"] == "tier1_inference"
        assert frame_t1["percent"] == 40

        # 4. Server streams progress: tier 2 routing/skipping
        frame_routing = websocket.receive_json()
        assert frame_routing["type"] == "progress"
        assert frame_routing["step"] in ("tier2_escalation", "skipping_tier2")
        assert frame_routing["percent"] == 70

        # 5. Server streams progress: Grad-CAM heatmap overlay
        frame_gcam = websocket.receive_json()
        assert frame_gcam["type"] == "progress"
        assert frame_gcam["step"] == "gradcam_generation"
        assert frame_gcam["percent"] == 90

        # 6. Server streams final diagnostic prediction payload
        frame_res = websocket.receive_json()
        assert frame_res["type"] == "result"
        result_dto = frame_res["data"]
        assert "request_id" in result_dto
        assert result_dto["prediction"] in ("Pneumothorax", "No Finding")
        assert 0.0 <= result_dto["confidence"] <= 1.0
        assert result_dto["tier_used"] in (1, 2)
