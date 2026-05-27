"""Dependency injection layer for the chest X-ray backend API.

Manages singleton system state, thread-safe model instantiations, routing threshold
updates, and connection pools for SQLite databases.
"""

from __future__ import annotations

import os
import threading

import torch
from fastapi import Depends

# Ensure model modules are registered in ModelFactory registry
import core.models.tier1_mobilenet
import core.models.tier2_ark
import core.models.tier2_efficientnet  # noqa: F401
from core.interfaces.base_model import BaseClassifier
from core.interfaces.base_router import BaseRouter
from core.models.factory import ModelFactory
from core.routing.dynamic_router import DynamicThresholdRouter
from core.routing.static_router import StaticThresholdRouter
from infrastructure.persistence.prediction_log import HistoryDatabaseManager


class SystemState:
    """Manages shared application singletons including diagnostic models and databases."""

    def __init__(self) -> None:
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tier1_model: BaseClassifier | None = None
        self.tier2_model: BaseClassifier | None = None
        self.router: BaseRouter | None = None
        self.db_manager: HistoryDatabaseManager | None = None
        self.threshold_value: float = 0.75
        self.threshold_mode: str = "static"
        self._lock = threading.Lock()

    def initialize(
        self,
        db_path: str = "outputs/history.db",
        t1_path: str = "outputs/models/best_tier1_mobilenet.pth",
        t2_path: str = "outputs/models/best_tier2_efficientnet.pth",
    ) -> None:
        """Instantiates all singletons and loads model parameters in a thread-safe manner."""
        with self._lock:
            # 1. Setup Database
            if self.db_manager is None:
                self.db_manager = HistoryDatabaseManager(db_path)

            # 2. Setup Router
            self._update_router_unlocked()

            # 3. Load Tier 1 Model (MobileNetV2)
            if self.tier1_model is None:
                model = ModelFactory.create("mobilenet_v2")
                if os.path.exists(t1_path):
                    # For PyTorch 2.6 safety, we load with weights_only=False for custom picklings if needed
                    # standard load_state_dict is safe
                    state_dict = torch.load(t1_path, map_location=self.device)
                    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.tier1_model = model

            # 4. Load Tier 2 Model (EfficientNetB4)
            if self.tier2_model is None:
                model = ModelFactory.create("efficientnet_b4")
                if os.path.exists(t2_path):
                    state_dict = torch.load(t2_path, map_location=self.device)
                    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                        state_dict = state_dict["model_state_dict"]
                    model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.tier2_model = model

    def update_threshold(self, value: float, mode: str | None = None) -> None:
        """Thread-safe runtime threshold and router configuration update."""
        with self._lock:
            self.threshold_value = value
            if mode is not None:
                self.threshold_mode = mode.lower()
            self._update_router_unlocked()

    def reload_tier(self, tier: int, backbone_name: str, weights_path: str) -> None:
        """Dynamically reloads a specific model tier with a new backbone and weights."""
        with self._lock:
            model = ModelFactory.create(backbone_name)
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            if tier == 1:
                self.tier1_model = model
            elif tier == 2:
                self.tier2_model = model
            else:
                raise ValueError(f"Invalid model tier: {tier}")


    def _update_router_unlocked(self) -> None:
        """Internal helper to instantiate the active router pattern (must hold lock)."""
        if self.threshold_mode == "dynamic":
            self.router = DynamicThresholdRouter(
                initial_threshold=self.threshold_value, window_size=50
            )
        else:
            self.router = StaticThresholdRouter(threshold=self.threshold_value)


# Global Singleton instance of SystemState
_global_state = SystemState()


def get_system_state() -> SystemState:
    """FastAPI Dependency providing access to the shared system state singleton."""
    return _global_state


def get_db(state: SystemState = Depends(get_system_state)) -> HistoryDatabaseManager:
    """FastAPI Dependency providing the history database manager."""
    if state.db_manager is None:
        # Graceful fallback initialization if lifespan event was skipped in test suites
        state.initialize()
    assert state.db_manager is not None
    return state.db_manager


def get_tier1_model(state: SystemState = Depends(get_system_state)) -> BaseClassifier:
    """FastAPI Dependency providing the loaded Tier 1 diagnostic model."""
    if state.tier1_model is None:
        state.initialize()
    assert state.tier1_model is not None
    return state.tier1_model


def get_tier2_model(state: SystemState = Depends(get_system_state)) -> BaseClassifier:
    """FastAPI Dependency providing the loaded Tier 2 diagnostic model."""
    if state.tier2_model is None:
        state.initialize()
    assert state.tier2_model is not None
    return state.tier2_model


def get_router(state: SystemState = Depends(get_system_state)) -> BaseRouter:
    """FastAPI Dependency providing the active routing algorithm."""
    if state.router is None:
        state.initialize()
    assert state.router is not None
    return state.router
