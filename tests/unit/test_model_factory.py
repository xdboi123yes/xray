"""Unit tests for the ModelFactory class.

Verifies dynamic registration, lists, instantiation, and error modes of ModelFactory.
"""

from typing import Any

import pytest

from core.interfaces.base_model import BaseClassifier
from core.models.factory import ModelFactory


def test_model_factory_registration() -> None:
    """Test that a new class can be registered dynamically in ModelFactory."""
    initial_models = ModelFactory.list_models()

    @ModelFactory.register("mock_model_for_test")
    class MockTestModel(BaseClassifier):
        """Mock classifier class for registration test."""

        def __init__(self, some_param: str = "default") -> None:
            super().__init__()
            self.some_param = some_param

        def forward(self, x: Any) -> Any:
            return x

        def get_confidence(self, logits: Any) -> Any:
            return logits

        def mc_forward(self, x: Any, T: int = 20) -> Any:
            return x

        def tta_forward(self, x: Any, n_augments: int = 10) -> Any:
            return x

        def mc_tta_forward(self, x: Any, T: int = 20, n_augments: int = 10) -> Any:
            return x

    assert "mock_model_for_test" in ModelFactory.list_models()
    assert len(ModelFactory.list_models()) == len(initial_models) + 1

    # Test instantiation
    instance = ModelFactory.create("mock_model_for_test", some_param="custom")
    assert isinstance(instance, BaseClassifier)
    assert instance.some_param == "custom"


def test_model_factory_duplicate_registration_raises_error() -> None:
    """Test that registering duplicate keys in ModelFactory raises ValueError."""
    # Ensure "mobilenet_v2" is registered (from porting tier1)
    assert "mobilenet_v2" in ModelFactory.list_models()

    with pytest.raises(ValueError, match="is already registered"):

        @ModelFactory.register("mobilenet_v2")
        class DuplicateModel(BaseClassifier):
            pass


def test_model_factory_unknown_key_raises_error() -> None:
    """Test that requesting an unregistered key in ModelFactory raises ValueError."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        ModelFactory.create("non_existent_key")
