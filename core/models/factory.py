"""Registry-driven factory for tiered classifier instantiation.

This module provides the ModelFactory class which decouples model registration
and instantiation from configuration, enabling flexible runtime selection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from core.interfaces.base_model import BaseClassifier


class ModelFactory:
    """Registry-driven factory for model creation.

    Enables dynamic registration and initialization of classifiers using keys
    specified in configurations (e.g. config.yaml).
    """

    _registry: ClassVar[dict[str, type[BaseClassifier]]] = {}

    @classmethod
    def register(cls, key: str) -> Callable[[type[BaseClassifier]], type[BaseClassifier]]:
        """Decorator to register a classifier class to the factory.

        Args:
            key: The unique string identifier for the model.

        Returns:
            A decorator function that registers the class.

        Raises:
            ValueError: If a model class with the given key is already registered.
        """

        def decorator(model_cls: type[BaseClassifier]) -> type[BaseClassifier]:
            if key in cls._registry:
                raise ValueError(f"Model with key '{key}' is already registered")
            cls._registry[key] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, model_type: str, **kwargs: Any) -> BaseClassifier:
        """Instantiate a registered model by its type key.

        Args:
            model_type: Unique string identifier for the model.
            **kwargs: Dynamic arguments passed to the model's constructor.

        Returns:
            An instance of a class derived from BaseClassifier.

        Raises:
            ValueError: If the requested model_type is not registered.
        """
        if not cls._registry:
            # Dynamically import model modules to trigger their registration decorators
            from core.models import tier1_mobilenet, tier2_ark, tier2_efficientnet  # noqa: F401

        if model_type not in cls._registry:
            raise ValueError(
                f"Unknown model_type='{model_type}'. "
                f"Available registered models: {cls.list_models()}"
            )
        return cls._registry[model_type](**kwargs)

    @classmethod
    def list_models(cls) -> list[str]:
        """Get the sorted list of all registered model type keys.

        Returns:
            List of registered model keys.
        """
        if not cls._registry:
            # Dynamically import model modules to trigger their registration decorators
            from core.models import tier1_mobilenet, tier2_ark, tier2_efficientnet  # noqa: F401

        return sorted(cls._registry.keys())
