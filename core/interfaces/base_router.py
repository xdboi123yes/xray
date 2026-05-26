"""Base router strategy interface for the Chest X-Ray classification system.

This module defines the abstract base class for routing strategies that determine
whether to escalate predictions from Tier 1 to Tier 2.
"""

from abc import ABC, abstractmethod


class BaseRouter(ABC):
    """Abstract base class for routing strategies.

    Enforces the strategy pattern for static and dynamic threshold routing
    mechanisms within the tiered system.
    """

    @abstractmethod
    def route(self, confidence: float) -> int:
        """Determine which tier to route the prediction to based on confidence.

        Args:
            confidence: The prediction confidence output by Tier 1.

        Returns:
            The tier number (1 for Tier 1, 2 for Tier 2).
        """
        pass

    @abstractmethod
    def update(self, confidence: float) -> None:
        """Update any adaptive threshold states using the latest prediction confidence.

        Args:
            confidence: The latest prediction confidence to record in internal state.
        """
        pass

    @property
    @abstractmethod
    def current_threshold(self) -> float:
        """Get the current decision threshold value.

        Returns:
            The current active confidence threshold.
        """
        pass
