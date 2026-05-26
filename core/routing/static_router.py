"""Static threshold router for tiered chest X-ray classification.

Escalates input images to Tier 2 based on a fixed confidence threshold.
"""

from core.interfaces.base_router import BaseRouter


class StaticThresholdRouter(BaseRouter):
    """Static threshold routing strategy.

    Compares input confidence against a fixed threshold to choose either Tier 1
    (fast economic path) or Tier 2 (robust escalated path).
    """

    def __init__(self, threshold: float = 0.75) -> None:
        """Initialize the static threshold router.

        Args:
            threshold: Fixed confidence score boundary (range 0.0 to 1.0).
        """
        self._threshold = threshold

    def route(self, confidence: float) -> int:
        """Determine tier routing based on static threshold comparison.

        Args:
            confidence: The prediction confidence of Tier 1.

        Returns:
            The routed tier number: 1 if confidence >= threshold, else 2.
        """
        return 1 if confidence >= self._threshold else 2

    def update(self, confidence: float) -> None:
        """No-op update for static thresholds.

        Args:
            confidence: The latest confidence score (ignored).
        """
        pass

    @property
    def current_threshold(self) -> float:
        """Get the active static threshold.

        Returns:
            The fixed confidence threshold.
        """
        return self._threshold
