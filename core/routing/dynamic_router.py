"""Dynamic threshold router for tiered chest X-ray classification.

Escalates input images to Tier 2 based on an adaptive threshold that shifts
dynamically according to rolling statistics of recent prediction confidences.
"""

from core.interfaces.base_router import BaseRouter


class DynamicThresholdRouter(BaseRouter):
    """Dynamic adaptive threshold routing strategy.

    Maintains a rolling window of recent prediction confidences and shifts the
    escalation threshold dynamically to balance throughput and medical accuracy.
    """

    def __init__(
        self,
        initial_threshold: float = 0.75,
        window_size: int = 50,
        threshold_delta: float = 0.05,
    ) -> None:
        """Initialize the dynamic threshold router.

        Args:
            initial_threshold: The starting confidence routing boundary.
            window_size: The capacity of the rolling statistics window.
            threshold_delta: The increment value for threshold adjustments.
        """
        self._current_threshold = initial_threshold
        self._window_size = window_size
        self._threshold_delta = threshold_delta
        self._recent_confidences: list[float] = []

    def route(self, confidence: float) -> int:
        """Determine tier routing based on the active dynamic threshold.

        Args:
            confidence: The prediction confidence of Tier 1.

        Returns:
            The routed tier number: 1 if confidence >= current_threshold, else 2.
        """
        return 1 if confidence >= self._current_threshold else 2

    def update(self, confidence: float) -> None:
        """Update the rolling statistics and dynamically shift the threshold.

        Saves recent confidence. If window capacity is reached, calculates the
        rolling mean. If mean < 0.65 (hard stream), lowers threshold to route more
        cases to Tier 2. If mean > 0.85 (easy stream), raises threshold to keep
        throughput fast.

        Args:
            confidence: The latest confidence score to record.
        """
        self._recent_confidences.append(confidence)

        if len(self._recent_confidences) > self._window_size:
            self._recent_confidences.pop(0)

            mean_conf = sum(self._recent_confidences) / self._window_size

            # Adaptive adjustment logic
            if mean_conf < 0.65:
                # Many hard cases: route more aggressively to Tier 2 by lowering threshold
                self._current_threshold = max(0.5, self._current_threshold - self._threshold_delta)
            elif mean_conf > 0.85:
                # Many easy cases: keep throughput high by raising threshold
                self._current_threshold = min(0.95, self._current_threshold + self._threshold_delta)

    @property
    def current_threshold(self) -> float:
        """Get the active dynamically-adjusted threshold.

        Returns:
            The current active confidence threshold.
        """
        return self._current_threshold
