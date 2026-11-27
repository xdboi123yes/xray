"""Confidence-routing strategies for Tier 1 to Tier 2 Escalation in TCR-P.

Provides both static threshold routing and the label-free Adaptive Routing
Threshold deadband controller over a sliding FIFO window of Tier-1 confidences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRouter(ABC):
    """Interface for Tier 1 -> Tier 2 confidence routing strategies."""

    @abstractmethod
    def route(self, confidence: float) -> int:
        """Return 1 if resolved by Tier 1, or 2 if escalated to Tier 2."""

    @abstractmethod
    def update(self, confidence: float) -> None:
        """Update internal routing state with the latest Tier-1 confidence."""

    @property
    @abstractmethod
    def current_threshold(self) -> float:
        """Return the active confidence threshold."""


class StaticThresholdRouter(BaseRouter):
    """Fixed confidence threshold router (baseline without adaptive control)."""

    def __init__(self, threshold: float = 0.75) -> None:
        self._threshold = threshold

    def route(self, confidence: float) -> int:
        """Resolve in Tier 1 if confidence >= threshold, else escalate to Tier 2."""
        return 1 if confidence >= self._threshold else 2

    def update(self, confidence: float) -> None:
        """No-op for static threshold routing."""

    @property
    def current_threshold(self) -> float:
        """Return the fixed confidence threshold."""
        return self._threshold


class DynamicThresholdRouter(BaseRouter):
    """Adaptive Routing Threshold deadband controller over a FIFO confidence window.

    Once the sliding window exceeds capacity ``W``, retains the most recent ``W``
    Tier-1 confidences and adjusts ``current_threshold`` inside ``[0.50, 0.95]``:
    - ``mean_conf < 0.65`` (hard stream): lower threshold by ``delta`` (floor ``0.50``)
      to prevent Tier-2 saturation.
    - ``mean_conf > 0.85`` (easy stream): raise threshold by ``delta`` (ceiling ``0.95``)
      so borderline cases still receive Tier-2 evaluation.
    - ``0.65 <= mean_conf <= 0.85``: hold threshold unchanged.
    """

    def __init__(
        self,
        initial_threshold: float = 0.75,
        window_size: int = 50,
        threshold_delta: float = 0.05,
    ) -> None:
        self._current_threshold = initial_threshold
        self._window_size = window_size
        self._threshold_delta = threshold_delta
        self._recent_confidences: list[float] = []

    def route(self, confidence: float) -> int:
        """Resolve in Tier 1 if confidence >= current_threshold, else escalate to Tier 2."""
        return 1 if confidence >= self._current_threshold else 2

    def update(self, confidence: float) -> None:
        """Record latest Tier-1 confidence and apply deadband threshold control."""
        self._recent_confidences.append(confidence)
        if len(self._recent_confidences) > self._window_size:
            self._recent_confidences.pop(0)
            mean_conf = sum(self._recent_confidences) / self._window_size

            if mean_conf < 0.65:
                self._current_threshold = max(0.5, self._current_threshold - self._threshold_delta)
            elif mean_conf > 0.85:
                self._current_threshold = min(
                    0.95, self._current_threshold + self._threshold_delta
                )

    @property
    def current_threshold(self) -> float:
        """Return the current Adaptive Routing Threshold."""
        return self._current_threshold
