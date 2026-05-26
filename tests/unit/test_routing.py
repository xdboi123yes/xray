"""Unit tests for the static and dynamic routers.

Verifies static confidence limits and dynamic rolling window threshold updates
for StaticThresholdRouter and DynamicThresholdRouter.
"""

from core.routing.dynamic_router import DynamicThresholdRouter
from core.routing.static_router import StaticThresholdRouter


def test_static_threshold_router() -> None:
    """Test StaticThresholdRouter boundaries and routing decisions."""
    router = StaticThresholdRouter(threshold=0.75)

    assert router.current_threshold == 0.75

    # Should route to Tier 1 (1) if confidence >= threshold
    assert router.route(0.80) == 1
    assert router.route(0.75) == 1

    # Should escalate to Tier 2 (2) if confidence < threshold
    assert router.route(0.74) == 2
    assert router.route(0.50) == 2

    # Update should be a no-op
    router.update(0.10)
    assert router.current_threshold == 0.75


def test_dynamic_threshold_router_initial() -> None:
    """Test DynamicThresholdRouter initial routing behavior."""
    router = DynamicThresholdRouter(initial_threshold=0.75, window_size=5)

    assert router.current_threshold == 0.75
    assert router.route(0.78) == 1
    assert router.route(0.72) == 2


def test_dynamic_threshold_router_adaptation_low() -> None:
    """Test that DynamicThresholdRouter lowers threshold on difficult stream."""
    # Window size 3 for fast simulation
    router = DynamicThresholdRouter(
        initial_threshold=0.75, window_size=3, threshold_delta=0.05
    )

    # Stream very low confidences (hard cases)
    router.update(0.50)
    router.update(0.45)
    router.update(0.52)
    # The 4th update forces window eviction and recalculates rolling mean
    # rolling window: [0.50, 0.45, 0.52] -> mean = 0.49 (< 0.65)
    router.update(0.48)

    # Threshold should decrease
    assert router.current_threshold == 0.70


def test_dynamic_threshold_router_adaptation_high() -> None:
    """Test that DynamicThresholdRouter raises threshold on easy stream."""
    router = DynamicThresholdRouter(
        initial_threshold=0.75, window_size=3, threshold_delta=0.05
    )

    # Stream very high confidences (easy cases)
    router.update(0.90)
    router.update(0.95)
    router.update(0.88)
    # 4th update forces window eviction
    # rolling window: [0.90, 0.95, 0.88] -> mean = 0.91 (> 0.85)
    router.update(0.92)

    # Threshold should increase
    assert router.current_threshold == 0.80
