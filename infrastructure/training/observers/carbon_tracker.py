"""Carbon footprint tracking observer for sustainable deep learning.

Uses the codecarbon library to calculate energy consumption and CO2 emissions
associated with training chest X-ray classifiers, helping compile sustainable
AI statistics for the bachelor thesis.
"""

from typing import Any

import structlog

from core.interfaces.base_observer import TrainingObserver

log = structlog.get_logger(__name__)


class CarbonTrackerObserver(TrainingObserver):
    """Observer that tracks CO2 emissions and energy footprints.

    Optionally hooks codecarbon EmissionsTracker to measure kWh consumption
    and environmental impact during model training.
    """

    def __init__(self, project_name: str = "chest-xray-tiered") -> None:
        """Initialize the carbon tracker observer.

        Args:
            project_name: The project name logged in emission reports.
        """
        self._project_name = project_name
        self._tracker: Any = None

    def on_train_start(self, trainer: Any) -> None:
        """Initialize and start the emissions tracker.

        Args:
            trainer: The active Trainer instance.
        """
        try:
            from codecarbon import EmissionsTracker

            # Configure tracker silently
            self._tracker = EmissionsTracker(
                project_name=self._project_name,
                log_level="warning",
            )
            self._tracker.start()
            log.info("[CarbonTracker] CodeCarbon EmissionsTracker started.")
        except ImportError:
            log.warning(
                "[CarbonTracker] Warning: 'codecarbon' library is not installed. "
                "Carbon tracking will be skipped."
            )
        except Exception as e:
            log.error(f"[CarbonTracker] Warning: Failed to start tracker: {e}")

    def on_train_end(self, trainer: Any) -> None:
        """Stop tracking and print calculated energy statistics.

        Args:
            trainer: The active Trainer instance.
        """
        if self._tracker is not None:
            try:
                emissions = self._tracker.stop()
                if emissions is not None:
                    log.info(
                        f"\n--> [CarbonTracker] Training complete! "
                        f"Cumulative CO2 emissions: {emissions:.6f} kg. "
                        f"Verify full energy logs in 'emissions.csv'"
                    )
            except Exception as e:
                log.error(f"[CarbonTracker] Warning: Failed to stop tracker cleanly: {e}")
