"""Forward compatibility shim redirecting prediction history database manager to infrastructure."""

from __future__ import annotations

import warnings

from infrastructure.persistence.prediction_log import HistoryDatabaseManager

__all__ = ["HistoryDatabaseManager"]

warnings.warn(
    "web.backend.database is deprecated and will be removed in a future version. "
    "Use infrastructure.persistence.prediction_log instead.",
    DeprecationWarning,
    stacklevel=2,
)
