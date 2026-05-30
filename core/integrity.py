"""Mock-data guard.

Keeps fabricated / dry-run placeholder data from silently entering a real run.
Every code path that would otherwise quietly substitute fake data for a missing
real input must call :func:`guard_mock` first, so a real experiment can never
produce results from fabricated data without an explicit, auditable opt-in.

Set the environment variable ``XRAY_ALLOW_MOCK=1`` to permit placeholder data
(only for dry-runs / CI / demos).
"""

from __future__ import annotations

import os

ALLOW_MOCK_ENV = "XRAY_ALLOW_MOCK"


def mock_allowed() -> bool:
    """Return True only when ``XRAY_ALLOW_MOCK=1`` is set in the environment."""
    return os.getenv(ALLOW_MOCK_ENV) == "1"


def guard_mock(context: str) -> None:
    """Raise unless placeholder/mock data is explicitly permitted.

    Args:
        context: Human-readable description of what is missing, included in the error.

    Raises:
        RuntimeError: If ``XRAY_ALLOW_MOCK`` is not ``"1"``.
    """
    if not mock_allowed():
        raise RuntimeError(
            f"{context}: refusing to fabricate placeholder data in a real run "
            f"(a real input is probably missing). Set {ALLOW_MOCK_ENV}=1 to "
            f"explicitly allow mock/dry-run data."
        )
