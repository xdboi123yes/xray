"""FastAPI custom middleware registration package."""

from __future__ import annotations

from web.backend.middleware.cors import setup_cors
from web.backend.middleware.error_handler import setup_error_handlers
from web.backend.middleware.logging import setup_logging
from web.backend.middleware.rate_limit import setup_rate_limiting, limiter

__all__ = [
    "setup_cors",
    "setup_error_handlers",
    "setup_logging",
    "setup_rate_limiting",
    "limiter",
]
