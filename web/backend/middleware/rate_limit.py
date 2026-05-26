"""Rate limiting middleware using slowapi with IP-based limits."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Define Limiter with IP-based key function
limiter = Limiter(key_func=get_remote_address)

def setup_rate_limiting(app: FastAPI) -> None:
    """Registers slowapi rate limiter and error handlers on the application."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
