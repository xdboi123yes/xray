"""Structured logging middleware using structlog with JSON rendering."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import structlog
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware for structured logging of all incoming HTTP requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        # Attach request_id to context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            latency = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "request_processed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=round(latency, 2),
            )
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                latency_ms=round(latency, 2),
            )
            raise


def setup_logging(app: FastAPI) -> None:
    """Registers the structured logging middleware on the application."""
    app.add_middleware(StructuredLoggingMiddleware)
