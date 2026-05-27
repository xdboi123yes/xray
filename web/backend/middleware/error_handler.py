"""Error handler middleware for catching and standardizing application-wide exceptions."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Catches all unhandled exceptions and standardizes error responses."""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": f"Internal Server Error: {exc!s}",
            "code": "INTERNAL_ERROR",
        },
    )


async def value_error_handler(request: Any, exc: ValueError) -> JSONResponse:
    """Catches input validation failures."""
    logger.warning("validation_error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": str(exc),
            "code": "INVALID_INPUT",
        },
    )


def setup_error_handlers(app: FastAPI) -> None:
    """Registers exception handlers on the FastAPI application."""
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(ValueError, value_error_handler)
