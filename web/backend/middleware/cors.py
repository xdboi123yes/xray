"""CORS configuration wrapper to decouple app.py setup."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import get_settings

def setup_cors(app: FastAPI) -> None:
    """Configures secure CORS rules matching the application environment settings."""
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
