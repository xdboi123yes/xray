"""Logging utilities for the core layered system."""

from __future__ import annotations

import csv
import os
from typing import Any


def log_inference(log_path: str, record: dict[str, Any]) -> None:
    """Appends one row to a CSV file at log_path with headers if missing.

    Args:
        log_path: CSV target file.
        record: Key-value log parameters dict.
    """
    file_exists = os.path.isfile(log_path)
    # Ensure directories exist
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
