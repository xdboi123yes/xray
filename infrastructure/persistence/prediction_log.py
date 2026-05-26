"""Lightweight SQLite database persistence layer for diagnostic prediction history.

Provides robust connection management, automatic migrations, record inserts,
queries with limit/offset pagination, and record deletes.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from config.settings import get_settings


class HistoryDatabaseManager:
    """Manages SQLite operations for saving and retrieving diagnostic history records."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initializes the database manager and ensures tables exist.

        Args:
            db_path: Explicit path to the SQLite file. If None, resolves from settings.
        """
        if db_path is None:
            settings = get_settings()
            # Ensure the results directory from settings exists
            os.makedirs(settings.paths.results, exist_ok=True)
            db_path = os.path.join(settings.paths.results, "history.db")

        self.db_path = db_path
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes a connection to the SQLite database.

        Returns:
            A sqlite3 connection object.
        """
        conn = sqlite3.connect(self.db_path)
        # Enable returning rows as dictionaries
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Executes table initialization if it does not already exist."""
        query = """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            tier_used INTEGER NOT NULL,
            mc_variance REAL,
            flagged_for_review INTEGER NOT NULL,
            timestamp TEXT NOT NULL
        );
        """
        with self._get_connection() as conn:
            conn.execute(query)
            conn.commit()

    def save_prediction(
        self,
        request_id: str,
        prediction: str,
        confidence: float,
        tier_used: int,
        mc_variance: float | None,
        flagged_for_review: bool,
        timestamp: str,
    ) -> None:
        """Saves a diagnostic record to the history database.

        Args:
            request_id: Unique UUID of the prediction request.
            prediction: Outcome ('Pneumothorax' or 'No Finding').
            confidence: Probability confidence score (0.0 to 1.0).
            tier_used: Classification tier (1 or 2).
            mc_variance: Uncertainty MC variance value.
            flagged_for_review: True if flagged due to high uncertainty.
            timestamp: ISO UTC timestamp.
        """
        query = """
        INSERT OR REPLACE INTO prediction_history (
            request_id, prediction, confidence, tier_used, mc_variance, flagged_for_review, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        flag_val = 1 if flagged_for_review else 0
        with self._get_connection() as conn:
            conn.execute(
                query,
                (
                    request_id,
                    prediction,
                    confidence,
                    tier_used,
                    mc_variance,
                    flag_val,
                    timestamp,
                ),
            )
            conn.commit()

    def get_history(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Retrieves prediction history records with limit and offset pagination.

        Args:
            limit: Maximum number of records to retrieve.
            offset: Number of initial records to skip.

        Returns:
            A list of dictionary records.
        """
        query = """
        SELECT id, request_id, prediction, confidence, tier_used, mc_variance, flagged_for_review, timestamp
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ? OFFSET ?;
        """
        with self._get_connection() as conn:
            cursor = conn.execute(query, (limit, offset))
            rows = cursor.fetchall()
            records = []
            for row in rows:
                records.append(
                    {
                        "id": row["id"],
                        "request_id": row["request_id"],
                        "prediction": row["prediction"],
                        "confidence": row["confidence"],
                        "tier_used": row["tier_used"],
                        "mc_variance": row["mc_variance"],
                        "flagged_for_review": bool(row["flagged_for_review"]),
                        "timestamp": row["timestamp"],
                    }
                )
            return records

    def delete_record(self, request_id: str) -> bool:
        """Deletes a specific diagnostic record by its request_id.

        Args:
            request_id: UUID of the prediction request.

        Returns:
            True if a record was deleted, False otherwise.
        """
        query = "DELETE FROM prediction_history WHERE request_id = ?;"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (request_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_history(self) -> None:
        """Truncates all history records from the database."""
        query = "DELETE FROM prediction_history;"
        with self._get_connection() as conn:
            conn.execute(query)
            conn.commit()
