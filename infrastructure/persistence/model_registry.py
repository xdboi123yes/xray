"""Model registry interface for scanning and validating local weights files.

Provides detailed metadata about available trained check-points and local weights files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class LocalModelRegistry:
    """Manages local model weights discovery, metadata scanning, and validation."""

    def __init__(self, models_dir: str = "outputs/models") -> None:
        self.models_dir = Path(models_dir)

    def scan_available_checkpoints(self) -> list[dict[str, Any]]:
        """Scans the outputs/models directory for valid .pth weight files.

        Returns:
            A list of dictionary descriptors containing metadata about each model.
        """
        if not self.models_dir.exists():
            return []

        checkpoints = []
        for file_path in self.models_dir.glob("*.pth"):
            size_bytes = file_path.stat().st_size
            sha256_hash = self._compute_sha256(file_path)

            name_lower = file_path.stem.lower()
            tier = 1 if "tier1" in name_lower or "mobilenet" in name_lower else 2
            backbone = "mobilenet_v2" if tier == 1 else "efficientnet_b4"
            if "ark" in name_lower:
                backbone = "ark_plus"

            checkpoints.append(
                {
                    "name": file_path.name,
                    "stem": file_path.stem,
                    "tier": tier,
                    "backbone": backbone,
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "checksum": sha256_hash[:8],
                    "path": str(file_path),
                }
            )
        return checkpoints

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute the sha256 checksum of a file on disk."""
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return "unknown"
