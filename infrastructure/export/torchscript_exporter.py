"""TorchScript export wrapper for PyTorch diagnostic backbones.

Enables compiling deep PyTorch classifiers into serialized TorchScript trace files.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TorchScriptModelExporter:
    """Utility class to compile PyTorch models to standard TorchScript format."""

    @staticmethod
    def export(
        model: nn.Module,
        export_path: str,
        input_size: tuple[int, ...] = (1, 3, 224, 224),
    ) -> None:
        """Export the PyTorch model to TorchScript format.

        Args:
            model: PyTorch nn.Module instance to export.
            export_path: Destination path for the exported TorchScript model (.pt).
            input_size: Tuple representing standard input dimensions.
        """
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size, device=device)

        # PyTorch to TorchScript trace compilation
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(export_path)
