"""Unit tests for TorchScriptModelExporter."""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from infrastructure.export.torchscript_exporter import TorchScriptModelExporter


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 224 * 224, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def test_torchscript_export_pipeline() -> None:
    """Verify that TorchScriptModelExporter exports model successfully and file is loadable."""
    model = SimpleModel()
    export_path = "tests/test_model_traced.pt"

    try:
        TorchScriptModelExporter.export(model, export_path)
        assert os.path.exists(export_path)

        # Load back to verify
        loaded = torch.jit.load(export_path)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = loaded(dummy_input)
        assert output.shape == (1, 2)

    finally:
        if os.path.exists(export_path):
            os.remove(export_path)
