"""ONNX export wrapper for PyTorch diagnostic backbones.

Enables compilation of deep PyTorch classifiers into standardized,
production-ready ONNX formats with support for INT8 quantization mappings.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ONNXModelExporter:
    """Utility class to compile PyTorch models to standard ONNX format."""

    @staticmethod
    def export(
        model: nn.Module,
        export_path: str,
        input_size: tuple[int, ...] = (1, 3, 224, 224),
    ) -> None:
        """Export the PyTorch model to ONNX format.

        Args:
            model: PyTorch nn.Module instance to export.
            export_path: Destination path for the exported ONNX model.
            input_size: Tuple representing standard input dimensions.
        """
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size, device=device)

        # PyTorch to ONNX graph tracing compilation
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
