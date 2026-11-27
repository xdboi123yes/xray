"""ONNX model export utility for Tier 1 and Tier 2 classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn


class ONNXModelExporter:
    """Exports PyTorch classifiers to ONNX with dynamic batch axes."""

    @staticmethod
    def export(
        model: nn.Module,
        export_path: str,
        input_size: tuple[int, ...] = (1, 3, 224, 224),
        opset_version: int = 18,
    ) -> None:
        """Trace and export ``model`` to ``export_path`` in ONNX format."""
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size, device=device)

        torch.onnx.export(
            model,
            (dummy_input,),
            export_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
