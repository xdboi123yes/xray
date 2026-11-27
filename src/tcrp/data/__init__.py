"""Data splitting, dataset loading, augmentation, and FID-gated synthetic generation."""

from __future__ import annotations

from tcrp.data.augmentation import ClassicalAugmentation, MixupCutmixAugmentation
from tcrp.data.datasets import CheXpertDataset, ImagePreprocessor, NIHChestXrayDataset
from tcrp.data.splits import (
    SplitSummary,
    build_patient_disjoint_splits,
    verify_patient_disjointness,
)
from tcrp.data.synthetic import (
    BatchFIDRecord,
    DiffusersGeneratorAdapter,
    FIDEvaluatorAdapter,
    GeneratorAdapter,
    PyTorchFIDAdapter,
    SyntheticBatchPipeline,
)

__all__ = [
    "BatchFIDRecord",
    "CheXpertDataset",
    "ClassicalAugmentation",
    "DiffusersGeneratorAdapter",
    "FIDEvaluatorAdapter",
    "GeneratorAdapter",
    "ImagePreprocessor",
    "MixupCutmixAugmentation",
    "NIHChestXrayDataset",
    "PyTorchFIDAdapter",
    "SplitSummary",
    "SyntheticBatchPipeline",
    "build_patient_disjoint_splits",
    "verify_patient_disjointness",
]
