"""Weighted loss functions for Chest X-Ray class imbalance management.

Calculates class weights dynamically based on sample distributions in target CSV.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import structlog
import torch
import torch.nn as nn

log = structlog.get_logger(__name__)


def get_class_weighted_loss(
    csv_path: str,
    target_classes: Sequence[str] = ("No Finding", "Pneumothorax"),
) -> nn.CrossEntropyLoss:
    """Computes class weights inversely proportional to frequencies in clinical dataset.

    Normalizes weights to ensure they sum to the number of target classes.

    Args:
        csv_path: Path to the training metadata CSV containing Finding Labels.
        target_classes: Sequence of finding label strings to balance.

    Returns:
        An instance of nn.CrossEntropyLoss initialized with class weights.
    """
    df = pd.read_csv(csv_path)

    # Filter only target classes
    df_filtered = df[df["Finding Labels"].isin(target_classes)]

    # Calculate frequencies
    class_counts = df_filtered["Finding Labels"].value_counts()

    # NIH dataset: 0 represents 'No Finding', 1 represents 'Pneumothorax'
    count_0 = int(class_counts.get("No Finding", 1))
    count_1 = int(class_counts.get("Pneumothorax", 1))

    total = count_0 + count_1

    # Inverse frequency weights
    weight_0 = total / count_0
    weight_1 = total / count_1

    # Normalize weights so they sum to the number of classes (2)
    sum_weights = weight_0 + weight_1
    weight_0 = (weight_0 / sum_weights) * 2.0
    weight_1 = (weight_1 / sum_weights) * 2.0

    weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)
    log.info(f"Computed Class Weights: [No Finding: {weight_0:.4f}, Pneumothorax: {weight_1:.4f}]")

    return nn.CrossEntropyLoss(weight=weights)
