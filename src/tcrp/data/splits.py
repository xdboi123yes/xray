"""Strict four-way Patient-Disjoint Partition builder for NIH ChestX-ray14.

Filters binary screening targets (isolated or comorbid Pneumothorax vs. strict
``No Finding`` capped at ``max_ratio:1``), groups radiographs by patient ID,
creates mutually exclusive train, Validation Split, Calibration Split, and test
partitions, and verifies zero pairwise patient overlap.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SplitCounts:
    """Image and unique-patient counts for one dataset split."""

    samples: int
    patients: int
    pneumothorax: int
    no_finding: int


@dataclass(frozen=True)
class SplitSummary:
    """Summary manifest for a generated Patient-Disjoint Partition."""

    output_dir: str
    image_dir: str
    seed: int
    max_ratio: int
    calibration_fraction_of_dev: float
    splits: dict[str, SplitCounts]
    pairwise_patient_overlaps: dict[str, int]


def extract_patient_id(image_index: str) -> str:
    """Extract the NIH patient identifier prefix from an image filename."""
    return str(image_index).split("_")[0]


def verify_patient_disjointness(split_frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Compute pairwise patient overlaps and raise if any patient spans two splits.

    Args:
        split_frames: Mapping of split name to DataFrame containing ``Image Index``
            (or ``Patient ID``).

    Returns:
        Mapping of ``"<split_a>__<split_b>"`` to overlap count (all zero).

    Raises:
        ValueError: If any patient ID appears in more than one split.
    """
    patient_sets: dict[str, set[str]] = {}
    for name, frame in split_frames.items():
        if "Patient ID" in frame.columns:
            ids = set(frame["Patient ID"].astype(str))
        else:
            ids = set(frame["Image Index"].astype(str).map(extract_patient_id))
        patient_sets[name] = ids

    names = sorted(patient_sets.keys())
    overlaps: dict[str, int] = {}
    violations: list[str] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            shared = patient_sets[name_a] & patient_sets[name_b]
            key = f"{name_a}__{name_b}"
            overlaps[key] = len(shared)
            if shared:
                violations.append(f"{key}={len(shared)}")

    if violations:
        raise ValueError(
            "Patient-Disjoint Partition violation detected across splits: " + ", ".join(violations)
        )
    return overlaps


def build_patient_disjoint_splits(
    csv_path: str,
    image_dir: str,
    output_dir: str,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    max_ratio: int = 5,
    calibration_fraction: float = 0.30,
    verify_images: bool = True,
) -> SplitSummary:
    """Build four-way Patient-Disjoint Partition CSVs and a verified manifest.

    Args:
        csv_path: Path to ``Data_Entry_2017.csv`` (must include ``Image Index`` and
            ``Finding Labels``).
        image_dir: Directory containing radiograph files. If it exists, rows missing
            on disk are dropped.
        output_dir: Destination directory for ``train.csv``, ``val.csv``,
            ``calibration.csv``, ``test.csv``, ``image_dir.txt``, and
            ``split_manifest.json``.
        train_split: Fraction of unique patients assigned to training.
        val_split: Fraction of unique patients assigned to development (further
            divided into Validation Split and Calibration Split).
        test_split: Fraction of unique patients assigned to held-out testing.
        seed: Random seed for reproducible balancing and stratified splitting.
        max_ratio: Cap on ``No Finding`` radiographs per ``Pneumothorax`` radiograph.
        calibration_fraction: Share of development patients reserved exclusively for
            the Calibration Split.

    Returns:
        A :class:`SplitSummary` recording verified image/patient counts and zero
        pairwise overlaps.

    Raises:
        ValueError: If no qualifying images exist or patient disjointness fails.
    """
    log.info("Loading metadata CSV", csv_path=csv_path)
    df = pd.read_csv(csv_path)

    mask_pneumo = df["Finding Labels"].str.contains("Pneumothorax", na=False)
    mask_normal = df["Finding Labels"] == "No Finding"
    df_filtered = df[mask_pneumo | mask_normal].copy()

    df_filtered["Label"] = df_filtered["Finding Labels"].apply(
        lambda x: 1 if "Pneumothorax" in str(x) else 0
    )

    n_pneumo = int((df_filtered["Label"] == 1).sum())
    max_normal = n_pneumo * max_ratio
    df_pneumo = df_filtered[df_filtered["Label"] == 1]
    df_normal = df_filtered[df_filtered["Label"] == 0]
    if len(df_normal) > max_normal:
        df_normal = df_normal.sample(n=max_normal, random_state=seed)
    df_filtered = pd.concat([df_pneumo, df_normal], ignore_index=True)

    if verify_images:
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"image_dir does not exist: '{image_dir}'. "
                "Pass verify_images=False to build splits without disk verification."
            )
        existing_files = set(os.listdir(image_dir))
        df_filtered = df_filtered[df_filtered["Image Index"].isin(existing_files)].copy()

    if len(df_filtered) == 0:
        raise ValueError("No qualifying radiograph rows found after filtering.")

    df_filtered["Patient ID"] = df_filtered["Image Index"].astype(str).map(extract_patient_id)
    patient_labels = (
        df_filtered.groupby("Patient ID", as_index=False)["Label"]
        .max()
        .rename(columns={"Label": "Patient Label"})
    )

    val_test_ratio = val_split + test_split
    train_patients, val_test_patients = train_test_split(
        patient_labels,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=patient_labels["Patient Label"],
    )

    test_ratio_of_val_test = test_split / val_test_ratio
    development_patients, test_patients = train_test_split(
        val_test_patients,
        test_size=test_ratio_of_val_test,
        random_state=seed,
        stratify=val_test_patients["Patient Label"],
    )

    val_patients, calibration_patients = train_test_split(
        development_patients,
        test_size=calibration_fraction,
        random_state=seed,
        stratify=development_patients["Patient Label"],
    )

    def _images_for(patients: pd.DataFrame) -> pd.DataFrame:
        return df_filtered[df_filtered["Patient ID"].isin(patients["Patient ID"])].copy()

    split_frames = {
        "train": _images_for(train_patients),
        "val": _images_for(val_patients),
        "calibration": _images_for(calibration_patients),
        "test": _images_for(test_patients),
    }

    overlaps = verify_patient_disjointness(split_frames)

    os.makedirs(output_dir, exist_ok=True)
    cols_to_save = ["Image Index", "Finding Labels", "Label", "Patient ID"]

    split_counts: dict[str, SplitCounts] = {}
    for name, split_df in split_frames.items():
        out_path = os.path.join(output_dir, f"{name}.csv")
        split_df[cols_to_save].to_csv(out_path, index=False)
        split_counts[name] = SplitCounts(
            samples=int(len(split_df)),
            patients=int(split_df["Patient ID"].nunique()),
            pneumothorax=int((split_df["Label"] == 1).sum()),
            no_finding=int((split_df["Label"] == 0).sum()),
        )

    image_dir_txt = os.path.join(output_dir, "image_dir.txt")
    with open(image_dir_txt, "w", encoding="utf-8") as f:
        f.write(image_dir)

    summary = SplitSummary(
        output_dir=output_dir,
        image_dir=image_dir,
        seed=seed,
        max_ratio=max_ratio,
        calibration_fraction_of_dev=calibration_fraction,
        splits=split_counts,
        pairwise_patient_overlaps=overlaps,
    )

    manifest_path = os.path.join(output_dir, "split_manifest.json")
    manifest_payload: dict[str, Any] = {
        **asdict(summary),
        "splits": {k: asdict(v) for k, v in split_counts.items()},
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    log.info("Patient-Disjoint Partition built", output_dir=output_dir, overlaps=overlaps)
    return summary
