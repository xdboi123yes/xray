"""Tests for the CheXpert zero-shot cohort builders (A14 validation cohort, A14b expanded cohort)."""

from __future__ import annotations

import pandas as pd

from scripts.prepare_chexpert import build_valid_cohort, sample_large_cohort


def _row(path: str, view: str = "Frontal", pneumo: float | None = 0.0, nf: float | None = 0.0) -> dict:
    return {"Path": path, "Frontal/Lateral": view, "Pneumothorax": pneumo, "No Finding": nf}


def test_valid_cohort_keeps_frontals_with_patient_qualified_image_ids(tmp_path):
    valid_dir = tmp_path / "valid"
    for sub in ("patient64541/study1/view1_frontal.jpg", "patient64542/study1/view1_frontal.jpg"):
        (valid_dir / sub).parent.mkdir(parents=True, exist_ok=True)
        (valid_dir / sub).write_bytes(b"jpg")
    df = pd.DataFrame(
        [
            _row("CheXpert-v1.0/valid/patient64541/study1/view1_frontal.jpg", pneumo=1.0),
            _row("CheXpert-v1.0/valid/patient64541/study1/view2_lateral.jpg", view="Lateral"),
            _row("CheXpert-v1.0/valid/patient64542/study1/view1_frontal.jpg", nf=1.0),
            _row("CheXpert-v1.0/valid/patient64543/study1/view1_frontal.jpg"),  # image not on disk
        ]
    )
    out = build_valid_cohort(df, valid_dir)
    assert out["Image Index"].tolist() == [
        "patient64541/study1/view1_frontal.jpg",
        "patient64542/study1/view1_frontal.jpg",
    ]
    assert out["Patient ID"].tolist() == ["patient64541", "patient64542"]
    assert all(str(valid_dir) in p for p in out["Path"])


def test_large_cohort_excludes_uncertain_labels_laterals_and_cross_class_patients():
    rows = [
        _row("train/patient00001/study1/view1_frontal.jpg", pneumo=1.0),
        _row("train/patient00001/study2/view1_frontal.jpg", pneumo=0.0),  # same patient as a positive
        _row("train/patient00002/study1/view1_frontal.jpg", pneumo=-1.0),  # uncertain
        _row("train/patient00003/study1/view1_frontal.jpg", pneumo=None, nf=1.0),  # clean negative
        _row("train/patient00004/study1/view1_lateral.jpg", view="Lateral", pneumo=1.0),
        _row("train/patient00005/study1/view1_frontal.jpg", pneumo=None, nf=0.0),  # unlabeled, other finding
    ]
    out = sample_large_cohort(pd.DataFrame(rows), n_pos=10, n_neg=10, seed=42)
    assert sorted(out["Patient ID"]) == ["patient00001", "patient00003"]
    assert out.set_index("Patient ID").loc["patient00001", "Pneumothorax"] == 1.0


def test_large_cohort_takes_one_image_per_patient_and_is_deterministic():
    rows = [_row(f"train/patient{p:05d}/study{s}/view1_frontal.jpg", pneumo=1.0 if p < 20 else 0.0)
            for p in range(100) for s in (1, 2)]
    df = pd.DataFrame(rows)
    a = sample_large_cohort(df, n_pos=5, n_neg=15, seed=42)
    b = sample_large_cohort(df, n_pos=5, n_neg=15, seed=42)
    assert a["Path"].tolist() == b["Path"].tolist()
    assert a["Patient ID"].is_unique
    assert int((a["Pneumothorax"] == 1.0).sum()) == 5
    assert len(a) == 20
