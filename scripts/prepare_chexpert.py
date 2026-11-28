"""Build the real CheXpert zero-shot cohorts for A14 and A14b.

* A14 (``data/processed/chexpert_test.csv``): every frontal radiograph of the official
  CheXpert validation split (202 frontals / 200 patients).
* A14b (``data/processed/chexpert_large_test.csv``): an expanded cohort sampled from CheXpert
  ``train.csv``: frontal views only, definitive labels only (uncertain ``-1`` excluded), one image
  per patient, and no patient contributing to both classes.

CheXpert ``valid`` and ``train`` patients are disjoint by construction of the public release, and
none of them overlap NIH ChestX-ray14, so both cohorts are fully out-of-domain and patient-disjoint.
Images are fetched once from Kaggle (``ashery/chexpert``) and cached to ``--drive-cache-dir``.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

_PATIENT_RE = r"(patient\d+)"


def _relative_to(path: str, marker: str) -> str:
    """Return the part of a CheXpert ``Path`` after ``<marker>/`` (e.g. ``patientX/studyY/view.jpg``)."""
    p = str(path)
    return p.split(f"{marker}/", 1)[1] if f"{marker}/" in p else os.path.basename(p)


def _frontal(df: pd.DataFrame) -> pd.DataFrame:
    if "Frontal/Lateral" in df.columns:
        return df[df["Frontal/Lateral"] == "Frontal"].copy()
    return df.copy()


def build_valid_cohort(valid_df: pd.DataFrame, valid_dir: Path) -> pd.DataFrame:
    """A14 cohort: frontal validation images present on disk, with patient-qualified ids."""
    df = _frontal(valid_df)
    df["Image Index"] = df["Path"].map(lambda p: _relative_to(p, "valid"))
    df["Path"] = df["Image Index"].map(lambda sub: str(Path(valid_dir) / sub))
    df = df[df["Path"].map(os.path.exists)].copy()
    df["Patient ID"] = df["Image Index"].str.extract(_PATIENT_RE)[0]
    return df.reset_index(drop=True)


def sample_large_cohort(train_df: pd.DataFrame, n_pos: int = 800, n_neg: int = 3200, seed: int = 42) -> pd.DataFrame:
    """A14b cohort: one definitive-label frontal per patient; positives' patients never appear as negatives."""
    df = _frontal(train_df)
    df["Patient ID"] = df["Path"].astype(str).str.extract(_PATIENT_RE)[0]
    p_col = df["Pneumothorax"]
    nf_col = df["No Finding"] if "No Finding" in df.columns else pd.Series(0.0, index=df.index)
    pos = df[p_col == 1.0]
    neg = df[(p_col == 0.0) | (p_col.isna() & (nf_col == 1.0))]
    neg = neg[~neg["Patient ID"].isin(set(pos["Patient ID"]))]
    pos = pos.drop_duplicates("Patient ID")
    neg = neg.drop_duplicates("Patient ID")
    pos_s = pos.sample(n=min(n_pos, len(pos)), random_state=seed)
    neg_s = neg.sample(n=min(n_neg, len(neg)), random_state=seed)
    out = pd.concat([pos_s, neg_s], ignore_index=True)
    out["Image Index"] = out["Path"].map(lambda p: _relative_to(p, "train"))
    return out


def _find(root: Path, name: str) -> Path | None:
    hits = sorted(glob.glob(str(root / "**" / name), recursive=True))
    return Path(hits[0]) if hits else None


def _find_split_dir(root: Path, split: str) -> Path | None:
    for d in sorted(glob.glob(str(root / "**" / split), recursive=True)):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "patient*")):
            return Path(d)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("/content/chexpert_data"))
    parser.add_argument("--drive-cache-dir", type=Path, default=None,
                        help="Drive folder caching valid/, large_sample/ and both cohort CSVs.")
    parser.add_argument("--kaggle-dataset", default="ashery/chexpert")
    parser.add_argument("--valid-out", type=Path, default=Path("data/processed/chexpert_test.csv"))
    parser.add_argument("--large-out", type=Path, default=Path("data/processed/chexpert_large_test.csv"))
    parser.add_argument("--n-pos", type=int, default=800)
    parser.add_argument("--n-neg", type=int, default=3200)
    args = parser.parse_args()

    root: Path = args.data_root
    root.mkdir(parents=True, exist_ok=True)
    cache: Path | None = args.drive_cache_dir
    large_dir = root / "large_sample"

    if cache is not None and cache.is_dir() and not _find_split_dir(root, "valid"):
        print(f"[chexpert] restoring cache {cache} -> {root}")
        subprocess.check_call(["cp", "-r", f"{cache}/.", str(root)])

    valid_dir, valid_csv = _find_split_dir(root, "valid"), _find(root, "valid.csv")
    have_large = (root / "large_sample.csv").is_file() and large_dir.is_dir()
    if not (valid_dir and valid_csv and have_large):
        print(f"[chexpert] downloading {args.kaggle_dataset} from Kaggle")
        subprocess.check_call(["kaggle", "datasets", "download", "-d", args.kaggle_dataset,
                               "-p", str(root), "--unzip", "-q"])
        valid_dir, valid_csv = _find_split_dir(root, "valid"), _find(root, "valid.csv")
    if not (valid_dir and valid_csv):
        raise SystemExit(f"[chexpert] CheXpert valid split not found under {root}")

    valid = build_valid_cohort(pd.read_csv(valid_csv), valid_dir)
    if not have_large:
        train_dir, train_csv = _find_split_dir(root, "train"), _find(root, "train.csv")
        if not (train_dir and train_csv):
            raise SystemExit(f"[chexpert] CheXpert train split not found under {root}")
        large = sample_large_cohort(pd.read_csv(train_csv), n_pos=args.n_pos, n_neg=args.n_neg)
        kept = []
        for _, row in large.iterrows():
            src = train_dir / row["Image Index"]
            if src.is_file():
                dst = large_dir / row["Image Index"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                kept.append(row)
        pd.DataFrame(kept).to_csv(root / "large_sample.csv", index=False)
    large = pd.read_csv(root / "large_sample.csv")
    large["Path"] = large["Image Index"].map(lambda sub: str(large_dir / sub))
    large = large[large["Path"].map(os.path.exists)].reset_index(drop=True)

    for df, out in ((valid, args.valid_out), (large, args.large_out)):
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        n_pos = int((df["Pneumothorax"] == 1.0).sum())
        print(f"[chexpert] {out}: {len(df)} images, {df['Patient ID'].nunique()} patients, {n_pos} positive")

    if cache is not None and not (cache / "large_sample.csv").is_file():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["cp", "-r", str(valid_dir), str(cache / "valid")])
        shutil.copy2(valid_csv, cache / "valid.csv")
        subprocess.check_call(["cp", "-r", str(large_dir), str(cache / "large_sample")])
        shutil.copy2(root / "large_sample.csv", cache / "large_sample.csv")
        print(f"[chexpert] cached valid/ + large_sample/ to {cache}")


if __name__ == "__main__":
    main()
