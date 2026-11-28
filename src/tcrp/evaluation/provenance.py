"""SHA-256 provenance manifest builder and scientific integrity gate verifier.

Records git commit, Python/PyTorch/CUDA versions, patient-disjoint split
manifest checks, disjoint calibration protocol metadata (Validation Split
temperature scaling before Calibration Split conformal quantile estimation),
and SHA-256 digests of every critical checkpoint and evaluation artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

REQUIRED_SPLIT_NAMES = ("train", "val", "calibration", "test")
REQUIRED_OVERLAP_PAIRS = tuple(
    f"{a}__{b}" for a, b in combinations(sorted(REQUIRED_SPLIT_NAMES), 2)
)
_SHA256_HEX_LEN = 64


@dataclass(frozen=True)
class ProvenanceVerificationResult:
    """Outcome of verifying a persisted Track B provenance manifest."""

    valid: bool
    protocol: str
    verified_artifacts: int
    errors: tuple[str, ...] = ()


def compute_file_sha256(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Compute the lowercase hexadecimal SHA-256 digest of ``path``."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Artifact not found for SHA-256 hashing: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_git_commit() -> str:
    """Return current git HEAD SHA or ``'unknown'`` outside a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = res.stdout.strip()
        return commit if res.returncode == 0 and commit else "unknown"
    except OSError:
        return "unknown"


def _normalize_artifact_key(path: Path, base_dir: Path | None) -> str:
    """Normalize an artifact path to a POSIX key relative to ``base_dir`` when inside it."""
    resolved = path.resolve()
    if base_dir is not None:
        base_resolved = base_dir.resolve()
        try:
            return resolved.relative_to(base_resolved).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _resolve_artifact_path(key: str, base_dir: Path | None) -> Path:
    """Resolve a manifest artifact key back to a filesystem path."""
    candidate = Path(key)
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return base_dir / candidate


def _validate_split_manifest_payload(payload: Any) -> list[str]:
    """Validate zero pairwise patient overlap across all 6 split pairs and four-way split presence."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Split manifest JSON root must be an object."]

    splits = payload.get("splits")
    if not isinstance(splits, dict):
        errors.append("Split manifest missing 'splits' dictionary.")
    else:
        missing_splits = [name for name in REQUIRED_SPLIT_NAMES if name not in splits]
        if missing_splits:
            errors.append(f"Split manifest missing required splits: {missing_splits}.")

    overlaps = payload.get("pairwise_patient_overlaps")
    if not isinstance(overlaps, dict) or not overlaps:
        errors.append(
            "Split manifest missing non-empty 'pairwise_patient_overlaps' mapping."
        )
    else:
        missing_pairs = [pair for pair in REQUIRED_OVERLAP_PAIRS if pair not in overlaps]
        if missing_pairs:
            errors.append(
                f"Split manifest 'pairwise_patient_overlaps' missing pairs: {missing_pairs}."
            )
        non_zero: dict[str, Any] = {}
        invalid_entries: dict[str, Any] = {}
        for k, v in overlaps.items():
            if isinstance(v, bool) or not isinstance(v, int):
                invalid_entries[str(k)] = v
            elif v != 0:
                non_zero[str(k)] = v
        if invalid_entries:
            errors.append(
                f"Split manifest has non-integer patient overlap entries: {invalid_entries}."
            )
        if non_zero:
            errors.append(f"Split manifest reports non-zero patient overlap: {non_zero}.")
    return errors


def verify_split_csv_hashes(split_manifest_path: str | Path, processed_dir: str | Path) -> list[str]:
    """Return errors for split CSVs that are missing or differ from the manifest's SHA-256 digests."""
    payload = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
    errors: list[str] = []
    for name in REQUIRED_SPLIT_NAMES:
        expected = payload.get("splits", {}).get(name, {}).get("sha256")
        csv = Path(processed_dir) / f"{name}.csv"
        if not expected:
            errors.append(f"Split manifest has no sha256 for split {name!r}.")
        elif not csv.is_file():
            errors.append(f"Split CSV missing for {name!r}: {csv}")
        elif compute_file_sha256(csv) != expected:
            errors.append(f"Split CSV SHA-256 mismatch for {name!r}: {csv} differs from the split manifest.")
    return errors


def _validate_calibration_protocol(metadata: dict[str, Any]) -> list[str]:
    """Validate strict Validation-Split temperature before Calibration-Split conformal order."""
    errors: list[str] = []
    temp_split = str(metadata.get("temperature_split", "")).strip().lower()
    conf_split = str(metadata.get("conformal_split", "")).strip().lower()
    if temp_split != "val" or conf_split != "calibration" or temp_split == conf_split:
        errors.append(
            "Calibration protocol requires strictly disjoint temperature_split='val' "
            f"and conformal_split='calibration' (got temperature_split={temp_split!r}, "
            f"conformal_split={conf_split!r})."
        )
    if metadata.get("temperature_before_conformal") is not True:
        errors.append(
            "Calibration protocol must assert temperature_before_conformal=True."
        )
    temp_val = metadata.get("temperature")
    if (
        isinstance(temp_val, bool)
        or not isinstance(temp_val, (int, float))
        or not math.isfinite(float(temp_val))
        or float(temp_val) <= 0.0
    ):
        errors.append(
            f"Calibration protocol requires finite positive temperature, got {temp_val!r}."
        )
    return errors


def build_provenance_manifest(
    output_path: str | Path,
    artifact_paths: Iterable[str | Path],
    protocol: str = "patient-disjoint-v1",
    split_manifest_path: str | Path | None = None,
    calibration_metadata: dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build and write a SHA-256 provenance manifest for verified pipeline artifacts.

    Raises:
        FileNotFoundError: If any listed artifact or split manifest is missing.
        ValueError: If the split manifest reports patient overlap or calibration
            protocol metadata violates disjoint Validation -> Calibration order.
    """
    base = Path(base_dir).resolve() if base_dir is not None else None
    all_paths: list[Path] = []
    seen_keys: set[str] = set()

    def _add_path(raw: str | Path) -> str:
        p = Path(raw)
        if not p.is_absolute() and base is not None:
            p = base / p
        if not p.is_file():
            raise FileNotFoundError(f"Missing critical provenance artifact: {p}")
        key = _normalize_artifact_key(p, base)
        if key not in seen_keys:
            seen_keys.add(key)
            all_paths.append(p)
        return key

    split_manifest_key: str | None = None
    if split_manifest_path is not None:
        split_manifest_key = _add_path(split_manifest_path)
        split_resolved = _resolve_artifact_path(split_manifest_key, base)
        split_payload = json.loads(split_resolved.read_text(encoding="utf-8"))
        split_errors = _validate_split_manifest_payload(split_payload)
        if split_errors:
            raise ValueError("; ".join(split_errors))

    if calibration_metadata is not None:
        cal_errors = _validate_calibration_protocol(calibration_metadata)
        if cal_errors:
            raise ValueError("; ".join(cal_errors))

    for art in artifact_paths:
        _add_path(art)

    if not all_paths:
        raise ValueError("Cannot build provenance manifest with zero artifacts.")

    hashes: dict[str, str] = {}
    for p in sorted(all_paths, key=lambda item: _normalize_artifact_key(item, base)):
        key = _normalize_artifact_key(p, base)
        hashes[key] = compute_file_sha256(p)

    manifest: dict[str, Any] = {
        "protocol": protocol,
        "git_commit": git_commit if git_commit is not None else _detect_git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "artifacts_sha256": hashes,
    }
    if split_manifest_key is not None:
        manifest["split_manifest_path"] = split_manifest_key
    if calibration_metadata is not None:
        manifest["calibration_protocol"] = dict(calibration_metadata)

    out = Path(output_path)
    os.makedirs(out.parent, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_provenance_manifest(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
    expected_protocol: str = "patient-disjoint-v1",
    required_artifacts: Iterable[str] = (),
    strict: bool = True,
) -> ProvenanceVerificationResult:
    """Verify a provenance manifest's protocol, SHA-256 digests, split disjointness, and calibration order."""
    path = Path(manifest_path)
    base = Path(base_dir).resolve() if base_dir is not None else None
    errors: list[str] = []

    if not path.is_file():
        errors.append(f"Provenance manifest not found: {path}")
        if strict:
            raise ValueError("; ".join(errors))
        return ProvenanceVerificationResult(
            valid=False, protocol="", verified_artifacts=0, errors=tuple(errors)
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Could not parse provenance manifest JSON ({path}): {exc}")
        if strict:
            raise ValueError("; ".join(errors)) from exc
        return ProvenanceVerificationResult(
            valid=False, protocol="", verified_artifacts=0, errors=tuple(errors)
        )

    if not isinstance(payload, dict):
        errors.append("Provenance manifest JSON root must be an object.")
        if strict:
            raise ValueError("; ".join(errors))
        return ProvenanceVerificationResult(
            valid=False, protocol="", verified_artifacts=0, errors=tuple(errors)
        )

    protocol = str(payload.get("protocol", ""))
    if protocol != expected_protocol:
        errors.append(
            f"Protocol mismatch: expected {expected_protocol!r}, got {protocol!r}."
        )

    artifacts = payload.get("artifacts_sha256")
    verified_count = 0
    if not isinstance(artifacts, dict) or not artifacts:
        errors.append("Manifest has empty or missing 'artifacts_sha256' mapping.")
    else:
        for req in required_artifacts:
            req_key = Path(req).as_posix()
            if req_key not in artifacts:
                errors.append(f"Missing required artifact in provenance manifest: {req_key}")

        for key, expected_hash in artifacts.items():
            if not isinstance(expected_hash, str) or len(expected_hash) != _SHA256_HEX_LEN:
                errors.append(f"Invalid SHA-256 digest format for {key!r}: {expected_hash!r}")
                continue
            target = _resolve_artifact_path(key, base)
            if not target.is_file():
                errors.append(f"Artifact listed in manifest is missing on disk: {key}")
                continue
            actual_hash = compute_file_sha256(target)
            if actual_hash != expected_hash:
                errors.append(
                    f"SHA-256 mismatch for {key}: expected {expected_hash[:12]}..., "
                    f"got {actual_hash[:12]}..."
                )
            else:
                verified_count += 1

    if "split_manifest_path" in payload:
        split_manifest_key = payload["split_manifest_path"]
        if not isinstance(split_manifest_key, str) or not split_manifest_key.strip():
            errors.append(
                f"Malformed 'split_manifest_path' in manifest: {split_manifest_key!r}."
            )
        else:
            split_target = _resolve_artifact_path(split_manifest_key, base)
            if not split_target.is_file():
                errors.append(f"Referenced split manifest missing: {split_manifest_key}")
            else:
                try:
                    split_payload = json.loads(split_target.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(
                        f"Could not parse split manifest JSON ({split_manifest_key}): {exc}"
                    )
                else:
                    errors.extend(_validate_split_manifest_payload(split_payload))

    if "calibration_protocol" in payload:
        cal_meta = payload["calibration_protocol"]
        if not isinstance(cal_meta, dict):
            errors.append(
                f"Malformed 'calibration_protocol' in manifest (expected dict): {cal_meta!r}."
            )
        else:
            errors.extend(_validate_calibration_protocol(cal_meta))

    result = ProvenanceVerificationResult(
        valid=not errors,
        protocol=protocol,
        verified_artifacts=verified_count,
        errors=tuple(errors),
    )
    if strict and not result.valid:
        raise ValueError("Provenance verification failed: " + "; ".join(result.errors))
    return result
