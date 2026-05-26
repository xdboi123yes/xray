#!/usr/bin/env python3
"""Acquire pretrained weights for the Tier-2 Ark+ Swin-Base backbone.

Strategy:
  1. Try a series of candidate URLs for the official Ark+ checkpoint.
     The Ark project (JLiangLab/Ark) historically distributes the
     ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar checkpoint via the
     GitHub releases page or a HuggingFace mirror. Both can move, so we
     probe each candidate in order until one returns HTTP 200.
  2. If none succeed, fall back to a clean Swin-Base ImageNet checkpoint
     provided by timm. The Ark+ class will load this via the standard
     `timm.create_model(..., pretrained=True)` code path at runtime, so
     no additional download is strictly required - but we still attempt
     this download so the cache is warm and the build is reproducible
     without internet access during training.

The fallback path is documented in PLAN.md section 5.1 (Tertiary: Swin-Base
ImageNet via timm). The thesis can honestly cite either "Ark+ official
checkpoint" or "Swin-Base ImageNet equivalent" depending on which path
succeeded.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import structlog

log = structlog.get_logger(__name__)


# Candidate URLs in priority order. The first that returns HTTP 200 wins.
ARK_PRIMARY_FILENAME = "ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar"
ARK_URL_CANDIDATES: list[str] = [
    # 1. JLiangLab releases (URL pattern Ark has used historically, may 404)
    f"https://github.com/jlianglab/Ark/releases/download/v1.0/{ARK_PRIMARY_FILENAME}",
    f"https://github.com/jlianglab/Ark/releases/download/v0.1/{ARK_PRIMARY_FILENAME}",
    # 2. HuggingFace mirrors that occasionally re-host Ark checkpoints
    f"https://huggingface.co/jlianglab/Ark/resolve/main/{ARK_PRIMARY_FILENAME}",
    f"https://huggingface.co/JLiangLab/Ark/resolve/main/{ARK_PRIMARY_FILENAME}",
]

OUTPUT_DIR = Path("outputs/models")
ARK_OUTPUT_PATH = OUTPUT_DIR / "ark_plus_swin_base.pth"
SWIN_FALLBACK_PATH = OUTPUT_DIR / "swin_base_imagenet_fallback.pth"


def _probe(url: str, timeout: float = 10.0) -> int:
    """Return the HTTP status code for a HEAD-like check, or 0 on network failure."""
    req = Request(url, method="HEAD")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except URLError as exc:
        log.warning("ark_url_probe_failed", url=url, error=str(exc))
        return 0
    except Exception as exc:
        log.warning("ark_url_probe_exception", url=url, error=str(exc))
        return 0


def _download(url: str, dest: Path, timeout: float = 600.0) -> bool:
    """Stream a URL to disk in 1 MB chunks. Return True on success."""
    log.info("ark_download_attempt", url=url, dest=str(dest))
    try:
        with urlopen(url, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            chunk_size = 1024 * 1024
            written = 0
            sha = hashlib.sha256()
            with open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    sha.update(chunk)
                    written += len(chunk)
        log.info(
            "ark_download_complete",
            url=url,
            dest=str(dest),
            bytes_written=written,
            expected_bytes=total or None,
            sha256_prefix=sha.hexdigest()[:12],
        )
        return True
    except Exception as exc:
        log.error("ark_download_failed", url=url, error=str(exc))
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return False


def try_ark_official() -> bool:
    """Try every Ark+ candidate URL in turn. Return True on first success."""
    if ARK_OUTPUT_PATH.exists():
        log.info("ark_already_downloaded", path=str(ARK_OUTPUT_PATH))
        return True

    for url in ARK_URL_CANDIDATES:
        status = _probe(url)
        if status == 200 and _download(url, ARK_OUTPUT_PATH):
            return True
        log.info("ark_candidate_unavailable", url=url, status=status)
    return False


def try_swin_base_fallback() -> bool:
    """Materialize a Swin-Base ImageNet checkpoint via timm so training is reproducible offline."""
    if SWIN_FALLBACK_PATH.exists():
        log.info("swin_fallback_present", path=str(SWIN_FALLBACK_PATH))
        return True

    try:
        import timm  # type: ignore[import-not-found]
        import torch
    except ImportError as exc:
        log.error("swin_fallback_import_failed", error=str(exc))
        return False

    try:
        log.info("swin_fallback_create")
        model = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=True, num_classes=0
        )
        torch.save(model.state_dict(), SWIN_FALLBACK_PATH)
        log.info("swin_fallback_saved", path=str(SWIN_FALLBACK_PATH))
        return True
    except Exception as exc:
        log.error("swin_fallback_failed", error=str(exc))
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-fallback",
        action="store_true",
        help="Do not attempt the Swin-Base ImageNet fallback if Ark+ is unreachable.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if try_ark_official():
        log.info("ark_acquisition_success", source="official_ark")
        return 0

    log.warning("ark_official_unavailable", message="All Ark+ candidate URLs failed.")
    if args.skip_fallback:
        log.error("ark_acquisition_failed_no_fallback")
        return 2

    if try_swin_base_fallback():
        log.info(
            "ark_acquisition_success",
            source="swin_base_imagenet_fallback",
            note="Thesis must cite this as a Swin-Base ImageNet equivalent, not as the Ark+ paper checkpoint.",
        )
        return 0

    log.error("ark_acquisition_failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
