#!/usr/bin/env python3
"""Hugging Face Spaces deployment automation utility script."""

from __future__ import annotations

import argparse
import os
import sys


def deploy_to_spaces(
    token: str,
    space_id: str,
    weights_path: str,
    dry_run: bool = False,
) -> bool:
    """Automates model weights deployment to Hugging Face Spaces.

    Args:
        token: Hugging Face API access token.
        space_id: Space repository target (e.g., 'username/spaces-name').
        weights_path: Absolute path to weights file to upload.
        dry_run: If True, simulates the upload process without network request.
    """
    print(f"Starting Hugging Face Spaces deployment to space: {space_id}")
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}", file=sys.stderr)
        return False

    if dry_run:
        print("[Dry-Run] Bypassing connection, weight transfer, and active server spin-up.")
        print(f"[Dry-Run] Successfully validated {weights_path} target readiness.")
        return True

    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except ImportError:
        print("Error: huggingface_hub package is not installed. Please run pip install huggingface-hub", file=sys.stderr)
        return False

    try:
        api = HfApi()
        print(f"Uploading {weights_path} to HF Space {space_id}...")
        api.upload_file(
            path_or_fileobj=weights_path,
            path_in_repo="model_weights.pth",
            repo_id=space_id,
            repo_type="space",
            token=token,
        )
        print("Upload completed successfully.")
        return True
    except Exception as e:
        print(f"Deployment failed: {e!s}", file=sys.stderr)
        return False

def main() -> None:
    """CLI entry point for HF Spaces deployment."""
    parser = argparse.ArgumentParser(description="Hugging Face Spaces deployment helper.")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="HF API Token.")
    parser.add_argument("--space-id", type=str, required=True, help="HF Space repo id.")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry-run execution.")

    args = parser.parse_args()

    if not args.token and not args.dry_run:
        parser.error("HF token must be provided via --token or HF_TOKEN env var unless running as dry-run.")

    success = deploy_to_spaces(
        token=args.token or "",
        space_id=args.space_id,
        weights_path=args.weights,
        dry_run=args.dry_run,
    )
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
