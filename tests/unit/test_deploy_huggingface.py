"""Unit tests for HuggingFace deployment helper script."""

from __future__ import annotations

import os

from scripts.deploy_huggingface import deploy_to_spaces


def test_deploy_huggingface_dry_run() -> None:
    """Verify that deploy_to_spaces runs successfully in dry-run mode."""
    dummy_weights = "tests/dummy_weights.pth"
    with open(dummy_weights, "w") as f:
        f.write("mock weights content")

    try:
        success = deploy_to_spaces(
            token="dummy_token",
            space_id="dummy_user/dummy_space",
            weights_path=dummy_weights,
            dry_run=True,
        )
        assert success is True
    finally:
        if os.path.exists(dummy_weights):
            os.remove(dummy_weights)


def test_deploy_huggingface_missing_file() -> None:
    """Verify that deploy_to_spaces returns False if the weight file is missing."""
    success = deploy_to_spaces(
        token="dummy_token",
        space_id="dummy_user/dummy_space",
        weights_path="non_existent_file.pth",
        dry_run=True,
    )
    assert success is False
