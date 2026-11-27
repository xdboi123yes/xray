"""Tests for the Python 3.13 Colab dependency specification and notebook setup."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COLAB_REQS_PATH = REPO_ROOT / "requirements-colab.txt"
COLAB_NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "xray_colab_produce_all.ipynb"

FORBIDDEN_COLAB_PACKAGES = {
    "fastapi",
    "uvicorn",
    "python-multipart",
    "websockets",
    "slowapi",
    "gradio",
    "plotly",
    "nonconformist",
    "optuna",
    "memory-profiler",
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
    "ruff",
    "mypy",
    "pre-commit",
    "import-linter",
    "httpx",
    # Preinstalled by Colab with CUDA / C-extension ABI — must never be pinned directly
    "torch",
    "torchvision",
    "numpy",
}

REQUIRED_COLAB_PACKAGES = {
    "pydantic",
    "pydantic-settings",
    "pyyaml",
    "structlog",
    "timm",
    "albumentations",
    "opencv-python-headless",
    "pillow",
    "grad-cam",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "tqdm",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "pytorch-fid",
    "mlflow-skinny",
    "codecarbon",
    "onnx",
    "onnxruntime",
    "onnxscript",
    "kaggle",
    "gdown",
    "nbconvert",
}


def _parse_requirement_names(path: Path) -> set[str]:
    """Extract normalized PEP 503 package names from a requirements file."""
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)", line)
        if match:
            names.add(re.sub(r"[-_.]+", "-", match.group(1)).lower())
    return names


def test_colab_requirements_spec_is_lean_complete_and_py313_resolvable() -> None:
    """requirements-colab.txt contains only ML/research deps and resolves cleanly on Python 3.13."""
    assert COLAB_REQS_PATH.is_file(), "requirements-colab.txt must exist at repo root"

    declared = _parse_requirement_names(COLAB_REQS_PATH)
    leaked = sorted(declared & FORBIDDEN_COLAB_PACKAGES)
    assert not leaked, f"Forbidden legacy/dev/ABI-pinned packages in requirements-colab.txt: {leaked}"

    missing = sorted(REQUIRED_COLAB_PACKAGES - declared)
    assert not missing, f"Required ML/eval/synthetic/plotting packages missing: {missing}"

    uv_bin = shutil.which("uv")
    if uv_bin is not None:
        result = subprocess.run(
            [
                uv_bin,
                "pip",
                "compile",
                "--python-version",
                "3.13",
                "--python-platform",
                "x86_64-unknown-linux-gnu",
                "--no-header",
                str(COLAB_REQS_PATH),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"Python 3.13 Linux resolution failed for requirements-colab.txt:\n{result.stderr}"
        )


def _get_setup_cell_source() -> str:
    """Return the source code of the notebook's 'Clone Repo & Install Dependencies' cell."""
    nb = json.loads(COLAB_NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]
    for idx, cell in enumerate(cells):
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "markdown" and "Clone Repo & Install Dependencies" in src:
            next_cell = cells[idx + 1]
            next_src = next_cell["source"]
            return "".join(next_src) if isinstance(next_src, list) else next_src
    raise AssertionError("Could not find 'Clone Repo & Install Dependencies' cell in notebook")


def test_colab_notebook_setup_uses_lean_spec_and_preserves_abi() -> None:
    """Notebook setup installs requirements-colab.txt + editable tcrp with ABI constraints."""
    src = _get_setup_cell_source()

    assert "requirements-colab.txt" in src, "Setup cell must install requirements-colab.txt"
    for legacy_ref in (
        "requirements.txt",
        "requirements-training.txt",
        "fastapi",
        "uvicorn",
        "gradio",
        "plotly",
        "nonconformist",
        "import-linter",
        "slowapi",
        "websockets",
    ):
        assert legacy_ref not in src, f"Setup cell still references legacy dependency: {legacy_ref}"

    # Must constrain preinstalled torch, torchvision, and numpy so pip preserves Colab's ABI
    assert "-c" in src and "constraints" in src.lower(), (
        "Setup cell must pass a pip constraints file pinning Colab's preinstalled ABI packages"
    )
    for abi_pkg in ("torch", "torchvision", "numpy"):
        assert abi_pkg in src, f"Setup cell must preserve preinstalled {abi_pkg} version"

    # Must install local tcrp package cleanly without triggering dependency re-resolution
    assert "--no-build-isolation" in src and "--no-deps" in src and "'-e', '.'" in src, (
        "Setup cell must install editable tcrp package with --no-build-isolation --no-deps"
    )

    # Fresh-process verification of scientific + package stack
    for mod in ("numpy", "pandas", "torch", "torchvision", "tcrp"):
        assert mod in src, f"Setup cell must verify {mod} importability in a fresh process"

