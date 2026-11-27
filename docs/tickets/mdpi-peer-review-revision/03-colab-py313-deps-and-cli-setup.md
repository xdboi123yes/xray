# 03: Add Python 3.13 Colab dependency spec and set up local Colab CLI workflow

**What to build:** Create a lean Python 3.13-compatible Colab dependency specification containing only ML training, synthetic generation, evaluation, benchmarking, and plotting dependencies while preserving Colab's preinstalled PyTorch, CUDA, and NumPy ABI; update the notebook dependency installation step to use it; install and authenticate Google Colab CLI tooling locally on macOS; verify a live Colab Pro L4 runtime can clone the revision branch, install dependencies cleanly on Python 3.13, mount Google Drive, and report the NVIDIA L4 GPU.

**Blocked by:** 01: Clone repository, archive full-stack application, and prune to research-only layout

**Status:** resolved

- [x] A lean Colab-only dependency specification exists without legacy API, UI, or dev pins and passes Python 3.13 resolution cleanly.
- [x] The Colab production notebook setup step uses the lean Colab dependency specification and preserves the runtime's preinstalled PyTorch, CUDA, and NumPy ABI without binary incompatibility errors.
- [x] Local Google Colab CLI and OAuth authentication are installed and verified against the user's Colab Pro account.
- [x] A live smoke check on a Colab Pro L4 runtime successfully clones the revision branch, completes dependency installation with exit code 0, mounts Google Drive, and verifies CUDA access to the NVIDIA L4 GPU.

## Comments

- Added lean Python 3.13 dependency spec at `requirements-colab.txt` (omitting `torch`, `torchvision`, `numpy`, and all legacy API/UI/dev packages) and updated Cell 3 of `notebooks/xray_colab_produce_all.ipynb` to pin preinstalled `torch`/`torchvision`/`numpy` via pip constraints, install `requirements-colab.txt` + editable `tcrp` (`-e . --no-build-isolation --no-deps`), and verify fresh-process imports (`802d0db`).
- Added seam tests in `tests/unit/test_colab_setup.py` verifying dependency hygiene, Python 3.13 Linux x86_64 resolution via `uv pip compile`, and notebook setup cell invariants. Full local suite passed (104/104, 85.15% coverage, ruff/mypy/import-linter/comment-language green).
- Installed `google-colab-cli==0.8.0` via `uv tool install google-colab-cli` and authenticated OAuth2 against `alperen.yoruk.gm@gmail.com` (Colab Pro, 398.39 compute units remaining after smoke test).
- Verified live on Colab Pro L4 (`gpu-l4-s-ass1a0-3s0pw0yogbd9p`, Python 3.13.16): cloned `rev/mdpi-peer-review` @ `802d0db`, `pip install -r requirements-colab.txt -c /tmp/colab-abi-constraints.txt` exited 0, preserved exact ABI (`torch 2.14.1+cu130`, `torchvision 0.29.1+cu130`, `numpy 2.4.6`), verified `NVIDIA L4` (23.7 GB VRAM) with a `tcrp` CUDA forward pass, verified `/content/drive/MyDrive` (`xray_outputs` present), and terminated the smoke runtime cleanly (`0` active assignments).
