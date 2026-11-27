# 03: Add Python 3.13 Colab dependency spec and set up local Colab CLI workflow

**What to build:** Create a lean Python 3.13-compatible Colab dependency specification containing only ML training, synthetic generation, evaluation, benchmarking, and plotting dependencies while preserving Colab's preinstalled PyTorch, CUDA, and NumPy ABI; update the notebook dependency installation step to use it; install and authenticate Google Colab CLI tooling locally on macOS; verify a live Colab Pro L4 runtime can clone the revision branch, install dependencies cleanly on Python 3.13, mount Google Drive, and report the NVIDIA L4 GPU.

**Blocked by:** 01: Clone repository, archive full-stack application, and prune to research-only layout

**Status:** open

- [ ] A lean Colab-only dependency specification exists without legacy API, UI, or dev pins and passes Python 3.13 resolution cleanly.
- [ ] The Colab production notebook setup step uses the lean Colab dependency specification and preserves the runtime's preinstalled PyTorch, CUDA, and NumPy ABI without binary incompatibility errors.
- [ ] Local Google Colab CLI and OAuth authentication are installed and verified against the user's Colab Pro account.
- [ ] A live smoke check on a Colab Pro L4 runtime successfully clones the revision branch, completes dependency installation with exit code 0, mounts Google Drive, and verifies CUDA access to the NVIDIA L4 GPU.
