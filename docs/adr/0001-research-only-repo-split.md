# Split public research repository from full-stack application archive

Status: accepted

We split `xdboi123yes/xray` into a minimal, research-only reproducibility repository for the MDPI *Diagnostics* manuscript (`configs/`, `src/tcrp/`, `scripts/`, `notebooks/`, `paper/`, `tests/`), while preserving the web application (`web/`), container stack (`docker/`, `docker-compose*.yml`), `remote_runners/`, and legacy Springer artifacts in a separate local archive repository at `/Users/alperen/Developer/xray-app`.

## Considered Options

- Keep `web/`, `docker/`, and `remote_runners/` inside `xdboi123yes/xray` alongside the paper code.
- Delete non-paper application files permanently without archiving.
- Split into a clean public research repo (`xray`) and a separate local application archive (`xray-app`).

## Consequences

- Peer reviewers and readers get a focused, reproducible repository matching the paper's claims and `v1.0-mdpi-rev1` GitHub Release (code, frozen `requirements-lock.txt`, and model weights).
- CI workflows and dependency files (`requirements-colab.txt`) no longer break on API/UI packages unrelated to training or evaluation.
- Application code remains intact and organized in `/Users/alperen/Developer/xray-app` for future product work.
