#!/usr/bin/env bash
set -Eeuo pipefail

# One-command bootstrap for a remote university server.
# Configuration is via environment variables so the same command works through SSH.
REPO_URL="${XRAY_REPO_URL:-https://github.com/xdboi123yes/xray.git}"
BRANCH="${XRAY_BRANCH:-main}"
WORKSPACE="${XRAY_WORKSPACE:-$HOME/xray-production}"
REPO_DIR="$WORKSPACE/repo"
VENV_DIR="$WORKSPACE/venv"
LOG_DIR="$WORKSPACE/logs"

COMMAND="${1:-run}"
if [[ "$COMMAND" == "start" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux is required for detached remote runs"; exit 1; }
  if tmux has-session -t xray-production 2>/dev/null; then
    echo "xray-production is already running. Attach with: $0 attach"
    exit 0
  fi
  tmux new-session -d -s xray-production "XRAY_REPO_URL='$REPO_URL' XRAY_BRANCH='$BRANCH' XRAY_WORKSPACE='$WORKSPACE' bash '$(realpath "$0")' run"
  echo "Started detached session: xray-production"
  echo "Attach: $0 attach"
  echo "Status: $0 status"
  exit 0
elif [[ "$COMMAND" == "attach" ]]; then
  exec tmux attach-session -t xray-production
elif [[ "$COMMAND" == "status" ]]; then
  if [[ -f "$WORKSPACE/state.json" ]]; then
    cat "$WORKSPACE/state.json"
  else
    echo "No state file yet."
  fi
  latest="$(ls -1t "$WORKSPACE"/logs/pipeline-*.log 2>/dev/null | head -n 1 || true)"
  [[ -n "$latest" ]] && { echo; echo "--- latest log ---"; tail -n 30 "$latest"; }
  exit 0
elif [[ "$COMMAND" != "run" ]]; then
  echo "Usage: $0 {start|attach|status|run}"
  exit 2
fi
shift || true

mkdir -p "$WORKSPACE" "$LOG_DIR" "$WORKSPACE/artifacts" "$WORKSPACE/runtime"
BOOT_LOG="$LOG_DIR/bootstrap.log"
exec > >(tee -a "$BOOT_LOG") 2>&1

say() { printf '\n\033[1;36m[xray-control]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[failed]\033[0m %s\n' "$*" >&2; exit 1; }

say "workspace: $WORKSPACE"

command -v git >/dev/null 2>&1 || die "git is required. Ask the server administrator to install git."
command -v python3 >/dev/null 2>&1 || die "Python 3.10+ is required. Ask the server administrator to install python3 and python3-venv."

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' \
  || die "Python $PY_VER is too old; Python 3.10+ is required."

if ! command -v tectonic >/dev/null 2>&1 && ! command -v latexmk >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    say "installing TeX compiler and native build prerequisites"
    sudo apt-get update || true
    sudo apt-get install -y python3-venv python3-dev build-essential latexmk \
      texlive-latex-base texlive-latex-extra texlive-fonts-recommended || true
  elif command -v cargo >/dev/null 2>&1; then
    say "installing Tectonic through Cargo"
    cargo install tectonic --locked || true
  else
    say "No TeX compiler found; PDF compilation will be skipped gracefully."
  fi
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  say "cloning $REPO_URL ($BRANCH)"
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
else
  say "updating repository"
  git -C "$REPO_DIR" fetch origin "$BRANCH"
  if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
    say "Cleaning local workspace modifications to sync with remote..."
    git -C "$REPO_DIR" checkout .
    git -C "$REPO_DIR" clean -fd
  fi
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  say "creating isolated Python environment"
  python3 -m venv "$VENV_DIR" || die "venv creation failed; install python3-venv on the server"
fi

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
say "installing/updating project dependencies"
"$PY" -m pip install --upgrade pip wheel setuptools
"$PIP" install -r "$REPO_DIR/requirements.txt" -r "$REPO_DIR/requirements-training.txt"
"$PIP" install rich psutil jupyter nbconvert ipykernel

say "registering notebook kernel"
"$PY" -m ipykernel install --user --name xray-production --display-name "XRay Production" >/dev/null

export XRAY_PROJECT_ROOT="$REPO_DIR"
export XRAY_RUNTIME_ROOT="$WORKSPACE/runtime"
export XRAY_WORKSPACE="$WORKSPACE"
export PYTHONUNBUFFERED=1

say "starting production controller"
exec "$PY" "$REPO_DIR/scripts/university_pipeline.py" "$@"
