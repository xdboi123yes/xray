# University server remote runner

## Before connecting

The university runner clones GitHub. Therefore all pipeline/notebook fixes must be committed and pushed to the configured branch before starting the server job.

The server needs:

- SSH access.
- Git and Python 3.10+.
- NVIDIA drivers and a CUDA-capable GPU for the real run.
- At least 120 GiB free disk space.
- `tmux` for disconnect-safe execution.
- Passwordless sudo for automatic TeX installation, or Tectonic/latexmk already installed.

## Upload the launcher once

From the local machine:

```bash
scp remote_runners/university_pipeline.sh USER@SERVER:~/university_pipeline.sh
ssh USER@SERVER 'chmod +x ~/university_pipeline.sh'
```

## Start the detached production run

```bash
ssh USER@SERVER \
  'XRAY_REPO_URL=https://github.com/xdboi123yes/xray.git \
   XRAY_BRANCH=main \
   XRAY_WORKSPACE=$HOME/xray-production \
   ~/university_pipeline.sh start'
```

The SSH connection may then be closed. Training continues inside `tmux`.

## Watch live progress

```bash
ssh -t USER@SERVER '~/university_pipeline.sh attach'
```

Detach without stopping the run by pressing `Ctrl-b`, then `d`.

For a noninteractive status snapshot:

```bash
ssh USER@SERVER '~/university_pipeline.sh status'
```

## Run a safe preflight first

To test installation, GPU detection and disk capacity without training:

```bash
ssh -t USER@SERVER '~/university_pipeline.sh run --preflight-only'
```

## Folder structure created automatically

```text
~/xray-production/
├── repo/                 # clean GitHub checkout
├── venv/                 # isolated Python environment
├── runtime/              # downloaded/cache data for the notebook
├── logs/                 # bootstrap and streaming pipeline logs
├── runs/<timestamp>/     # executed notebook
├── artifacts/<timestamp>/
│   ├── manuscript.pdf
│   ├── manuscript.tex
│   ├── executed_notebook.ipynb
│   ├── outputs/results/
│   ├── outputs/provenance/
│   └── figures/
├── artifacts/<timestamp>.zip
└── state.json
```

The ZIP is created only after the patient-disjoint scientific-integrity gate passes and the manuscript compiles.

## Download the completed presentation bundle

Find the latest artifact:

```bash
ssh USER@SERVER 'ls -1t ~/xray-production/artifacts/*.zip | head -n 1'
```

Then download it:

```bash
scp USER@SERVER:~/xray-production/artifacts/YYYYMMDD-HHMMSS.zip .
```

## Failure recovery

Read status and logs:

```bash
ssh USER@SERVER '~/university_pipeline.sh status'
```

The notebook keeps its battle-tested step markers and subprocess isolation. Fix the reported cause and start another run. During the mandatory clean rebuild, the integrity gate refuses a bundle containing a mixture of old and new checkpoints.

Do not bypass the gate or switch `PATIENT_DISJOINT_REBUILD` off until one complete production run succeeds.
