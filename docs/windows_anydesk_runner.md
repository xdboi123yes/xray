# Windows + AnyDesk production runner

## Intended workflow

1. Connect to the Romanian university PC using AnyDesk.
2. Navigate to the `remote_runners/` directory in the repository (or copy `remote_runners/Run-XRay-Pipeline.bat` and `remote_runners/windows_pipeline.ps1` into one folder).
3. Double-click `Run-XRay-Pipeline.bat` (located in `remote_runners/`).
4. Click **Preflight only** first.
5. If preflight succeeds, click **Start full pipeline**.
6. AnyDesk may disconnect. The worker runs locally on the university PC and prevents Windows sleep while active.
7. Reconnect later and inspect the same control panel/logs.
8. Click **Open output folder** when complete.

On first launch, the control panel asks you to select your Kaggle `kaggle.json`. It copies the credential to `%USERPROFILE%\.kaggle\kaggle.json`; it is not copied into the repository or final bundle.

## What the launcher installs

When absent, it uses Windows Package Manager (`winget`) to install:

- Git
- Python 3.11
- MiKTeX

It then creates an isolated virtual environment and installs all repository training requirements, Jupyter, the notebook kernel and terminal-dashboard dependencies.

The PC must already have working NVIDIA drivers. CUDA/PyTorch compatibility is checked during preflight and notebook diagnostics.

## Automatic workspace

```text
%USERPROFILE%\XRay-Production\
├── repo\
├── venv\
├── runtime\
├── logs\
├── runs\<timestamp>\
├── artifacts\<timestamp>\
└── artifacts\<timestamp>.zip
```

The final ZIP contains the compiled manuscript, TeX source, executed notebook, result files, provenance manifests and figures.

## Important Windows settings

- Connect the PC to AC power.
- In Windows Update, pause updates during the production run if university policy permits.
- Ensure the GPU is not being shared with another training job.
- Ensure at least 120 GiB of free disk space.
- AnyDesk disconnecting is safe; rebooting or signing out is not.
- Do not close the production control window while training is active.

## GitHub requirement

The launcher clones the configured GitHub branch. Commit and push all notebook, pipeline and manuscript changes before running it. For a private repository, authenticate Git Credential Manager on the university PC or use an authorized repository URL.

## Logs and recovery

Bootstrap logs:

```text
%USERPROFILE%\XRay-Production\logs\windows-worker.log
%USERPROFILE%\XRay-Production\logs\windows-worker-error.log
```

Pipeline logs and state are under the same workspace. A failed run does not produce a presentation-ready ZIP. Fix the reported cause and start again; the notebook's existing recovery markers remain available, while the scientific-integrity gate prevents mixed old/new output delivery.
