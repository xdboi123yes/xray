# Remote Execution Runners

This directory contains standalone execution scripts and launchers for running the **Chest X-Ray Tiered Classification** training and evaluation pipeline on remote hardware.

---

## 📁 Included Runners

| Runner Script | Operating System | Description |
|---|---|---|
| [`Run-XRay-Pipeline.bat`](file:///Users/alperen/Developer/xray/remote_runners/Run-XRay-Pipeline.bat) | Windows | Double-click batch launcher to open the graphical WPF control panel. |
| [`windows_pipeline.ps1`](file:///Users/alperen/Developer/xray/remote_runners/windows_pipeline.ps1) | Windows | PowerShell WPF GUI controller with stay-awake prevention and Kaggle authentication. |
| [`university_pipeline.sh`](file:///Users/alperen/Developer/xray/remote_runners/university_pipeline.sh) | Linux / macOS | Shell runner for headless SSH / tmux remote server execution. |

---

## 🚀 Usage Summary

### Windows (AnyDesk / Local PC)
1. Copy or clone this repository to the remote Windows machine.
2. Double-click [`Run-XRay-Pipeline.bat`](file:///Users/alperen/Developer/xray/remote_runners/Run-XRay-Pipeline.bat).
3. Select `kaggle.json` when prompted.
4. Click **PREFLIGHT ONLY** to verify GPU, CUDA, Python 3.11, and dependencies.
5. Click **START FULL PIPELINE** to begin training.

### Linux / SSH Server
```bash
# Upload and run on remote server:
scp remote_runners/university_pipeline.sh USER@SERVER:~/university_pipeline.sh
ssh USER@SERVER 'chmod +x ~/university_pipeline.sh && ~/university_pipeline.sh start'
```

---

For full documentation and setup details, see:
- 📖 [Windows AnyDesk Guide](../docs/windows_anydesk_runner.md)
- 📖 [University Remote Linux Guide](../docs/university_remote_runner.md)
- 📖 [Patient-Disjoint Rerun Checklist](../docs/patient_disjoint_rerun_todo.md)
