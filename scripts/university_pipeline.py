#!/usr/bin/env python3
"""Remote-safe production controller for the patient-disjoint manuscript rerun."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class Stage:
    name: str
    label: str
    status: str = "WAITING"
    detail: str = ""
    elapsed: float = 0.0


@dataclass
class State:
    started_at: str
    commit: str = ""
    current: str = ""
    stages: list[Stage] = field(default_factory=list)
    last_lines: list[str] = field(default_factory=list)


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = Path(os.environ.get("XRAY_PROJECT_ROOT", Path.cwd())).resolve()
        self.workspace = Path(os.environ.get("XRAY_WORKSPACE", self.repo.parent)).resolve()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.workspace / "runs" / stamp
        self.log_dir = self.workspace / "logs"
        self.artifact_dir = self.workspace / "artifacts" / stamp
        for path in (self.run_dir, self.log_dir, self.artifact_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"pipeline-{stamp}.log"
        self.state_path = self.workspace / "state.json"
        self.console = Console()
        self.state = State(
            started_at=datetime.now(timezone.utc).isoformat(),
            stages=[
                Stage("preflight", "Server + GPU preflight"),
                Stage("notebook", "Patient-disjoint notebook"),
                Stage("integrity", "Scientific integrity gate"),
                Stage("paper", "Tables, figures + manuscript"),
                Stage("package", "Presentation bundle"),
            ],
        )
        self._stop = False
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)

    def _signal(self, *_: object) -> None:
        self._stop = True

    def save_state(self) -> None:
        payload = {
            "started_at": self.state.started_at,
            "commit": self.state.commit,
            "current": self.state.current,
            "run_dir": str(self.run_dir),
            "log": str(self.log_path),
            "stages": [vars(s) for s in self.state.stages],
        }
        self.state_path.write_text(json.dumps(payload, indent=2) + "\n")

    def stage(self, name: str) -> Stage:
        return next(s for s in self.state.stages if s.name == name)

    def render(self) -> Group:
        header = Text("XRAY / UNIVERSITY PRODUCTION CONTROL", style="bold black on bright_cyan")
        meta = Table.grid(expand=True)
        meta.add_column(style="dim")
        meta.add_column(justify="right")
        meta.add_row(f"commit {self.state.commit[:12] or 'pending'}", f"workspace {self.workspace}")
        table = Table(expand=True, box=None, padding=(0, 1))
        table.add_column("STAGE", style="bold")
        table.add_column("STATE", width=11)
        table.add_column("ELAPSED", justify="right", width=10)
        table.add_column("DETAIL")
        styles = {"WAITING": "dim", "RUNNING": "yellow", "DONE": "green", "FAILED": "red"}
        for s in self.state.stages:
            table.add_row(s.label, Text(s.status, style=styles[s.status]), f"{s.elapsed:,.0f}s", s.detail[-70:])
        tail = "\n".join(self.state.last_lines[-8:]) or "waiting for first process output..."
        return Group(
            Panel(Group(header, meta), border_style="bright_cyan"),
            Panel(table, title="PIPELINE", border_style="cyan"),
            Panel(tail, title="LIVE LOG", border_style="grey50"),
        )

    def run_command(
        self, stage_name: str, cmd: Sequence[str], env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        stage = self.stage(stage_name)
        stage.status = "RUNNING"
        stage.detail = " ".join(cmd)
        self.state.current = stage_name
        self.save_state()
        started = time.monotonic()
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(cmd)}\n")
            proc = subprocess.Popen(
                list(cmd), cwd=cwd or self.repo, env=full_env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                log.write(raw)
                log.flush()
                self.state.last_lines.append(line)
                self.state.last_lines = self.state.last_lines[-40:]
                stage.elapsed = time.monotonic() - started
                if self._stop:
                    proc.terminate()
                    raise KeyboardInterrupt
            rc = proc.wait()
        stage.elapsed = time.monotonic() - started
        if rc:
            stage.status = "FAILED"
            stage.detail = f"exit {rc}; see {self.log_path}"
            self.save_state()
            raise RuntimeError(stage.detail)
        stage.status = "DONE"
        stage.detail = "complete"
        self.save_state()

    def preflight(self) -> None:
        stage = self.stage("preflight")
        stage.status = "RUNNING"
        started = time.monotonic()
        self.state.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        disk = shutil.disk_usage(self.workspace)
        if disk.free < self.args.min_free_gb * 1024**3:
            raise RuntimeError(f"only {disk.free / 1024**3:.1f} GiB free; need {self.args.min_free_gb} GiB")
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        if gpu.returncode and not self.args.allow_cpu:
            raise RuntimeError("NVIDIA GPU not available; use --allow-cpu only for plumbing tests")
        stage.status = "DONE"
        stage.elapsed = time.monotonic() - started
        stage.detail = gpu.stdout.strip() or "CPU-only override"
        self.save_state()

    def execute_notebook(self) -> None:
        notebook = self.repo / "notebooks" / "xray_colab_produce_all.ipynb"
        executed = self.run_dir / "xray_colab_produce_all.executed.ipynb"
        env = {
            "XRAY_PROJECT_ROOT": str(self.repo),
            "XRAY_RUNTIME_ROOT": str(self.workspace / "runtime"),
            "PYTHONPATH": str(self.repo),
        }
        self.run_command(
            "notebook",
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
             "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=xray-production",
             "--output", str(executed), str(notebook)],
            env,
        )

    def verify_integrity(self) -> None:
        stage = self.stage("integrity")
        stage.status = "RUNNING"
        started = time.monotonic()
        rerun = self.repo / "outputs" / "provenance" / "rerun_manifest.json"
        split = self.repo / "outputs" / "provenance" / "split_manifest.json"
        if not rerun.exists() or not split.exists():
            raise RuntimeError("notebook finished without provenance manifests")
        data = json.loads(rerun.read_text())
        if data.get("protocol") != "patient-disjoint-v1":
            raise RuntimeError(f"unexpected protocol {data.get('protocol')}")
        stage.status = "DONE"
        stage.elapsed = time.monotonic() - started
        stage.detail = "patient-disjoint-v1 verified"
        self.save_state()

    def build_paper(self) -> None:
        updater = self.repo / "scripts" / "update_paper_from_results.py"
        if updater.exists():
            self.run_command("paper", [sys.executable, str(updater)])
        else:
            stage = self.stage("paper")
            stage.status = "RUNNING"
        paper = self.repo / "paper"
        if shutil.which("tectonic"):
            self.run_command("paper", ["tectonic", "manuscript.tex", "--keep-logs"], cwd=paper)
        elif shutil.which("latexmk"):
            self.run_command("paper", ["latexmk", "-pdf", "-interaction=nonstopmode", "manuscript.tex"], cwd=paper)
        elif shutil.which("pdflatex"):
            self.run_command("paper", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript.tex"], cwd=paper)
            self.run_command("paper", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript.tex"], cwd=paper)
        else:
            raise RuntimeError("Tectonic, latexmk or pdflatex is required to compile the final manuscript")
        if not (paper / "manuscript.pdf").exists():
            raise RuntimeError("paper/manuscript.pdf was not created")

    def package(self) -> None:
        stage = self.stage("package")
        stage.status = "RUNNING"
        started = time.monotonic()
        targets = {
            self.repo / "paper" / "manuscript.pdf": self.artifact_dir / "manuscript.pdf",
            self.repo / "paper" / "manuscript.tex": self.artifact_dir / "manuscript.tex",
            self.repo / "outputs" / "results": self.artifact_dir / "outputs" / "results",
            self.repo / "outputs" / "provenance": self.artifact_dir / "outputs" / "provenance",
            self.repo / "thesis" / "figures": self.artifact_dir / "figures",
        }
        for src, dst in targets.items():
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        shutil.copy2(self.run_dir / "xray_colab_produce_all.executed.ipynb", self.artifact_dir / "executed_notebook.ipynb")
        shutil.make_archive(str(self.artifact_dir), "zip", self.artifact_dir)
        stage.status = "DONE"
        stage.elapsed = time.monotonic() - started
        stage.detail = f"{self.artifact_dir}.zip"
        self.save_state()

    def run(self) -> None:
        with Live(self.render(), console=self.console, refresh_per_second=4) as live:
            def refresh() -> None:
                while not self._stop:
                    live.update(self.render())
                    time.sleep(0.25)
            thread = threading.Thread(target=refresh, daemon=True)
            thread.start()
            self.preflight()
            if not self.args.preflight_only:
                self.execute_notebook()
                self.verify_integrity()
                self.build_paper()
                self.package()
            self._stop = True
            live.update(self.render(), refresh=True)
        self.console.print(f"\n[bold green]READY[/] {self.artifact_dir}.zip")
        self.console.print(f"log: {self.log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--min-free-gb", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    Controller(parse_args()).run()
