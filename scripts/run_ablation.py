"""CLI wrapper script to run Week 4 & Week 5 Ablation experiments.

Enables targeted execution of A8, A9, A11, A12, A13, A14, and A15, compiling summary figures and CSV tables.

With ``--skip-existing`` the script inspects the outputs directory and skips any
experiment whose trained weights (training ablations) or evaluation results
(evaluation ablations) are already present. This makes the notebook idempotent:
after restoring previously trained models/results from Google Drive, re-running
only spends GPU time on the experiments that are still missing.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from application.orchestration.ablation_runner import AblationRunner


def _run_name_of(config: dict) -> str | None:
    """Extract the ``--run-name`` value from an ablation config's argument list."""
    args = config.get("args", [])
    if "--run-name" in args:
        idx = args.index("--run-name")
        if idx + 1 < len(args):
            return str(args[idx + 1])
    return None


def completion_marker(config: dict) -> str | None:
    """Return the artifact path that indicates the experiment is already done.

    Training ablations (``scripts/train_tier2.py``) finish by writing
    ``outputs/models/<run_name>/best_model.pth``. Evaluation ablations
    (``scripts/evaluate_*.py``) finish by writing ``outputs/results/<run_name>.json``.
    """
    run_name = _run_name_of(config)
    if not run_name:
        return None
    script = config.get("script", "")
    if "train" in script:
        return os.path.join("outputs", "models", run_name, "best_model.pth")
    return os.path.join("outputs", "results", f"{run_name}.json")


def is_complete(config: dict) -> bool:
    """True if the experiment's output artifact already exists.

    For training weights we also scan nested subdirectories, so a checkpoint that
    was restored into an unexpected nesting level (e.g. from a re-zipped archive)
    still counts as complete and is not retrained.
    """
    marker = completion_marker(config)
    if not marker:
        return False
    if os.path.exists(marker):
        return True
    # Nesting-proof fallback for restored training checkpoints.
    if marker.endswith("best_model.pth"):
        run_name = _run_name_of(config) or ""
        nested = glob.glob(
            os.path.join("outputs", "models", "**", run_name, "best_model.pth"),
            recursive=True,
        )
        return len(nested) > 0
    return False


def main() -> None:
    """CLI entrypoint for running chest X-ray tiered ablation experiments."""
    parser = argparse.ArgumentParser(description="Run Week 4 & Week 5 Ablation Experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["A8", "A9", "A11", "A12", "A13", "A14", "A15"],
        choices=["A8", "A9", "A11", "A12", "A13", "A14", "A15"],
        help="List of specific ablation IDs to execute sequentially.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If active, runs training steps for only 1 epoch.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip experiments whose trained weights or evaluation results already "
            "exist under outputs/. Use after restoring models from Google Drive to "
            "focus compute only on the experiments that are still missing."
        ),
    )
    args = parser.parse_args()

    print("Initializing AblationRunner CLI...")
    runner = AblationRunner()

    # Build the execution plan up front so the user can see what will run vs skip.
    pending: list[str] = []
    skipped_ids: list[str] = []
    print("\n" + "=" * 60)
    print("ABLATION EXECUTION PLAN")
    print("=" * 60)
    for exp_id in args.experiments:
        config = runner.ablation_configs.get(exp_id, {})
        marker = completion_marker(config)
        if args.skip_existing and is_complete(config):
            skipped_ids.append(exp_id)
            print(f"  [SKIP] {exp_id:<4} already complete  ->  {marker}")
        else:
            pending.append(exp_id)
            tag = "RUN " if not (args.skip_existing) else "RUN*"
            print(f"  [{tag}] {exp_id:<4} will execute       ->  {marker}")
    if args.skip_existing:
        print(
            f"\nSummary: {len(skipped_ids)} already done (skipped), "
            f"{len(pending)} to run -> {pending if pending else 'nothing, all complete!'}"
        )
    print("=" * 60 + "\n")

    all_results = []

    # Record skipped experiments so they still appear in the summary table.
    for exp_id in skipped_ids:
        config = runner.ablation_configs.get(exp_id, {})
        all_results.append(
            {
                "ablation_id": exp_id,
                "name": config.get("name", exp_id),
                "status": "SKIPPED",
                "marker": completion_marker(config),
            }
        )

    for exp_id in pending:
        try:
            res = runner.run_experiment(exp_id, dry_run=args.dry_run)
            all_results.append(res)
        except Exception as ex:
            print(f"Error executing experiment {exp_id}: {ex}")
            all_results.append(
                {
                    "ablation_id": exp_id,
                    "status": "ERROR",
                    "error": str(ex),
                }
            )

    # Compile and save ablation summary
    df = pd.DataFrame(all_results)
    os.makedirs("outputs/results", exist_ok=True)
    summary_path = "outputs/results/ablation_week5_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(f"ABLATION EXECUTION COMPLETE. Summary saved to: {summary_path}")
    print("=" * 60)
    # 'name' is absent only for hard-errored rows; guard the column selection.
    cols = [c for c in ["ablation_id", "name", "status"] if c in df.columns]
    print(df[cols])
    print("=" * 60)


if __name__ == "__main__":
    main()
