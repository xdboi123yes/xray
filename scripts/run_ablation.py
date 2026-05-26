"""CLI wrapper script to run Week 4 & Week 5 Ablation experiments.

Enables targeted execution of A8, A9, A11, A12, A13, A14, and A15, compiling summary figures and CSV tables.
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from application.orchestration.ablation_runner import AblationRunner


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
    args = parser.parse_args()

    print(f"Initializing AblationRunner CLI...")
    runner = AblationRunner()

    all_results = []
    for exp_id in args.experiments:
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
    print(df[["ablation_id", "name", "status"]])
    print("=" * 60)


if __name__ == "__main__":
    main()
