"""
Create a session from manually run notebooks.

This script collects results from results/metrics and results/plots
and organizes them into a proper session for report generation.
"""

import shutil
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.session import SessionManager


def create_session_from_manual_run(description="Manual notebook run"):
    """
    Create a session from manually executed notebooks.

    Copies results from results/metrics/ and results/plots/
    into a proper session structure.
    """
    print("Creating session from manual run...")

    # Create new session
    session = SessionManager.create_session(
        profile="default",
        description=description
    )

    # Copy metrics
    old_metrics = Path("results/metrics")
    if old_metrics.exists():
        print("Copying metrics...")
        for metric_file in old_metrics.glob("*"):
            if metric_file.is_file() and metric_file.name != ".gitkeep":
                dest = session.metrics_dir / metric_file.name
                shutil.copy2(metric_file, dest)
                print(f"  OK!  {metric_file.name}")

    # Copy plots
    old_plots = Path("results/plots")
    if old_plots.exists():
        print("Copying plots...")
        for plot_file in old_plots.glob("*.png"):
            dest = session.plots_dir / plot_file.name
            shutil.copy2(plot_file, dest)
            print(f"  OK!  {plot_file.name}")

    # Update session metadata
    with open(session.run_dir / "session.json", "r") as f:
        metadata = json.load(f)

    metadata["source"] = "manual_run"
    metadata["manual_run_date"] = datetime.now().isoformat()

    with open(session.run_dir / "session.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOK!  Session created: {session.session_id}")
    print(f"   Location: {session.run_dir}")

    # End session
    SessionManager.end_session()

    print("\nOK!  You can now generate a report:")
    print(f"   python scripts/generate_report.py --session {session.session_id}")

    return session.session_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create session from manual notebook run"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Manual notebook run",
        help="Session description"
    )

    args = parser.parse_args()

    session_id = create_session_from_manual_run(args.description)
