"""
Master pipeline script for running the complete spatial clustering analysis.

This script orchestrates the execution of all notebooks in sequence,
with proper session management and logging.

Usage:
    python scripts/run_pipeline.py --profile default
    python scripts/run_pipeline.py --profile optimized --generate-report
"""

import sys
import argparse
from pathlib import Path
import subprocess
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.session import SessionManager
from src.utils.logger import setup_logger


NOTEBOOKS = [
    "01_explore_dataset.ipynb",
    "02_baseline.ipynb",
    "03_spatial_weighted_similarity.ipynb",
    "04_multiview_clustering.ipynb",
    "05_final_plots.ipynb",
    "06_sensitivity_analysis.ipynb",
]

# Timeout settings per notebook (in seconds, -1 = no timeout)
NOTEBOOK_TIMEOUTS = {
    "01_explore_dataset.ipynb": 600,
    "02_baseline.ipynb": 600,
    "03_spatial_weighted_similarity.ipynb": 600,
    "04_multiview_clustering.ipynb": 600,
    "05_final_plots.ipynb": 600,
    "06_sensitivity_analysis.ipynb": -1,  # No timeout - sensitivity analysis may take 1-2 hours
}


def run_pipeline(
    profile: str = "default",
    notebooks: list = None,
    generate_report: bool = True,
    verbose: bool = True
):
    """
    Run the complete pipeline.

    Parameters
    ----------
    profile : str
        Configuration profile to use.
    notebooks : list, optional
        List of notebooks to run. If None, runs all notebooks.
    generate_report : bool
        Whether to generate report after completion.
    verbose : bool
        Whether to print detailed output.

    Notes
    -----
    Notebook 06 (sensitivity analysis) may take 1-2 hours to complete.
    """
    # Create session
    session = SessionManager.create_session(
        profile=profile,
        description=f"Full pipeline run with {profile} profile"
    )

    logger = setup_logger(
        "pipeline",
        log_file=session.logs_dir / "pipeline.log"
    )

    logger.info(f"Starting pipeline with profile: {profile}")
    logger.info(f"Session ID: {session.session_id}")

    # Determine which notebooks to run
    if notebooks is None:
        notebooks_to_run = NOTEBOOKS.copy()
    else:
        notebooks_to_run = notebooks

    notebooks_dir = Path("notebooks")
    results = []

    # Run each notebook
    for i, notebook in enumerate(notebooks_to_run, 1):
        logger.info(f"[{i}/{len(notebooks_to_run)}] Running {notebook}...")

        notebook_path = notebooks_dir / notebook
        if not notebook_path.exists():
            logger.error(f"Notebook not found: {notebook_path}")
            results.append({"notebook": notebook, "status": "failed", "error": "not_found"})
            continue

        try:
            # Get timeout for this notebook
            timeout = NOTEBOOK_TIMEOUTS.get(notebook, 600)

            # Execute notebook using nbconvert
            cmd = [
                "jupyter",
                "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                f"--ExecutePreprocessor.timeout={timeout}",
                str(notebook_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"OK! {notebook} completed successfully")
            results.append({"notebook": notebook, "status": "success"})

        except subprocess.CalledProcessError as e:
            logger.error(f"NOK! {notebook} failed")
            logger.error(f"Error: {e.stderr}")
            results.append({
                "notebook": notebook,
                "status": "failed",
                "error": str(e)
            })

    # Save execution summary
    summary = {
        "session_id": session.session_id,
        "profile": profile,
        "notebooks": results,
        "total": len(results),
        "succeeded": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed")
    }

    summary_file = session.run_dir / "execution_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline execution completed")
    logger.info(f"  Succeeded: {summary['succeeded']}/{summary['total']}")
    logger.info(f"  Failed: {summary['failed']}/{summary['total']}")
    logger.info(f"{'='*60}\n")

    # End session
    SessionManager.end_session()

    # Generate report if requested
    if generate_report:
        logger.info("Generating report...")
        try:
            from scripts.generate_report import generate_report as gen_report
            gen_report(session.session_id, output_format="html")
            logger.info("OK! Report generated successfully")
        except Exception as e:
            logger.error(f"NOK! Report generation failed: {e}")

    # Update optimal config if sensitivity analysis completed
    if "06_sensitivity_analysis.ipynb" in [r["notebook"] for r in results if r["status"] == "success"]:
        logger.info("Updating optimal configuration from sensitivity analysis...")
        try:
            from scripts.update_optimal_config import update_optimal_config
            update_optimal_config(session.session_id)
            logger.info("OK! Optimal configuration updated in config.yaml")
        except Exception as e:
            logger.warning(f"WRNG:  Could not update optimal config: {e}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the spatial clustering pipeline"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=["default", "optimized", "auto_optimize"],
        help="Configuration profile to use"
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        help="Specific notebooks to run (default: all)"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    run_pipeline(
        profile=args.profile,
        notebooks=args.notebooks,
        generate_report=not args.no_report,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
