"""
Update config.yaml with optimal parameters from sensitivity analysis.

This script reads the results from notebook 06 (sensitivity analysis)
and automatically updates the 'optimized' profile in config.yaml.

Usage:
    python scripts/update_optimal_config.py
    python scripts/update_optimal_config.py --session run_2026-02-06_19-25-50
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.session import SessionManager


def update_optimal_config(session_id: str = None):
    """
    Update config.yaml with optimal parameters from sensitivity analysis.

    Parameters
    ----------
    session_id : str, optional
        Session ID to read results from. If None, uses current/latest session.
    """
    # Find session directory
    if session_id:
        session_dir = Path("results/runs") / session_id
    else:
        # Try to get current session or find latest
        runs_dir = Path("results/runs")
        if not runs_dir.exists():
            print("NOK! No sessions found. Run the pipeline first.")
            return False

        sessions = sorted(runs_dir.glob("run_*"))
        if not sessions:
            print("NOK! No sessions found. Run the pipeline first.")
            return False

        session_dir = sessions[-1]
        session_id = session_dir.name

    if not session_dir.exists():
        print(f"NOK! Session not found: {session_id}")
        return False

    print(f"Reading results from session: {session_id}")

    # Define paths
    metrics_dir = session_dir / "metrics"

    # Check if sensitivity analysis results exist
    sensitivity_file = metrics_dir / "sensitivity_results.npy"
    optimized_resolutions_file = metrics_dir / "optimized_resolutions.npy"

    if not sensitivity_file.exists():
        print("WARNING: Sensitivity analysis results not found.")
        print("   Make sure notebook 06 completed successfully.")
        return False

    # Load sensitivity results
    print("Loading sensitivity analysis results...")
    sensitivity_results = np.load(sensitivity_file, allow_pickle=True).item()

    # Extract optimal weights
    combinations = sensitivity_results['combinations']
    silhouette_scores = sensitivity_results['silhouette_scores']

    # Find combination with highest Silhouette score
    best_idx = np.argmax(silhouette_scores)
    optimal_weights = combinations[best_idx]

    print(f"\nOK! Optimal weights found:")
    print(f"   α (expression): {optimal_weights[0]:.2f}")
    print(f"   β (spatial):    {optimal_weights[1]:.2f}")
    print(f"   γ (MoG):        {optimal_weights[2]:.2f}")
    print(f"   Silhouette:     {silhouette_scores[best_idx]:.3f}")

    # Load optimized resolutions if available
    optimal_resolution = 1.0  # default
    if optimized_resolutions_file.exists():
        print("\nLoading optimized resolutions...")
        optimized_resolutions = np.load(optimized_resolutions_file, allow_pickle=True).item()
        # Use weighted view resolution
        if 'weighted' in optimized_resolutions:
            optimal_resolution = optimized_resolutions['weighted']
            print(f"OK! Optimal resolution (weighted view): {optimal_resolution:.2f}")

    # Load current config
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("NOK! config.yaml not found!")
        return False

    print(f"\nUpdating {config_path}...")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update optimized profile
    if 'profiles' not in config:
        config['profiles'] = {}

    if 'optimized' not in config['profiles']:
        # Copy from default if doesn't exist
        config['profiles']['optimized'] = config['profiles'].get('default', {}).copy()

    # Update weights
    config['profiles']['optimized']['similarity']['weights']['alpha'] = float(optimal_weights[0])
    config['profiles']['optimized']['similarity']['weights']['beta'] = float(optimal_weights[1])
    config['profiles']['optimized']['similarity']['weights']['gamma'] = float(optimal_weights[2])

    # Update resolution
    config['profiles']['optimized']['clustering']['resolution'] = float(optimal_resolution)
    config['profiles']['optimized']['clustering']['optimize_resolution'] = False  # Already optimized

    # Add comment
    config['profiles']['optimized']['_updated'] = {
        'session': session_id,
        'date': session_id.replace('run_', '').replace('_', ' ').replace('-', ':'),
        'note': 'Auto-updated from sensitivity analysis (notebook 06)'
    }

    # Save updated config
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"OK! config.yaml updated successfully!")
    print(f"\n{'='*60}")
    print("UPDATED OPTIMIZED PROFILE:")
    print(f"{'='*60}")
    print(f"  Resolution:     {optimal_resolution:.2f}")
    print(f"  Alpha (expr):   {optimal_weights[0]:.2f}")
    print(f"  Beta (spatial): {optimal_weights[1]:.2f}")
    print(f"  Gamma (MoG):    {optimal_weights[2]:.2f}")
    print(f"{'='*60}")

    print(f"\n!!! Next steps:")
    print(f"   1. Review the updated config.yaml")
    print(f"   2. Run with optimized profile:")
    print(f"      python scripts/run_pipeline.py --profile optimized")
    print(f"   3. Compare results with default profile")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update config.yaml with optimal parameters from sensitivity analysis"
    )
    parser.add_argument(
        "--session",
        type=str,
        help="Session ID to read results from (default: latest)"
    )

    args = parser.parse_args()

    success = update_optimal_config(args.session)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
