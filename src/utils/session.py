"""
Session management for tracking pipeline runs.

Handles creation, retrieval, and management of execution sessions,
ensuring all notebooks in a pipeline run log to the same folder.
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


class SessionManager:
    """
    Manages execution sessions for the spatial clustering pipeline.

    A session represents a single pipeline run and tracks all outputs,
    logs, and metrics generated during that run.

    Sessions can be created automatically (when running notebooks individually)
    or manually (when running the full pipeline script).
    """

    SESSION_FILE = ".current_session"
    RUNS_DIR = "results/runs"

    def __init__(self, session_id: str, run_dir: Path, config: Dict[str, Any]):
        """
        Initialize session manager.

        Parameters
        ----------
        session_id : str
            Unique identifier for this session.
        run_dir : Path
            Directory where session outputs are stored.
        config : dict
            Configuration parameters for this session.
        """
        self.session_id = session_id
        self.run_dir = Path(run_dir)
        self.config = config
        self.dataset_path = config.get("dataset_path", "data/DLPFC-151673")
        self.start_time = datetime.now()

        # Create directory structure
        self.logs_dir = self.run_dir / "logs"
        self.metrics_dir = self.run_dir / "metrics"
        self.plots_dir = self.run_dir / "plots"

        for dir_path in [self.logs_dir, self.metrics_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_session(
        cls,
        profile: str = "default",
        description: Optional[str] = None
    ) -> SessionManager:
        """
        Create a new session.

        Parameters
        ----------
        profile : str
            Configuration profile to use ("default" or "optimized").
        description : str, optional
            Human-readable description of this run.

        Returns
        -------
        SessionManager
            New session instance.
        """
        from .config import load_config

        # Generate session ID from timestamp
        session_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")

        # Create run directory
        run_dir = Path(cls.RUNS_DIR) / session_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        config = load_config(profile)

        # Create session instance
        session = cls(session_id, run_dir, config)

        # Save session metadata
        metadata = {
            "session_id": session_id,
            "profile": profile,
            "dataset_path": config.get("dataset_path", "data/DLPFC-151673"),
            "description": description or f"Pipeline run with {profile} profile",
            "start_time": session.start_time.isoformat(),
            "config": config
        }

        with open(session.run_dir / "session.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save session ID to file
        with open(cls.SESSION_FILE, "w") as f:
            f.write(session_id)

        print(f"   Session created: {session_id}")
        print(f"   Profile: {profile}")
        print(f"   Dataset: {config.get('dataset_path', 'data/DLPFC-151673')}")
        print(f"   Output: {run_dir}")

        return session

    @classmethod
    def get_current_session(cls) -> Optional[SessionManager]:
        """
        Get the currently active session.

        Returns
        -------
        SessionManager or None
            Current session if one exists, None otherwise.
        """
        if not os.path.exists(cls.SESSION_FILE):
            return None

        try:
            with open(cls.SESSION_FILE, "r") as f:
                session_id = f.read().strip()

            run_dir = Path(cls.RUNS_DIR) / session_id

            if not run_dir.exists():
                warnings.warn(f"Session {session_id} directory not found")
                return None

            # Load session metadata
            with open(run_dir / "session.json", "r") as f:
                metadata = json.load(f)

            return cls(session_id, run_dir, metadata["config"])

        except Exception as e:
            warnings.warn(f"Error loading session: {e}")
            return None

    @classmethod
    def get_or_create_session(
        cls,
        profile: str = "default",
        auto_create: bool = True
    ) -> SessionManager:
        """
        Get current session or create new one if none exists.

        This is the main entry point for notebooks running in hybrid mode.

        Parameters
        ----------
        profile : str
            Configuration profile to use if creating new session.
        auto_create : bool
            Whether to auto-create session if none exists.

        Returns
        -------
        SessionManager
            Current or newly created session.
        """
        session = cls.get_current_session()

        if session is None:
            if auto_create:
                print("No active session found. Creating new session...")
                session = cls.create_session(profile=profile)
            else:
                raise RuntimeError("No active session. Run start_session.py first.")
        else:
            print(f"Using existing session: {session.session_id}")

        return session

    @classmethod
    def end_session(cls) -> bool:
        """
        End the current session.

        Returns
        -------
        bool
            True if session was ended successfully, False otherwise.
        """
        if not os.path.exists(cls.SESSION_FILE):
            print("No active session to end")
            return False

        session = cls.get_current_session()
        if session:
            # Update session metadata with end time
            metadata_file = session.run_dir / "session.json"
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            metadata["end_time"] = datetime.now().isoformat()
            metadata["status"] = "completed"

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Session ended: {session.session_id}")

        # Remove current session file
        os.remove(cls.SESSION_FILE)
        return True

    def log(self, message: str, level: str = "INFO", notebook: Optional[str] = None):
        """
        Log a message to the session log.

        Parameters
        ----------
        message : str
            Message to log.
        level : str
            Log level (INFO, WARNING, ERROR, DEBUG).
        notebook : str, optional
            Notebook name (e.g., "01_explore") for notebook-specific logs.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"

        # Write to main log
        main_log = self.logs_dir / "pipeline.log"
        with open(main_log, "a") as f:
            f.write(log_entry)

        # Write to notebook-specific log if specified
        if notebook:
            notebook_log = self.logs_dir / f"{notebook}.log"
            with open(notebook_log, "a") as f:
                f.write(log_entry)

    def save_metric(self, name: str, value: Any, notebook: Optional[str] = None):
        """
        Save a metric value.

        Parameters
        ----------
        name : str
            Metric name.
        value : any
            Metric value (must be JSON-serializable).
        notebook : str, optional
            Notebook name for organization.
        """
        if notebook:
            metric_file = self.metrics_dir / f"{notebook}_metrics.json"
        else:
            metric_file = self.metrics_dir / "metrics.json"

        # Load existing metrics
        if metric_file.exists():
            with open(metric_file, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        # Update metrics
        metrics[name] = value
        metrics["last_updated"] = datetime.now().isoformat()

        # Save metrics
        with open(metric_file, "w") as f:
            json.dump(metrics, f, indent=2)

    def get_plot_path(self, filename: str) -> Path:
        """
        Get path for saving a plot.

        Parameters
        ----------
        filename : str
            Plot filename.

        Returns
        -------
        Path
            Full path where plot should be saved.
        """
        return self.plots_dir / filename

    def get_metric_path(self, filename: str) -> Path:
        """
        Get path for saving a metric file.

        Parameters
        ----------
        filename : str
            Metric filename.

        Returns
        -------
        Path
            Full path where metric should be saved.
        """
        return self.metrics_dir / filename
