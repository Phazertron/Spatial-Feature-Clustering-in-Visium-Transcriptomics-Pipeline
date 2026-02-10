"""
Logging utilities for the spatial clustering pipeline.

Provides consistent logging across all modules and notebooks.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "spatial_clustering",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : Path, optional
        Path to log file. If None, only logs to console.
    level : int
        Logging level (logging.DEBUG, logging.INFO, etc.).
    format_string : str, optional
        Custom format string. If None, uses default format.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> from src.utils.logger import setup_logger
    >>> logger = setup_logger("my_analysis")
    >>> logger.info("Analysis started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler with UTF-8 encoding for Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    # Force UTF-8 encoding on Windows to handle emojis
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass  # Ignore if reconfigure fails
    logger.addHandler(console_handler)

    # File handler (if specified) with UTF-8 encoding
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
