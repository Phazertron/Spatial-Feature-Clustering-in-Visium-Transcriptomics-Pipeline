"""
Configuration management for the spatial clustering pipeline.

Handles loading and validation of configuration profiles.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import warnings


DEFAULT_CONFIG = {
    "dataset_path": "data/DLPFC-151673",
    "clustering": {
        "method": "louvain",
        "resolution": 1.0,
        "random_state": 0,
        "optimize_resolution": False,
        "resolution_range": [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    },
    "similarity": {
        "weights": {
            "alpha": 0.5,  # Expression weight
            "beta": 0.3,   # Spatial weight
            "gamma": 0.2   # MoG weight
        }
    },
    "preprocessing": {
        "n_top_genes": 300,
        "min_gene_expression": 300,
        "n_top_genes_hvg": 3000,
        "pca_components": 50
    },
    "evaluation": {
        "n_neighbors": 6,
        "use_pca_for_coherence": False
    }
}


def load_config(profile: str = "default", config_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration for a specific profile.

    Parameters
    ----------
    profile : str
        Configuration profile to load ("default" or "optimized").
    config_file : Path, optional
        Path to config file. If None, uses "config.yaml" in project root.

    Returns
    -------
    dict
        Configuration dictionary.

    Examples
    --------
    >>> from src.utils.config import load_config
    >>> config = load_config("default")
    >>> config["clustering"]["resolution"]
    1.0
    """
    if config_file is None:
        config_file = Path("config.yaml")

    # If config file doesn't exist, use defaults
    if not config_file.exists():
        warnings.warn(
            f"Config file {config_file} not found. Using default configuration."
        )
        return DEFAULT_CONFIG.copy()

    # Load YAML config
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Get profile-specific config
    if "profiles" in config_data:
        if profile not in config_data["profiles"]:
            warnings.warn(
                f"Profile '{profile}' not found in config. Using 'default'."
            )
            profile = "default"

        profile_config = config_data["profiles"][profile]
    else:
        # No profiles defined, use entire config as default
        profile_config = config_data

    # Merge with defaults (fill in missing values)
    config = _merge_configs(DEFAULT_CONFIG, profile_config)

    return config


def _merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override config into default config.

    Parameters
    ----------
    default : dict
        Default configuration.
    override : dict
        Override values.

    Returns
    -------
    dict
        Merged configuration.
    """
    result = default.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Dict[str, Any], output_file: Path):
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : dict
        Configuration to save.
    output_file : Path
        Output file path.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_param(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Get a parameter from config using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    path : str
        Parameter path (e.g., "clustering.resolution").
    default : any
        Default value if path doesn't exist.

    Returns
    -------
    any
        Parameter value.

    Examples
    --------
    >>> config = {"clustering": {"resolution": 1.5}}
    >>> get_param(config, "clustering.resolution")
    1.5
    >>> get_param(config, "missing.key", default=0)
    0
    """
    keys = path.split(".")
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
