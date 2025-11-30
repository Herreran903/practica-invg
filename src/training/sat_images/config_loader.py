# -*- coding: utf-8 -*-
"""
Configuration loader for SAT image-based solver selection training.

Relaxed loader that does NOT require 'data.csv_path' to be present at file load time,
because the SAT CLI accepts --csv and merges it afterwards.

We still reuse merge/resolve helpers from the JSSP training config loader.
"""

import os
from typing import Any, Dict

import yaml

from ..jssp_images.config_loader import (
    get_default_config_path as _jssp_get_default_config_path,
    get_config_value,
    merge_cli_args,
    resolve_paths,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load SAT training configuration from YAML without enforcing 'data.csv_path'.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary (may not include data.csv_path yet;
        the CLI will inject it from --csv and then call resolve_paths)

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required top-level sections are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Ensure top-level sections exist (csv_path injected later by CLI)
    for key in ["experiment", "data", "model", "training", "output"]:
        config.setdefault(key, {})

    # Optional minimal validations
    if "target_height" not in config["data"]:
        config["data"].setdefault("target_height", 128)
    if "target_width" not in config["data"]:
        config["data"].setdefault("target_width", 128)

    return config


def get_default_config_path() -> str:
    """Return default config path co-located in this module."""
    return _jssp_get_default_config_path().replace("jssp_images", "sat_images")


__all__ = [
    "load_config",
    "merge_cli_args",
    "resolve_paths",
    "get_config_value",
    "get_default_config_path",
]
