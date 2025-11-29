# -*- coding: utf-8 -*-
"""
Configuration loader for JSSP tensor-based solver selection training.

This module handles loading, validation, and merging of configuration
from YAML files and command-line arguments.
"""

import argparse
import os
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary with configuration parameters.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Validate required sections
    required_sections = ["experiment", "data", "model", "training", "output"]
    for section in required_sections:
        if section not in config:
            config[section] = {}

    return config


def merge_cli_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge command-line arguments into configuration dictionary.

    CLI arguments take precedence over config file values.

    Args:
        config: Configuration dictionary from YAML.
        args: Parsed command-line arguments.

    Returns:
        Updated configuration dictionary.
    """
    # Training parameters
    if hasattr(args, "epochs") and args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if hasattr(args, "batch_size") and args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    if hasattr(args, "folds") and args.folds is not None:
        config["training"]["k_folds"] = args.folds

    if hasattr(args, "learning_rate") and args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    if hasattr(args, "seed") and args.seed is not None:
        config["training"]["seed"] = args.seed

    # Output parameters
    if hasattr(args, "out_parent") and args.out_parent is not None:
        config["output"]["parent_dir"] = args.out_parent

    if hasattr(args, "run_name") and args.run_name is not None:
        config["output"]["run_name"] = args.run_name

    return config


def resolve_paths(
    config: Dict[str, Any], base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration to absolute paths.

    Args:
        config: Configuration dictionary.
        base_dir: Base directory for resolving relative paths.
                 If None, uses current working directory.

    Returns:
        Configuration with resolved paths.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    # Resolve output parent directory
    if "output" in config and "parent_dir" in config["output"]:
        parent_dir = config["output"]["parent_dir"]
        if not os.path.isabs(parent_dir):
            config["output"]["parent_dir"] = os.path.normpath(
                os.path.join(base_dir, parent_dir)
            )

    return config


def get_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get nested configuration value.

    Args:
        config: Configuration dictionary.
        *keys: Sequence of keys to traverse (e.g., 'training', 'epochs').
        default: Default value if key path doesn't exist.

    Returns:
        Configuration value or default.

    Example:
        >>> epochs = get_config_value(config, 'training', 'epochs', default=25)
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
