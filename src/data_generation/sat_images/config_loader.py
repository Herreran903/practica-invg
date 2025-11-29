"""
Configuration loader for SAT Images data generation.

This module provides utilities to load and validate configuration from YAML files,
making it easy to manage all parameters centrally without hardcoding values.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class SATImagesConfig:
    """
    Configuration container for SAT Images data generation.

    Loads settings from a YAML file and provides convenient access to all
    configuration parameters needed for the ASlib data generation pipeline.
    """

    def __init__(self, config_path: str):
        """
        Initialize configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file (relative to project root)
        """
        self.project_root = self._find_project_root()
        self.config_path = os.path.join(self.project_root, config_path)
        self.config = self._load_config()

    def _find_project_root(self) -> str:
        """
        Find the project root directory by looking for key marker files.

        Returns:
            Absolute path to the project root directory
        """
        current = Path(__file__).resolve().parent

        # Navigate up until we find the project root (contains requirements.txt, pyproject.toml, or .git)
        while current != current.parent:
            if (
                (current / "requirements.txt").exists()
                or (current / "pyproject.toml").exists()
                or (current / ".git").exists()
            ):
                return str(current)
            current = current.parent

        # Fallback: assume we're in src/data_generation/sat_images/ -> go up to repo root
        return str(Path(__file__).resolve().parent.parent.parent.parent)

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dictionary containing all configuration parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Please ensure the config file exists at the specified location."
            )

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def get_absolute_path(self, relative_path: str) -> str:
        """
        Convert a relative path (from config) to an absolute path.

        Args:
            relative_path: Path relative to project root

        Returns:
            Absolute path
        """
        return os.path.join(self.project_root, relative_path)

    # Output directories
    @property
    def base_output_dir(self) -> str:
        """Get absolute path to base output directory."""
        return self.get_absolute_path(self.config["output"]["base_dir"])

    @property
    def default_output_dir(self) -> str:
        """Get absolute path to default output directory."""
        base = self.base_output_dir
        subdir = self.config["output"]["default_output_dir"]
        return os.path.join(base, subdir)

    # Output filenames
    @property
    def ground_truth_csv_name(self) -> str:
        """Get filename for ground truth CSV output."""
        return self.config["filenames"]["ground_truth_csv"]

    # Image parameters
    @property
    def image_target_size(self) -> int:
        """Get target size for generated images (width and height)."""
        return self.config["image"]["target_size"]

    # ASlib parameters
    @property
    def default_timeout_s(self) -> float:
        """Get default timeout in seconds if not found in description.txt."""
        return float(self.config["aslib"]["default_timeout_s"])

    @property
    def prefix_map(self) -> Dict[str, str]:
        """Get prefix mapping for resolving instance paths."""
        return self.config["aslib"]["prefix_map"]


def load_config(
    config_path: str = "src/data_generation/sat_images/config.yaml",
) -> SATImagesConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file relative to project root

    Returns:
        SATImagesConfig object with all settings loaded
    """
    return SATImagesConfig(config_path)
