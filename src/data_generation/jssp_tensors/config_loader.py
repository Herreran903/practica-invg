"""
Configuration loader for JSSP Tensors data generation.

This module provides utilities to load and validate configuration from YAML files,
making it easy to manage all parameters centrally without hardcoding values.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class JSPPTensorsConfig:
    """
    Configuration container for JSSP Tensors data generation.

    Loads settings from a YAML file and provides convenient access to all
    configuration parameters needed for the data generation pipeline.
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

        # Fallback: assume we're in src/data_generation/jssp_tensors/ -> go up to repo root
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

    # Model paths
    @property
    def cp_model_path(self) -> str:
        """Get absolute path to CP model (.mzn file)."""
        return self.get_absolute_path(self.config["models"]["cp_model"])

    @property
    def mip_model_path(self) -> Optional[str]:
        """Get absolute path to MIP model (.mzn file), or None if not configured."""
        mip_path = self.config["models"].get("mip_model")
        return self.get_absolute_path(mip_path) if mip_path else None

    # Output directories
    @property
    def base_output_dir(self) -> str:
        """Get absolute path to base output directory."""
        return self.get_absolute_path(self.config["output"]["base_dir"])

    @property
    def academic_output_dir(self) -> str:
        """Get absolute path to academic mode output directory."""
        base = self.base_output_dir
        subdir = self.config["output"]["academic_dir"]
        return os.path.join(base, subdir)

    @property
    def generated_output_dir(self) -> str:
        """Get absolute path to generated mode output directory."""
        base = self.base_output_dir
        subdir = self.config["output"]["generated_dir"]
        return os.path.join(base, subdir)

    # Output filenames
    @property
    def academic_csv_name(self) -> str:
        """Get filename for academic mode CSV output."""
        return self.config["filenames"]["academic_csv"]

    @property
    def generated_csv_name(self) -> str:
        """Get filename for generated mode CSV output."""
        return self.config["filenames"]["generated_csv"]

    # Tensor parameters
    @property
    def tensor_standardize(self) -> bool:
        """Get whether to apply z-score standardization to tensors."""
        return self.config["tensor"]["standardize"]

    # Academic mode configuration
    @property
    def academic_instances(self) -> List[str]:
        """Get list of JSPLIB instance names for academic mode."""
        return self.config["academic"]["instances"]

    @property
    def academic_time_limit_ms(self) -> int:
        """Get time limit in milliseconds for academic mode."""
        return self.config["academic"]["time_limit_ms"]

    @property
    def academic_penalty_factor(self) -> float:
        """Get penalty factor K for academic mode scoring."""
        return self.config["academic"]["penalty_factor_k"]

    @property
    def academic_solver_strategies(self) -> List[Tuple[str, str, str]]:
        """
        Get solver strategies for academic mode.

        Returns:
            List of tuples (solver_id, strategy_label, key_identifier)
        """
        strategies = self.config["academic"]["solver_strategies"]
        return [(s["solver"], s["strategy"], s["key"]) for s in strategies]

    # Generated mode configuration
    @property
    def generated_time_limits_ms(self) -> List[int]:
        """Get list of time limits in milliseconds for generated mode benchmarking."""
        return self.config["generated"]["time_limits_ms"]

    @property
    def generated_random_seeds(self) -> List[int]:
        """Get list of random seeds for generated mode."""
        return self.config["generated"]["random_seeds"]

    @property
    def generated_cases(self) -> List[Tuple[int, int, int]]:
        """
        Get generation cases for generated mode.

        Returns:
            List of tuples (num_jobs, num_machines, num_instances)
        """
        return [tuple(case) for case in self.config["generated"]["generation_cases"]]

    @property
    def instance_duration_range(self) -> Tuple[int, int]:
        """Get duration range for instance generation (min, max)."""
        range_list = self.config["generated"]["instance_generation"]["duration_range"]
        return tuple(range_list)

    @property
    def instance_generation_seed(self) -> int:
        """Get seed for instance generator."""
        return self.config["generated"]["instance_generation"]["seed"]

    # Solver candidates
    @property
    def solver_candidates(self) -> List[Tuple[str, str, str, Dict[str, Any]]]:
        """
        Get solver candidates for generated mode.

        Returns:
            List of tuples (solver_id, key_identifier, type, options_dict)
        """
        candidates = self.config["solver_candidates"]
        return [(c["solver_id"], c["key"], c["type"], c["options"]) for c in candidates]


def load_config(
    config_path: str = "src/data_generation/jssp_tensors/config.yaml",
) -> JSPPTensorsConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file relative to project root

    Returns:
        JSPPTensorsConfig object with all settings loaded
    """
    return JSPPTensorsConfig(config_path)
