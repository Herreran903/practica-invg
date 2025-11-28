"""
SAT Images Data Generation Module.

This module provides tools for generating SAT datasets with grayscale image
representations from ASlib scenarios. It processes algorithm performance data
and converts SAT instance files to images.
"""

from .aslib_parser import (
    build_pivot_runtime_table,
    compute_winner_key,
    load_algorithm_runs_dataframe,
    try_read_timeout_from_description
)
from .config_loader import SATImagesConfig, load_config
from .image_converter import convert_dataset_to_images, convert_raw_text_to_grayscale_image
from .instance_resolver import (
    build_instance_path_map,
    load_instance_map_csv,
    resolve_path_with_prefix_map,
    resolve_raw_text_path
)
from .prepare_aslib_dataset import prepare_aslib_dataset, prepare_aslib_dataset_with_config

__all__ = [
    # Configuration
    "SATImagesConfig",
    "load_config",
    # ASlib parsing
    "load_algorithm_runs_dataframe",
    "try_read_timeout_from_description",
    "build_pivot_runtime_table",
    "compute_winner_key",
    # Instance resolution
    "build_instance_path_map",
    "load_instance_map_csv",
    "resolve_raw_text_path",
    "resolve_path_with_prefix_map",
    # Dataset preparation
    "prepare_aslib_dataset",
    "prepare_aslib_dataset_with_config",
    # Image conversion
    "convert_raw_text_to_grayscale_image",
    "convert_dataset_to_images",
]

__version__ = "1.0.0"