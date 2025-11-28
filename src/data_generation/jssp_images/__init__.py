"""
JSSP Images Data Generation Module.

This module provides tools for generating JSSP datasets with grayscale image
representations. It supports both academic (JSPLIB benchmarks) and generated
(random instances) modes.
"""

from .config_loader import JSPPImagesConfig, load_config
from .image_converter import convert_dataset_to_images, convert_text_to_grayscale_image
from .jssp_instance_utils import (
    generate_random_instance,
    get_instance_optimum,
    load_academic_instance,
    save_instance_as_dzn
)
from .prepare_academic_dataset import prepare_academic_dataset
from .prepare_generated_dataset import prepare_generated_dataset

__all__ = [
    # Configuration
    "JSPPImagesConfig",
    "load_config",
    # Instance utilities
    "load_academic_instance",
    "generate_random_instance",
    "save_instance_as_dzn",
    "get_instance_optimum",
    # Dataset preparation
    "prepare_academic_dataset",
    "prepare_generated_dataset",
    # Image conversion
    "convert_text_to_grayscale_image",
    "convert_dataset_to_images",
]

__version__ = "1.0.0"