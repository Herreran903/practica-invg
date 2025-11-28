"""
JSSP Tensors Data Generation Module.

This module provides tools for generating JSSP datasets with 2D tensor/matrix
representations. It supports both academic (JSPLIB benchmarks) and generated
(random instances) modes.
"""

from .config_loader import JSPPTensorsConfig, load_config
from .prepare_academic_dataset import prepare_academic_dataset
from .prepare_generated_dataset import prepare_generated_dataset
from .tensor_converter import convert_dataset_to_tensors, convert_dzn_to_tensor

__all__ = [
    # Configuration
    "JSPPTensorsConfig",
    "load_config",
    # Dataset preparation
    "prepare_academic_dataset",
    "prepare_generated_dataset",
    # Tensor conversion
    "convert_dzn_to_tensor",
    "convert_dataset_to_tensors",
]

__version__ = "1.0.0"