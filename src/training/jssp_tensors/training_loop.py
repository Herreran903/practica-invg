# -*- coding: utf-8 -*-
"""
Training loop for JSSP tensor-based solver selection.

This module reuses the training logic from jssp_images since the
training process is identical - only the input data format differs.
"""

# Reuse training loop from jssp_images
from ..jssp_images.training_loop import (
    compute_bss_baseline,
    run_kfold,
    train_fold,
)
from .data_utils import bss_index, build_labels, make_dataset

# Import local modules for proper resolution
from .model_builder import build_model_from_config

__all__ = [
    "train_fold",
    "compute_bss_baseline",
    "run_kfold",
]
