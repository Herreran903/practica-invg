# -*- coding: utf-8 -*-
"""
Training loop for JSSP tensor-based solver selection.

This module reuses the training logic from jssp_images since the
training process is identical - only the input data format differs.
"""

# Reuse training loop from jssp_images
from ..jssp_images.training_loop import (
    train_fold,
    compute_bss_baseline,
    run_kfold,
)

# Import local modules for proper resolution
from .model_builder import build_model_from_config
from .data_utils import build_labels, make_dataset, bss_index

__all__ = [
    'train_fold',
    'compute_bss_baseline',
    'run_kfold',
]