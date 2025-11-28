# -*- coding: utf-8 -*-
"""
Model builder for SAT image-based solver selection.

Reuses CNN architecture from jssp_images (same 128x128x1 input).
"""

# Reuse model builder from jssp_images
from ..jssp_images.model_builder import (
    build_cnn,
    build_model_from_config,
    get_model_summary,
)

__all__ = [
    'build_cnn',
    'build_model_from_config',
    'get_model_summary',
]