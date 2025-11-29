# -*- coding: utf-8 -*-
"""
Evaluation metrics for JSSP tensor-based solver selection.

Reuses evaluation logic from jssp_images since metrics computation
is identical regardless of input format.
"""

# Reuse evaluation functions from jssp_images
from ..jssp_images.evaluation import (
    aggregate_fold_metrics,
    evaluate_classification,
    evaluate_fold,
    evaluate_multilabel,
    evaluate_regression,
)

__all__ = [
    "evaluate_fold",
    "evaluate_classification",
    "evaluate_multilabel",
    "evaluate_regression",
    "aggregate_fold_metrics",
]
