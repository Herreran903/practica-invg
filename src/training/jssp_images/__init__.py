# -*- coding: utf-8 -*-
"""
JSSP Image-Based Solver Selection Training Module.

This module provides a complete pipeline for training CNN models to select
the best solver for Job Shop Scheduling Problem (JSSP) instances based on
grayscale image representations.

Main Components:
    - config_loader: Configuration management
    - data_utils: Data loading and preprocessing
    - model_builder: CNN architecture construction
    - training_loop: K-Fold cross-validation
    - evaluation: Metrics computation
    - visualization: Result plotting
    - cli: Command-line interface

Usage:
    # As a command-line tool
    python -m src.training.jssp_images.cli --csv data.csv --task classification

    # As a Python module
    from src.training.jssp_images import run_kfold, load_config
    config = load_config("config.yaml")
    results, summary = run_kfold(df, task, solver_cols, use_score, config, outdir)
"""

from .config_loader import load_config, merge_cli_args, resolve_paths
from .data_utils import (
    detect_solver_cols,
    build_labels,
    make_dataset,
    normalize_image_paths,
    filter_valid_images,
    bss_index,
    multilabel_targets,
)
from .model_builder import (
    build_cnn,
    build_model_from_config,
    get_model_summary,
)
from .training_loop import (
    train_fold,
    run_kfold,
    compute_bss_baseline,
)
from .evaluation import (
    evaluate_fold,
    evaluate_classification,
    evaluate_multilabel,
    evaluate_regression,
    aggregate_fold_metrics,
)
from .visualization import (
    plot_confusion_matrix,
    plot_class_bars,
    plot_pr_curves_multilabel,
    plot_f1_bars_multilabel,
    plot_regression_scatter,
    plot_metrics_per_fold,
    plot_training_history,
)

__all__ = [
    # Config
    "load_config",
    "merge_cli_args",
    "resolve_paths",
    # Data
    "detect_solver_cols",
    "build_labels",
    "make_dataset",
    "normalize_image_paths",
    "filter_valid_images",
    "bss_index",
    "multilabel_targets",
    # Model
    "build_cnn",
    "build_model_from_config",
    "get_model_summary",
    # Training
    "train_fold",
    "run_kfold",
    "compute_bss_baseline",
    # Evaluation
    "evaluate_fold",
    "evaluate_classification",
    "evaluate_multilabel",
    "evaluate_regression",
    "aggregate_fold_metrics",
    # Visualization
    "plot_confusion_matrix",
    "plot_class_bars",
    "plot_pr_curves_multilabel",
    "plot_f1_bars_multilabel",
    "plot_regression_scatter",
    "plot_metrics_per_fold",
    "plot_training_history",
]

__version__ = "1.0.0"
__author__ = "JSSP Training Module"