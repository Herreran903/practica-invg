# -*- coding: utf-8 -*-
"""
JSSP Tensor-Based Solver Selection Training Module.

This module trains CNN models on 3D tensor representations (JOBS x MACHINES x 2)
of Job Shop Scheduling Problem instances for solver selection.

Main Components:
    - config_loader: Configuration management
    - data_utils: Tensor loading and preprocessing
    - model_builder: CNN architecture for 10x10x2 inputs
    - training_loop: K-Fold cross-validation
    - evaluation: Metrics computation
    - visualization: Result plotting
    - cli: Command-line interface

Usage:
    # As CLI
    python -m src.training.jssp_tensors.cli --csv data.csv --task classification

    # As module
    from src.training.jssp_tensors import run_kfold, load_config
    config = load_config("config.yaml")
    results, summary = run_kfold(df, task, solver_cols, use_score, config, outdir)
"""

from .config_loader import load_config, merge_cli_args, resolve_paths
from .data_utils import (
    bss_index,
    build_labels,
    detect_solver_cols,
    filter_valid_images,
    load_tensor_npy,
    make_dataset,
    normalize_image_paths,
)
from .evaluation import (
    aggregate_fold_metrics,
    evaluate_classification,
    evaluate_fold,
    evaluate_multilabel,
    evaluate_regression,
)
from .model_builder import (
    build_cnn,
    build_model_from_config,
    get_model_summary,
)
from .training_loop import (
    compute_bss_baseline,
    run_kfold,
    train_fold,
)
from .visualization import (
    plot_class_bars,
    plot_confusion_matrix,
    plot_f1_bars_multilabel,
    plot_metrics_per_fold,
    plot_pr_curves_multilabel,
    plot_regression_scatter,
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
    "load_tensor_npy",
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
