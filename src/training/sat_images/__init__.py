# -*- coding: utf-8 -*-
"""
SAT Image-Based Solver Selection Training Module.

This module trains CNN models on grayscale image representations of SAT
problem instances for solver selection, with SAT-specific metrics.

Features:
- K-Fold cross-validation with optional repetitions (e.g., 5x5)
- SAT-specific metrics: resolved_rate, AST (Average Solving Time)
- Solver filtering
- Status column handling

Main Components:
    - config_loader: Configuration management
    - data_utils: Data loading and SAT-specific utilities
    - model_builder: CNN architecture (128x128x1)
    - training_loop: K-Fold with repetitions
    - evaluation: Metrics with resolved_rate and AST
    - visualization: Result plotting
    - cli: Command-line interface

Usage:
    # As CLI
    python -m src.training.sat_images.cli --csv data.csv --task classification --folds 5 --repeats 5

    # As module
    from src.training.sat_images import run_kfold_with_repeats, load_config
    config = load_config("config.yaml")
    results, summary = run_kfold_with_repeats(df, task, solver_cols, use_score, config, outdir)
"""

from .config_loader import load_config, merge_cli_args, resolve_paths
from .data_utils import (
    bss_index,
    build_labels,
    compute_ast_bss,
    compute_ast_classification,
    compute_ast_multilabel,
    compute_resolved_rate_classification,
    compute_resolved_rate_multilabel,
    detect_solver_cols,
    filter_valid_images,
    get_status_column,
    is_solver_ok,
    make_dataset,
    normalize_image_paths,
)
from .evaluation import (
    aggregate_fold_metrics,
    evaluate_classification_sat,
    evaluate_fold,
    evaluate_multilabel_sat,
    evaluate_regression,
)
from .model_builder import (
    build_cnn,
    build_model_from_config,
    get_model_summary,
)
from .training_loop import (
    run_kfold,
    run_kfold_with_repeats,
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
    "get_status_column",
    "is_solver_ok",
    "compute_resolved_rate_classification",
    "compute_resolved_rate_multilabel",
    "compute_ast_classification",
    "compute_ast_multilabel",
    "compute_ast_bss",
    # Model
    "build_cnn",
    "build_model_from_config",
    "get_model_summary",
    # Training
    "run_kfold",
    "run_kfold_with_repeats",
    # Evaluation
    "evaluate_fold",
    "evaluate_classification_sat",
    "evaluate_multilabel_sat",
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
__author__ = "SAT Training Module"
