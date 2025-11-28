# -*- coding: utf-8 -*-
"""
Evaluation metrics for JSSP image-based solver selection.

This module computes task-specific metrics:
- Classification: accuracy, F1-macro, per-class metrics
- Multilabel: F1-micro, F1 per label, Average Precision per label
- Regression: MAE (overall and per solver)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Literal
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_absolute_error,
    average_precision_score,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    solver_names: List[str],
    fold_dir: str,
    fold_idx: int,
) -> Dict[str, float]:
    """
    Compute classification metrics and save detailed report.

    Metrics computed:
    - Overall accuracy
    - Macro F1-score
    - Per-class precision, recall, F1

    Args:
        y_true: True class labels [N].
        y_pred: Predicted class labels [N].
        solver_names: List of solver names (class labels).
        fold_dir: Output directory for this fold.
        fold_idx: Fold index.

    Returns:
        Dictionary with 'accuracy' and 'f1_macro' keys.
    """
    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Detailed classification report
    labels_full = list(range(len(solver_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_full,
        target_names=solver_names,
        output_dict=True,
        zero_division=0,
    )
    
    # Save report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(fold_dir, f"fold{fold_idx}_cls_report.csv"))
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
    }


def evaluate_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    solver_names: List[str],
    fold_dir: str,
    fold_idx: int,
) -> Dict[str, float]:
    """
    Compute multilabel metrics and save per-label statistics.

    Metrics computed:
    - Micro F1-score (overall)
    - F1-score per label
    - Average Precision per label (if scores available)

    Args:
        y_true: True binary labels [N, C].
        y_pred: Predicted binary labels [N, C].
        solver_names: List of solver names.
        fold_dir: Output directory for this fold.
        fold_idx: Fold index.

    Returns:
        Dictionary with 'f1_micro' and 'f1_macro' keys.
    """
    # Overall micro F1
    f1_micro = f1_score(
        y_true.flatten(), y_pred.flatten(), average="micro", zero_division=0
    )
    
    # Per-label F1
    f1_per_label = []
    for j in range(y_true.shape[1]):
        f1_j = f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
        f1_per_label.append(f1_j)
    
    f1_macro = float(np.mean(f1_per_label))
    
    # Save per-label F1
    f1_df = pd.DataFrame({"label": solver_names, "F1": f1_per_label})
    f1_df.to_csv(os.path.join(fold_dir, f"fold{fold_idx}_f1_per_label.csv"), index=False)
    
    # Try to load scores for AP computation
    scores_path = os.path.join(fold_dir, f"fold{fold_idx}_y_scores.npy")
    if os.path.exists(scores_path):
        y_scores = np.load(scores_path)
        ap_per_label = {}
        
        for j, name in enumerate(solver_names):
            yt = y_true[:, j]
            ys = y_scores[:, j]
            
            # Skip if only one class present
            if np.unique(yt).size < 2:
                continue
            
            ap = average_precision_score(yt, ys)
            ap_per_label[name] = ap
        
        # Save AP per label
        if ap_per_label:
            ap_df = pd.DataFrame({
                "label": list(ap_per_label.keys()),
                "AP": list(ap_per_label.values())
            })
            ap_df.to_csv(
                os.path.join(fold_dir, f"fold{fold_idx}_ap_per_label.csv"),
                index=False
            )
    
    return {
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    solver_names: List[str],
    fold_dir: str,
    fold_idx: int,
) -> Dict[str, float]:
    """
    Compute regression metrics and save per-solver statistics.

    Metrics computed:
    - Overall MAE (mean across all solvers)
    - MAE per solver

    Args:
        y_true: True runtime values [N, C].
        y_pred: Predicted runtime values [N, C].
        solver_names: List of solver names.
        fold_dir: Output directory for this fold.
        fold_idx: Fold index.

    Returns:
        Dictionary with 'mae' key.
    """
    # Overall MAE
    mae_overall = mean_absolute_error(y_true, y_pred)
    
    # Per-solver MAE
    mae_per_solver = []
    for j in range(y_true.shape[1]):
        mae_j = mean_absolute_error(y_true[:, j], y_pred[:, j])
        mae_per_solver.append(mae_j)
    
    # Save per-solver MAE
    mae_df = pd.DataFrame({"solver": solver_names, "MAE": mae_per_solver})
    mae_df.to_csv(
        os.path.join(fold_dir, f"fold{fold_idx}_mae_per_solver.csv"),
        index=False
    )
    
    return {
        "mae": float(mae_overall),
    }


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: Literal["classification", "multilabel", "regression"],
    solver_names: List[str],
    fold_dir: str,
    fold_idx: int,
) -> Dict[str, float]:
    """
    Evaluate a fold based on task type and generate visualizations.

    This is the main entry point for fold evaluation. It dispatches to
    task-specific evaluation functions and triggers visualization generation.

    Args:
        y_true: True labels/values.
        y_pred: Predicted labels/values.
        task: Task type.
        solver_names: List of solver names.
        fold_dir: Output directory for this fold.
        fold_idx: Fold index.

    Returns:
        Dictionary with task-specific metrics.

    Design Decision:
        Separates metric computation from visualization to maintain modularity.
        Visualization is triggered here but implemented in visualization.py.
    """
    from .visualization import (
        plot_confusion_matrix,
        plot_class_bars,
        plot_pr_curves_multilabel,
        plot_f1_bars_multilabel,
        plot_regression_scatter,
    )
    
    # Compute metrics based on task
    if task == "classification":
        metrics = evaluate_classification(
            y_true, y_pred, solver_names, fold_dir, fold_idx
        )
        
        # Generate visualizations
        plot_confusion_matrix(
            y_true, y_pred, solver_names, fold_dir, fold_idx
        )
        plot_class_bars(
            y_true, y_pred, solver_names, fold_dir, fold_idx
        )
        
    elif task == "multilabel":
        metrics = evaluate_multilabel(
            y_true, y_pred, solver_names, fold_dir, fold_idx
        )
        
        # Generate visualizations
        plot_pr_curves_multilabel(
            y_true, solver_names, fold_dir, fold_idx
        )
        plot_f1_bars_multilabel(
            y_true, y_pred, solver_names, fold_dir, fold_idx
        )
        
    else:  # regression
        metrics = evaluate_regression(
            y_true, y_pred, solver_names, fold_dir, fold_idx
        )
        
        # Generate visualizations
        plot_regression_scatter(
            y_true, y_pred, solver_names, fold_dir, fold_idx
        )
    
    return metrics


def aggregate_fold_metrics(
    fold_results: List[Dict],
    metric_key: str,
) -> Dict[str, float]:
    """
    Aggregate metrics across folds.

    Args:
        fold_results: List of dictionaries with per-fold metrics.
        metric_key: Key of the metric to aggregate (e.g., 'accuracy', 'f1_micro').

    Returns:
        Dictionary with 'mean', 'std', 'min', 'max' statistics.
    """
    values = [r[metric_key] for r in fold_results if metric_key in r]
    
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }