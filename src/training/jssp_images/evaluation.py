# -*- coding: utf-8 -*-
"""
Evaluation metrics for JSSP image-based solver selection.

This module computes task-specific metrics:
- Classification: accuracy, F1-macro, per-class metrics
- Multilabel: F1-micro, F1 per label, Average Precision per label
- Regression: MAE (overall and per solver)
"""

import os
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    mean_absolute_error,
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
    f1_df.to_csv(
        os.path.join(fold_dir, f"fold{fold_idx}_f1_per_label.csv"), index=False
    )

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
            ap_df = pd.DataFrame(
                {"label": list(ap_per_label.keys()), "AP": list(ap_per_label.values())}
            )
            ap_df.to_csv(
                os.path.join(fold_dir, f"fold{fold_idx}_ap_per_label.csv"), index=False
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
        os.path.join(fold_dir, f"fold{fold_idx}_mae_per_solver.csv"), index=False
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
    val_df: pd.DataFrame | None = None,
    solver_runtime_cols: List[str] | None = None,
    config: dict | None = None,
) -> Dict[str, float]:
    """
    Evaluate a fold based on task type and generate visualizations.
 
    This is the main entry point for fold evaluation. It dispatches to
    task-specific evaluation functions and triggers visualization generation.
 
    For JSSP (images/tensors), this function also optionally computes a
    "resolved_rate" metric for classification and multilabel tasks when
    validation data and runtime columns are provided:
 
        resolved_rate = fraction of validation instances where at least one
        predicted solver finishes within the configured time limit.
 
    Args:
        y_true: True labels/values.
        y_pred: Predicted labels/values.
        task: Task type.
        solver_names: List of solver names.
        fold_dir: Output directory for this fold.
        fold_idx: Fold index.
        val_df: Optional validation DataFrame (required for resolved_rate).
        solver_runtime_cols: Optional list of runtime column names aligned
            with solver_names / prediction indices.
        config: Optional configuration dict, used to read time_limit_s.
 
    Returns:
        Dictionary with task-specific metrics (plus resolved_rate when available).
 
    Design Decision:
        Separates metric computation from visualization to maintain modularity.
        Visualization is triggered here but implemented in visualization.py.
    """
    from .visualization import (
        plot_class_bars,
        plot_confusion_matrix,
        plot_f1_bars_multilabel,
        plot_pr_curves_multilabel,
        plot_regression_scatter,
    )
 
    # Helper: compute time limit from config (fallback 60s)
    def _get_time_limit_s() -> float:
        if config is None:
            return 60.0
        data_cfg = config.get("data", {})
        # SAT config uses data.time_limit_s; JSSP images uses same key
        return float(data_cfg.get("time_limit_s", 60.0))
 
    # Compute metrics based on task
    if task == "classification":
        metrics = evaluate_classification(
            y_true, y_pred, solver_names, fold_dir, fold_idx
        )
 
        # Optional resolved_rate for JSSP: predicted solver runtime < time_limit_s
        if (
            val_df is not None
            and solver_runtime_cols is not None
            and len(val_df) == len(y_pred)
        ):
            time_limit = _get_time_limit_s()
            n = len(val_df)
            resolved = 0
 
            for i, cls_idx in enumerate(y_pred):
                idx = int(cls_idx)
                if idx < 0 or idx >= len(solver_runtime_cols):
                    continue
                runtime_col = solver_runtime_cols[idx]
                try:
                    rt = float(val_df.iloc[i][runtime_col])
                except Exception:
                    continue
 
                if np.isfinite(rt) and rt < time_limit:
                    resolved += 1
 
            metrics["resolved_rate"] = float(resolved / n) if n > 0 else 0.0
 
        # Generate visualizations
        plot_confusion_matrix(y_true, y_pred, solver_names, fold_dir, fold_idx)
        plot_class_bars(y_true, y_pred, solver_names, fold_dir, fold_idx)
 
    elif task == "multilabel":
        metrics = evaluate_multilabel(y_true, y_pred, solver_names, fold_dir, fold_idx)
 
        # Optional resolved_rate for JSSP multilabel:
        # instance is "resolved" if any predicted-1 solver has runtime < time_limit_s
        if (
            val_df is not None
            and solver_runtime_cols is not None
            and len(val_df) == y_pred.shape[0]
        ):
            time_limit = _get_time_limit_s()
            n = len(val_df)
            resolved = 0
 
            for i in range(n):
                row_pred = y_pred[i]
                # Treat >0.5 as active solver (works with {0,1} or probabilities)
                active_idxs = np.where(row_pred > 0.5)[0]
                if active_idxs.size == 0:
                    continue
 
                row = val_df.iloc[i]
                instance_resolved = False
                for idx in active_idxs:
                    if idx < 0 or idx >= len(solver_runtime_cols):
                        continue
                    runtime_col = solver_runtime_cols[int(idx)]
                    try:
                        rt = float(row[runtime_col])
                    except Exception:
                        continue
 
                    if np.isfinite(rt) and rt < time_limit:
                        instance_resolved = True
                        break
 
                if instance_resolved:
                    resolved += 1
 
            metrics["resolved_rate"] = float(resolved / n) if n > 0 else 0.0
 
        # Generate visualizations
        plot_pr_curves_multilabel(y_true, solver_names, fold_dir, fold_idx)
        plot_f1_bars_multilabel(y_true, y_pred, solver_names, fold_dir, fold_idx)
 
    else:  # regression
        metrics = evaluate_regression(y_true, y_pred, solver_names, fold_dir, fold_idx)
 
        # Generate visualizations
        plot_regression_scatter(y_true, y_pred, solver_names, fold_dir, fold_idx)
 
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
