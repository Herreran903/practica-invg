# -*- coding: utf-8 -*-
"""
Evaluation metrics for SAT image-based solver selection.

Extends jssp_images evaluation with SAT-specific metrics:
- resolved_rate
- AST (Average Solving Time)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Literal

# Reuse base evaluation from jssp_images
from ..jssp_images.evaluation import (
    evaluate_classification as _evaluate_classification_base,
    evaluate_multilabel as _evaluate_multilabel_base,
    evaluate_regression,
    aggregate_fold_metrics,
)

# Import SAT-specific functions
from .data_utils import (
    compute_resolved_rate_classification,
    compute_resolved_rate_multilabel,
    compute_ast_classification,
    compute_ast_multilabel,
    get_status_column,
    is_solver_ok,
)


def evaluate_classification_sat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    solver_names: List[str],
    fold_dir: str,
    fold_idx: int,
    val_df: pd.DataFrame,
    solver_runtime_cols: List[str],
    config: dict,
) -> Dict[str, float]:
    """
    Evaluate classification with SAT-specific metrics.
    
    Computes standard metrics plus resolved_rate and AST.
    """
    # Base metrics
    metrics = _evaluate_classification_base(
        y_true, y_pred, solver_names, fold_dir, fold_idx
    )
    
    # SAT-specific metrics
    sat_cfg = config.get('sat', {})
    data_cfg = config.get('data', {})
    
    time_limit = data_cfg.get('time_limit_s', 1800.0)
    feat_time_col = data_cfg.get('feat_time_column', '_feat_time_zero_')
    valid_statuses = [s.lower() for s in sat_cfg.get('valid_statuses', ['ok', 'sat', 'unsat'])]
    invalid_statuses = [s.lower() for s in sat_cfg.get('invalid_statuses', 
                        ['timeout', 'time_out', 'timedout', 'memout', 'crash', 'error', 'fail'])]
    
    # Resolved rate
    resolved_rate = compute_resolved_rate_classification(
        val_df, solver_runtime_cols, y_pred, time_limit, valid_statuses, invalid_statuses
    )
    metrics['resolved_rate'] = float(resolved_rate)
    
    # AST
    ast_sec = compute_ast_classification(
        val_df, solver_runtime_cols, y_pred, feat_time_col, time_limit
    )
    metrics['AST_sec'] = float(ast_sec)
    
    # Save detailed resolution info
    detail_rows = []
    for i, cls_idx in enumerate(y_pred):
        runtime_col = solver_runtime_cols[int(cls_idx)]
        status_col = get_status_column(runtime_col)
        row = val_df.iloc[i]
        
        detail_rows.append({
            'Image_Npy_Path': row.get('Image_Npy_Path', ''),
            'pred_solver': solver_names[int(cls_idx)],
            'pred_runtime': row.get(runtime_col, np.nan),
            'pred_status': row.get(status_col, np.nan) if status_col in val_df.columns else np.nan,
            'resolved_ok': is_solver_ok(row, runtime_col, time_limit, valid_statuses, invalid_statuses),
            'feat_time_s': row.get(feat_time_col, np.nan),
        })
    
    pd.DataFrame(detail_rows).to_csv(
        os.path.join(fold_dir, f"fold{fold_idx}_resolved_detail.csv"),
        index=False
    )
    
    return metrics


def evaluate_multilabel_sat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    solver_names: List[str],
    fold_dir: str,
    fold_idx: int,
    val_df: pd.DataFrame,
    solver_runtime_cols: List[str],
    config: dict,
) -> Dict[str, float]:
    """
    Evaluate multilabel with SAT-specific metrics.
    """
    # Base metrics
    metrics = _evaluate_multilabel_base(
        y_true, y_pred, solver_names, fold_dir, fold_idx
    )
    
    # SAT-specific metrics
    sat_cfg = config.get('sat', {})
    data_cfg = config.get('data', {})
    tasks_cfg = config.get('tasks', {}).get('multilabel', {})
    
    time_limit = data_cfg.get('time_limit_s', 1800.0)
    feat_time_col = data_cfg.get('feat_time_column', '_feat_time_zero_')
    threshold = tasks_cfg.get('threshold', 0.5)
    valid_statuses = [s.lower() for s in sat_cfg.get('valid_statuses', ['ok', 'sat', 'unsat'])]
    invalid_statuses = [s.lower() for s in sat_cfg.get('invalid_statuses',
                        ['timeout', 'time_out', 'timedout', 'memout', 'crash', 'error', 'fail'])]
    
    # Resolved rate
    resolved_rate = compute_resolved_rate_multilabel(
        val_df, solver_runtime_cols, y_pred, time_limit, valid_statuses, invalid_statuses, threshold
    )
    metrics['resolved_rate'] = float(resolved_rate)
    
    # AST
    ast_sec = compute_ast_multilabel(
        val_df, solver_runtime_cols, y_pred, feat_time_col, time_limit, threshold
    )
    metrics['AST_sec'] = float(ast_sec)
    
    return metrics


def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: Literal["classification", "multilabel", "regression"],
    solver_names: List[str],
    fold_dir: str,
    fold_idx: int,
    val_df: pd.DataFrame = None,
    solver_runtime_cols: List[str] = None,
    config: dict = None,
) -> Dict[str, float]:
    """
    Evaluate fold with task-specific metrics.
    
    For SAT, includes resolved_rate and AST for classification/multilabel.
    """
    from ..jssp_images.visualization import (
        plot_confusion_matrix,
        plot_class_bars,
        plot_pr_curves_multilabel,
        plot_f1_bars_multilabel,
        plot_regression_scatter,
    )
    
    if task == "classification":
        if val_df is not None and solver_runtime_cols is not None and config is not None:
            metrics = evaluate_classification_sat(
                y_true, y_pred, solver_names, fold_dir, fold_idx,
                val_df, solver_runtime_cols, config
            )
        else:
            metrics = _evaluate_classification_base(
                y_true, y_pred, solver_names, fold_dir, fold_idx
            )
        
        plot_confusion_matrix(y_true, y_pred, solver_names, fold_dir, fold_idx)
        plot_class_bars(y_true, y_pred, solver_names, fold_dir, fold_idx)
        
    elif task == "multilabel":
        if val_df is not None and solver_runtime_cols is not None and config is not None:
            metrics = evaluate_multilabel_sat(
                y_true, y_pred, solver_names, fold_dir, fold_idx,
                val_df, solver_runtime_cols, config
            )
        else:
            metrics = _evaluate_multilabel_base(
                y_true, y_pred, solver_names, fold_dir, fold_idx
            )
        
        plot_pr_curves_multilabel(y_true, solver_names, fold_dir, fold_idx)
        plot_f1_bars_multilabel(y_true, y_pred, solver_names, fold_dir, fold_idx)
        
    else:  # regression
        metrics = evaluate_regression(
            y_true, y_pred, solver_names, fold_dir, fold_idx
        )
        plot_regression_scatter(y_true, y_pred, solver_names, fold_dir, fold_idx)
    
    return metrics


__all__ = [
    'evaluate_fold',
    'evaluate_classification_sat',
    'evaluate_multilabel_sat',
    'evaluate_regression',
    'aggregate_fold_metrics',
]