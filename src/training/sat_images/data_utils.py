# -*- coding: utf-8 -*-
"""
Data utilities for SAT image-based solver selection.

Extends jssp_images utilities with SAT-specific metrics:
- resolved_rate: Percentage of instances successfully resolved
- AST (Average Solving Time): Mean time including feature extraction
"""

from typing import Dict, List

import numpy as np
import pandas as pd

# Reuse base functions from jssp_images
from ..jssp_images.data_utils import (
    bss_index,
    build_labels,
    detect_solver_cols,
    filter_valid_images,
    make_dataset,
    normalize_image_paths,
    multilabel_targets,
)


def get_status_column(runtime_col: str) -> str:
    """Get status column name from runtime column name."""
    return runtime_col.replace("_Runtime_s", "_Status")


def is_solver_ok(
    row: pd.Series,
    runtime_col: str,
    time_limit: float,
    valid_statuses: List[str],
    invalid_statuses: List[str],
) -> bool:
    """
    Check if solver successfully resolved the instance.

    A solver is considered successful if:
    1. Status column indicates success (OK/SAT/UNSAT), OR
    2. Runtime is finite and below time limit

    Args:
        row: DataFrame row with solver data.
        runtime_col: Runtime column name.
        time_limit: Time limit in seconds.
        valid_statuses: List of valid status strings.
        invalid_statuses: List of invalid status strings.

    Returns:
        True if solver successfully resolved instance.
    """
    status_col = get_status_column(runtime_col)

    # Check status column if exists
    if status_col in row.index and pd.notna(row[status_col]):
        status = str(row[status_col]).strip().lower()
        if status in valid_statuses:
            return True
        if status in invalid_statuses:
            return False

    # Check runtime
    try:
        runtime = float(row.get(runtime_col, np.nan))
    except (ValueError, TypeError):
        runtime = np.nan

    return np.isfinite(runtime) and (runtime < time_limit)


def compute_resolved_rate_classification(
    val_df: pd.DataFrame,
    solver_runtime_cols: List[str],
    y_pred: np.ndarray,
    time_limit: float,
    valid_statuses: List[str],
    invalid_statuses: List[str],
) -> float:
    """
    Compute resolved rate for classification task.

    Args:
        val_df: Validation DataFrame.
        solver_runtime_cols: List of runtime column names.
        y_pred: Predicted class indices [N].
        time_limit: Time limit in seconds.
        valid_statuses: Valid status strings.
        invalid_statuses: Invalid status strings.

    Returns:
        Resolved rate (0.0 to 1.0).
    """
    ok_flags = []
    for i, cls_idx in enumerate(y_pred):
        try:
            runtime_col = solver_runtime_cols[int(cls_idx)]
        except (IndexError, ValueError):
            ok_flags.append(False)
            continue

        row = val_df.iloc[i]
        ok_flags.append(
            is_solver_ok(row, runtime_col, time_limit, valid_statuses, invalid_statuses)
        )

    return float(np.mean(ok_flags)) if ok_flags else 0.0


def compute_resolved_rate_multilabel(
    val_df: pd.DataFrame,
    solver_runtime_cols: List[str],
    y_pred_bin: np.ndarray,
    time_limit: float,
    valid_statuses: List[str],
    invalid_statuses: List[str],
    threshold: float = 0.5,
) -> float:
    """
    Compute resolved rate for multilabel task.

    Instance is resolved if ANY predicted solver succeeds.

    Args:
        val_df: Validation DataFrame.
        solver_runtime_cols: List of runtime column names.
        y_pred_bin: Predicted binary labels [N, C].
        time_limit: Time limit in seconds.
        valid_statuses: Valid status strings.
        invalid_statuses: Invalid status strings.
        threshold: Threshold for binary prediction.

    Returns:
        Resolved rate (0.0 to 1.0).
    """
    ok_flags = []
    n = y_pred_bin.shape[0]

    for i in range(n):
        row = val_df.iloc[i]
        pred_idxs = np.where(y_pred_bin[i] >= threshold)[0]

        if pred_idxs.size == 0:
            ok_flags.append(False)
            continue

        # Check if any predicted solver succeeds
        any_ok = False
        for j in pred_idxs:
            runtime_col = solver_runtime_cols[j]
            if is_solver_ok(
                row, runtime_col, time_limit, valid_statuses, invalid_statuses
            ):
                any_ok = True
                break

        ok_flags.append(any_ok)

    return float(np.mean(ok_flags)) if ok_flags else 0.0


def compute_ast_classification(
    val_df: pd.DataFrame,
    solver_runtime_cols: List[str],
    y_pred: np.ndarray,
    feat_time_col: str,
    time_limit: float,
) -> float:
    """
    Compute Average Solving Time (AST) for classification.

    AST = mean(feature_time + solver_time) across all instances.
    If solver fails or exceeds time limit, uses time_limit as penalty.

    Args:
        val_df: Validation DataFrame.
        solver_runtime_cols: List of runtime column names.
        y_pred: Predicted class indices [N].
        feat_time_col: Feature extraction time column name.
        time_limit: Time limit in seconds.

    Returns:
        Average solving time in seconds.
    """
    times = []

    for i, cls_idx in enumerate(y_pred):
        row = val_df.iloc[i]
        runtime_col = solver_runtime_cols[int(cls_idx)]

        try:
            runtime = float(row.get(runtime_col, np.inf))
        except (ValueError, TypeError):
            runtime = np.inf

        try:
            feat_time = float(row.get(feat_time_col, 0.0))
        except (ValueError, TypeError):
            feat_time = 0.0

        if not np.isfinite(runtime) or runtime >= time_limit:
            times.append(feat_time + time_limit)
        else:
            times.append(feat_time + runtime)

    return float(np.mean(times)) if times else 0.0


def compute_ast_multilabel(
    val_df: pd.DataFrame,
    solver_runtime_cols: List[str],
    y_pred_bin: np.ndarray,
    feat_time_col: str,
    time_limit: float,
    threshold: float = 0.5,
) -> float:
    """
    Compute Average Solving Time (AST) for multilabel.

    Uses minimum runtime among predicted solvers.

    Args:
        val_df: Validation DataFrame.
        solver_runtime_cols: List of runtime column names.
        y_pred_bin: Predicted binary labels [N, C].
        feat_time_col: Feature extraction time column name.
        time_limit: Time limit in seconds.
        threshold: Threshold for binary prediction.

    Returns:
        Average solving time in seconds.
    """
    times = []

    for i in range(y_pred_bin.shape[0]):
        row = val_df.iloc[i]
        pred_idxs = np.where(y_pred_bin[i] >= threshold)[0]

        try:
            feat_time = float(row.get(feat_time_col, 0.0))
        except (ValueError, TypeError):
            feat_time = 0.0

        if pred_idxs.size == 0:
            times.append(feat_time + time_limit)
            continue

        # Get minimum runtime among predicted solvers
        runtimes = []
        for j in pred_idxs:
            try:
                rt = float(row.get(solver_runtime_cols[j], np.inf))
            except (ValueError, TypeError):
                rt = np.inf
            runtimes.append(rt)

        best_runtime = np.nanmin(runtimes) if runtimes else np.inf

        if not np.isfinite(best_runtime) or best_runtime >= time_limit:
            times.append(feat_time + time_limit)
        else:
            times.append(feat_time + best_runtime)

    return float(np.mean(times)) if times else 0.0


def compute_ast_bss(
    val_df: pd.DataFrame,
    runtime_cols: List[str],
    bss_idx: int,
    feat_time_col: str,
    time_limit: float,
) -> float:
    """
    Compute AST for Baseline Single Solver (BSS).

    Args:
        val_df: Validation DataFrame.
        runtime_cols: List of runtime column names.
        bss_idx: Index of BSS solver.
        feat_time_col: Feature extraction time column name.
        time_limit: Time limit in seconds.

    Returns:
        Average solving time for BSS.
    """
    times = []

    for _, row in val_df.iterrows():
        try:
            runtime = float(row.get(runtime_cols[bss_idx], np.inf))
        except (ValueError, TypeError):
            runtime = np.inf

        try:
            feat_time = float(row.get(feat_time_col, 0.0))
        except (ValueError, TypeError):
            feat_time = 0.0

        if not np.isfinite(runtime) or runtime >= time_limit:
            times.append(feat_time + time_limit)
        else:
            times.append(feat_time + runtime)

    return float(np.mean(times)) if times else 0.0


__all__ = [
    # Base functions
    "detect_solver_cols",
    "bss_index",
    "normalize_image_paths",
    "filter_valid_images",
    "make_dataset",
    "build_labels",
    "multilabel_targets",
    # SAT-specific functions
    "get_status_column",
    "is_solver_ok",
    "compute_resolved_rate_classification",
    "compute_resolved_rate_multilabel",
    "compute_ast_classification",
    "compute_ast_multilabel",
    "compute_ast_bss",
]
