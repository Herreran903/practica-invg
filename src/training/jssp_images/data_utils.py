"""
Data utilities for JSSP Images CNN Training.

This module handles data loading, preprocessing, and label construction.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def detect_solver_cols(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect solver columns in the dataframe.

    Searches for columns ending with '_Runtime_s' and '_Score_S_rel'.

    Args:
        df: DataFrame with solver performance columns

    Returns:
        Dictionary with 'runtime' and 'score' lists of column names

    Raises:
        ValueError: If no runtime columns are found
    """
    runtime_cols = sorted([c for c in df.columns if c.endswith("_Runtime_s")])
    score_cols = sorted([c for c in df.columns if c.endswith("_Score_S_rel")])

    if not runtime_cols:
        raise ValueError("No columns ending with '_Runtime_s' found in CSV.")

    return {"runtime": runtime_cols, "score": score_cols}


def argmin_runtime_or_score(
    row: pd.Series, solver_cols: Dict[str, List[str]], use_score: bool = False
) -> Tuple[int, List[str]]:
    """
    Get the index of the best solver (minimum value) for a row.

    Args:
        row: DataFrame row with solver metrics
        solver_cols: Dictionary with 'runtime' and 'score' column lists
        use_score: If True and score columns exist, use them instead of runtime

    Returns:
        Tuple of (best_solver_index, columns_used)
    """
    cols = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    vals = row[cols].astype(float).values
    idx = int(np.nanargmin(vals))
    return idx, cols


def bss_index(
    train_df: pd.DataFrame, solver_cols: Dict[str, List[str]], use_score: bool = False
) -> int:
    """
    Calculate the Baseline Single Solver (BSS) index.

    Selects the solver with the lowest mean runtime (or score) in the training set.

    Args:
        train_df: Training partition DataFrame
        solver_cols: Dictionary with 'runtime' and 'score' column lists
        use_score: If True and score columns exist, use them instead of runtime

    Returns:
        Index of the BSS column
    """
    cols = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    means = train_df[cols].astype(float).mean(axis=0).values
    return int(np.nanargmin(means))


def multilabel_targets(
    row: pd.Series, runtime_cols: List[str], time_limit_s: float
) -> np.ndarray:
    """
    Construct binary vector of viable solvers (runtime < time_limit).

    Args:
        row: DataFrame row with runtime columns
        runtime_cols: List of runtime column names
        time_limit_s: Time limit threshold in seconds

    Returns:
        Binary array of shape [num_solvers] where 1 indicates viable solver
    """
    vals = row[runtime_cols].astype(float).values
    mask = np.isfinite(vals)
    out = np.zeros_like(vals, dtype=np.float32)
    out[mask] = (vals[mask] < time_limit_s).astype(np.float32)
    return out


def build_labels(
    df: pd.DataFrame,
    solver_cols: Dict[str, List[str]],
    task: str,
    use_score: bool,
    time_limit_s: float,
) -> np.ndarray:
    """
    Generate label matrix/vector according to the task.

    Args:
        df: DataFrame with solver metrics
        solver_cols: Dictionary with 'runtime' and 'score' column lists
        task: One of 'classification', 'multilabel', or 'regression'
        use_score: Whether to use score columns for "best" solver
        time_limit_s: Time limit for multilabel/regression tasks

    Returns:
        Label array with shape [N] for classification or [N, C] for multi/regression

    Raises:
        ValueError: If task is not recognized
    """
    if task == "classification":
        y = []
        for _, r in df.iterrows():
            idx, _ = argmin_runtime_or_score(r, solver_cols, use_score)
            y.append(idx)
        y = np.array(y, dtype=np.int32)

    elif task == "multilabel":
        rt_cols = solver_cols["runtime"]
        y = np.stack(
            [multilabel_targets(r, rt_cols, time_limit_s) for _, r in df.iterrows()],
            axis=0,
        )

    elif task == "regression":
        rt_cols = solver_cols["runtime"]
        y = df[rt_cols].astype(float).values
        # Impute NaN with large penalty (10x time limit)
        y[~np.isfinite(y)] = time_limit_s * 10.0

    else:
        raise ValueError(f"Unknown task: {task}")

    return y


def load_npy_image(path: bytes) -> np.ndarray:
    """
    Load a .npy image file and normalize to [H, W, 1] format.

    Args:
        path: Path to .npy file (as bytes from tf.numpy_function)

    Returns:
        Image array of shape [H, W, 1] in float32
    """
    arr = np.load(path.decode("utf-8")).astype(np.float32)

    # Ensure single channel
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[-1] != 1:
        arr = arr[..., :1]

    return arr


def make_dataset(
    paths: List[str],
    labels: np.ndarray,
    task: str,
    batch_size: int,
    shuffle: bool,
    target_h: int | None = None,
    target_w: int | None = None,
    seed: int = 42,
    config: dict | None = None,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset for training/validation.

    This function is used by both JSSP and SAT image training pipelines.
    To keep backward compatibility:

    - If ``target_h``/``target_w`` are provided, they take precedence.
    - Otherwise, if ``config`` is provided, values are read from
      ``config["data"]["image"]["target_height"/"target_width"]``.
    - As a last resort, defaults of 128×128 are used.

    Args:
        paths: List of paths to .npy image files
        labels: Label array
        task: One of 'classification', 'multilabel', or 'regression'
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        target_h: Target image height (optional; may be derived from config)
        target_w: Target image width (optional; may be derived from config)
        seed: Random seed for shuffling
        config: Optional configuration dict (from training YAML)

    Returns:
        TensorFlow dataset ready for training
    """
    # Derive target height/width if not explicitly provided
    if target_h is None or target_w is None:
        if config is not None:
            # Support both schemas:
            # - JSSP:   data.image.target_height / target_width
            # - SAT:    data.target_height / target_width
            data_cfg = config.get("data", {})
            image_cfg = data_cfg.get("image", {}) or {}
            target_h = image_cfg.get(
                "target_height", data_cfg.get("target_height", 128)
            )
            target_w = image_cfg.get(
                "target_width", data_cfg.get("target_width", 128)
            )
        else:
            target_h = target_h or 128
            target_w = target_w or 128

    x = tf.constant(paths)
    y = tf.constant(labels)

    def _map_fn(p, t):
        img = tf.numpy_function(load_npy_image, [p], tf.float32)
        img.set_shape([target_h, target_w, 1])

        # Per-image standardization to zero mean and unit variance (paper setting)
        mean = tf.reduce_mean(img)
        std = tf.math.reduce_std(img)
        img = (img - mean) / tf.maximum(std, tf.constant(1e-8, dtype=img.dtype))

        if task == "classification":
            t = tf.cast(t, tf.int32)
        else:
            t = tf.cast(t, tf.float32)

        return img, t

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


def normalize_image_paths(df: pd.DataFrame, project_root: str = None) -> pd.DataFrame:
    """
    Normalize image paths in DataFrame to absolute paths.

    Args:
        df: DataFrame with 'Image_Npy_Path' column
        project_root: Project root directory (defaults to cwd)

    Returns:
        DataFrame with normalized paths

    Raises:
        ValueError: If 'Image_Npy_Path' column is missing
    """
    if "Image_Npy_Path" not in df.columns:
        raise ValueError("DataFrame must have 'Image_Npy_Path' column")

    if project_root is None:
        project_root = os.getcwd()

    def _norm_path(p):
        if not isinstance(p, str) or not p.strip():
            return ""
        p = p.strip()
        if os.path.isabs(p):
            return os.path.normpath(p)
        return os.path.normpath(os.path.join(project_root, p))

    df["Image_Npy_Path"] = df["Image_Npy_Path"].apply(_norm_path)

    return df


def filter_valid_images(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Filter DataFrame to keep only rows with valid image paths.

    Args:
        df: DataFrame with 'Image_Npy_Path' column

    Returns:
        Tuple of (filtered_df, num_missing, total_rows)
    """
    exists_mask = df["Image_Npy_Path"].apply(
        lambda p: isinstance(p, str) and os.path.exists(p)
    )

    missing = int((~exists_mask).sum())
    total = int(len(df))

    df_filtered = df.loc[exists_mask].reset_index(drop=True)

    return df_filtered, missing, total


def get_solver_names(
    solver_cols: Dict[str, List[str]], use_score: bool = False
) -> List[str]:
    """
    Extract clean solver names from column names.

    Args:
        solver_cols: Dictionary with 'runtime' and 'score' column lists
        use_score: Whether to use score columns

    Returns:
        List of solver names (without suffixes)
    """
    cols = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )

    names = [c.replace("_Runtime_s", "").replace("_Score_S_rel", "") for c in cols]

    return names
