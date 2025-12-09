# -*- coding: utf-8 -*-
"""
Data utilities for JSSP tensor-based solver selection.

This module adapts the jssp_images data utilities for 3D tensor inputs.
Key difference: loads tensors (JOBS x MACHINES x 2) with padding to fixed size.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# Reuse common functions from jssp_images
from ..jssp_images.data_utils import (
    bss_index,
    detect_solver_cols,
    filter_valid_images,
)
from ..jssp_images.data_utils import multilabel_targets as _multilabel_targets_base
from ..jssp_images.data_utils import (
    normalize_image_paths,
)


def load_tensor_npy(
    path: bytes, max_jobs: int, max_machines: int, n_channels: int
) -> np.ndarray:
    """
    Load a JSSP tensor from .npy file and pad to fixed size.
 
    Supported input shapes:
    - 2D: (JOBS, MACHINES)            → interpreted as a single channel (duration)
    - 3D: (JOBS, MACHINES, C_in)      → first min(C_in, n_channels) channels copied
 
    Output shape is always (max_jobs, max_machines, n_channels).
 
    Args:
        path: Path to .npy file (as bytes from tf.numpy_function).
        max_jobs: Maximum number of jobs (for padding).
        max_machines: Maximum number of machines (for padding).
        n_channels: Number of channels (>=1).
 
    Returns:
        Padded tensor of shape (max_jobs, max_machines, n_channels) in float32.
 
    Raises:
        ValueError: If tensor shape is invalid or exceeds max dimensions.
    """
    arr = np.load(path.decode("utf-8")).astype(np.float32)
 
    # Case 1: 2D matrix (num_jobs x num_machines), as produced by tensor_converter.py
    if arr.ndim == 2:
        jobs, machines = arr.shape
        if jobs > max_jobs or machines > max_machines:
            raise ValueError(
                f"Tensor {arr.shape} exceeds limits "
                f"max_jobs={max_jobs}, max_machines={max_machines}"
            )
 
        out = np.zeros((max_jobs, max_machines, n_channels), dtype=np.float32)
        # Put durations in the first channel; additional channels remain zero.
        out[:jobs, :machines, 0] = arr
        return out
 
    # Case 2: 3D tensor (num_jobs x num_machines x C_in)
    if arr.ndim == 3:
        jobs, machines, c_in = arr.shape
        if jobs > max_jobs or machines > max_machines:
            raise ValueError(
                f"Tensor {arr.shape} exceeds limits "
                f"max_jobs={max_jobs}, max_machines={max_machines}"
            )
 
        out = np.zeros((max_jobs, max_machines, n_channels), dtype=np.float32)
        # Copy as many channels as possible
        c_copy = min(c_in, n_channels)
        out[:jobs, :machines, :c_copy] = arr[..., :c_copy]
        return out
 
    # Any other rank is unsupported
    raise ValueError(
        f"Expected tensor with 2 or 3 dims (JOBS, MACHINES[, CHANNELS]), got {arr.shape}"
    )


def make_dataset(
    paths: List[str],
    labels: np.ndarray,
    task: str,
    batch_size: int,
    shuffle: bool,
    config: dict,
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset for tensor inputs.

    Args:
        paths: List of paths to .npy tensor files.
        labels: Labels array.
        task: Task type ('classification' or 'multilabel').
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        config: Configuration dictionary.

    Returns:
        tf.data.Dataset ready for training.
    """
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})

    max_jobs = data_cfg.get("max_jobs", 10)
    max_machines = data_cfg.get("max_machines", 10)
    n_channels = data_cfg.get("n_channels", 2)
    seed = training_cfg.get("seed", 42)

    x = tf.constant(paths)
    y = tf.constant(labels)

    def _map(p, t):
        tensor = tf.numpy_function(
            lambda path: load_tensor_npy(path, max_jobs, max_machines, n_channels),
            [p],
            tf.float32,
        )
        tensor.set_shape([max_jobs, max_machines, n_channels])

        if task == "classification":
            t = tf.cast(t, tf.int32)
        else:
            t = tf.cast(t, tf.float32)
        return tensor, t

    ds = tf.data.Dataset.from_tensor_slices((x, y)).map(
        _map, num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_labels(
    df: pd.DataFrame,
    solver_cols: Dict[str, List[str]],
    task: str,
    use_score: bool,
    time_limit: float,
) -> np.ndarray:
    """
    Build labels for training based on task type (classification or multilabel).
 
    Reuses logic from jssp_images with time_limit parameter for multilabel.
 
    Args:
        df: DataFrame with solver performance data.
        solver_cols: Dictionary with 'runtime' and 'score' column lists.
        task: Task type ('classification' or 'multilabel').
        use_score: Whether to use score columns (classification only).
        time_limit: Time limit for multilabel task.
 
    Returns:
        Labels array with appropriate shape for the task.
 
    Raises:
        ValueError: If task is not supported.
    """
    if task == "classification":
        y = []
        for _, r in df.iterrows():
            cols = (
                solver_cols["score"]
                if (use_score and solver_cols["score"])
                else solver_cols["runtime"]
            )
            vals = r[cols].astype(float).values
            idx = int(np.nanargmin(vals))
            y.append(idx)
        y = np.array(y, dtype=np.int32)
 
    elif task == "multilabel":
        rt_cols = solver_cols["runtime"]
        y = np.stack(
            [
                _multilabel_targets_base(r, rt_cols, time_limit)
                for _, r in df.iterrows()
            ],
            axis=0,
        )
 
    else:
        raise ValueError(f"Unsupported task for jssp_tensors build_labels: {task}")
 
    return y


# Re-export commonly used functions
__all__ = [
    "detect_solver_cols",
    "bss_index",
    "normalize_image_paths",
    "filter_valid_images",
    "load_tensor_npy",
    "make_dataset",
    "build_labels",
]
