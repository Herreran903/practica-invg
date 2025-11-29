# -*- coding: utf-8 -*-
"""
Training loop and K-Fold cross-validation for JSSP image-based solver selection.

This module handles:
- Training individual folds with early stopping
- K-Fold cross-validation (stratified for classification)
- Baseline Single Solver (BSS) computation
- Per-fold predictions and metrics persistence
"""

import json
import os
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold

from .data_utils import bss_index, build_labels, make_dataset
from .model_builder import build_model_from_config


def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    solver_cols: Dict[str, List[str]],
    task: Literal["classification", "multilabel", "regression"],
    use_score: bool,
    config: dict,
    fold_dir: str,
    fold_idx: int,
) -> Tuple[tf.keras.Model, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train and evaluate a single fold.

    This function:
    1. Builds labels for train/val sets
    2. Creates TensorFlow datasets
    3. Builds and trains the model with early stopping
    4. Performs inference and saves predictions
    5. Returns model and raw predictions for evaluation

    Args:
        train_df: Training partition DataFrame.
        val_df: Validation partition DataFrame.
        solver_cols: Dictionary with 'runtime' and 'score' column lists.
        task: Task type - 'classification', 'multilabel', or 'regression'.
        use_score: Whether to use score columns for "best" solver.
        config: Configuration dictionary with training parameters.
        fold_dir: Output directory for this fold.
        fold_idx: Fold index (1-based).

    Returns:
        Tuple of (trained_model, metrics_dict, y_true, y_pred).
        - metrics_dict contains task-specific metrics computed during training
        - y_true and y_pred are raw arrays for further evaluation

    Design Decision:
        Separates training from evaluation to allow flexible metric computation
        and visualization in separate modules.
    """
    # Extract configuration parameters
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    epochs = training_cfg.get("epochs", 25)
    batch_size = training_cfg.get("batch_size", 64)
    patience = training_cfg.get("early_stopping_patience", 6)

    # Build labels
    y_train = build_labels(train_df, solver_cols, task, use_score)
    y_val = build_labels(val_df, solver_cols, task, use_score)

    # Get image paths
    paths_train = train_df["Image_Npy_Path"].tolist()
    paths_val = val_df["Image_Npy_Path"].tolist()

    # Determine output dimension
    if task == "classification":
        cols = (
            solver_cols["score"]
            if (use_score and solver_cols["score"])
            else solver_cols["runtime"]
        )
        output_dim = len(cols)
    else:
        output_dim = len(solver_cols["runtime"])

    # Create datasets
    ds_train = make_dataset(
        paths=paths_train,
        labels=y_train,
        task=task,
        batch_size=batch_size,
        shuffle=True,
        config=config,
    )
    ds_val = make_dataset(
        paths=paths_val,
        labels=y_val,
        task=task,
        batch_size=batch_size,
        shuffle=False,
        config=config,
    )

    # Build model
    model = build_model_from_config(config, output_dim=output_dim, task=task)

    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    # Train model
    print(f"[FOLD {fold_idx}] Training model...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Inference on validation set (batch by batch to avoid retracing)
    print(f"[FOLD {fold_idx}] Running inference on validation set...")
    y_true_list, y_pred_list, y_score_list = [], [], []

    for xb, yb in ds_val:
        out = model(xb, training=False).numpy()

        if task == "classification":
            y_true_list.append(yb.numpy())
            y_pred_list.append(np.argmax(out, axis=1))
        elif task == "multilabel":
            y_true_list.append(yb.numpy())
            y_score_list.append(out)
            y_pred_list.append((out >= 0.5).astype(np.float32))
        else:  # regression
            y_true_list.append(yb.numpy())
            y_pred_list.append(out)

    # Consolidate predictions
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # Save predictions
    np.save(os.path.join(fold_dir, f"fold{fold_idx}_y_true.npy"), y_true)
    np.save(os.path.join(fold_dir, f"fold{fold_idx}_y_pred.npy"), y_pred)

    # For multilabel, also save scores
    if task == "multilabel" and y_score_list:
        y_scores = np.concatenate(y_score_list, axis=0)
        np.save(os.path.join(fold_dir, f"fold{fold_idx}_y_scores.npy"), y_scores)

    # Extract training metrics from history
    metrics = {
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "epochs_trained": len(history.history["loss"]),
    }

    return model, metrics, y_true, y_pred


def compute_bss_baseline(
    val_df: pd.DataFrame,
    bss_idx: int,
    solver_cols: Dict[str, List[str]],
    task: Literal["classification", "multilabel", "regression"],
    use_score: bool,
    time_limit: float,
) -> Tuple[float, str]:
    """
    Compute baseline metric using Baseline Single Solver (BSS) strategy.

    BSS selects the solver with best average performance on training set
    and applies it to all validation instances.

    Args:
        val_df: Validation DataFrame.
        bss_idx: Index of the BSS solver.
        solver_cols: Dictionary with 'runtime' and 'score' column lists.
        task: Task type.
        use_score: Whether score columns were used.
        time_limit: Time limit for multilabel task.

    Returns:
        Tuple of (baseline_metric, metric_name).
    """
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    from .data_utils import multilabel_targets

    cols = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    bss_col = cols[bss_idx]

    if task == "classification":
        y_val_true = build_labels(val_df, solver_cols, task, use_score)
        y_bss = np.full_like(y_val_true, bss_idx)
        baseline_metric = accuracy_score(y_val_true, y_bss)
        metric_name = "accuracy"
    elif task == "multilabel":
        rt_cols = solver_cols["runtime"]
        y_val_true = np.stack(
            [multilabel_targets(r, rt_cols, time_limit) for _, r in val_df.iterrows()],
            axis=0,
        )
        y_bss = np.zeros_like(y_val_true)
        y_bss[:, bss_idx] = 1.0
        baseline_metric = f1_score(
            y_val_true.flatten(), y_bss.flatten(), average="micro", zero_division=0
        )
        metric_name = "f1_micro"
    else:  # regression
        rt_cols = solver_cols["runtime"]
        const_pred = (
            val_df[rt_cols[bss_idx]].astype(float).fillna(time_limit * 10).mean()
        )
        y_val_true = val_df[rt_cols].astype(float).fillna(time_limit * 10).values
        y_bss = np.full_like(y_val_true, const_pred)
        baseline_metric = mean_absolute_error(y_val_true, y_bss)
        metric_name = "mae"

    return baseline_metric, metric_name


def run_kfold(
    df: pd.DataFrame,
    task: Literal["classification", "multilabel", "regression"],
    solver_cols: Dict[str, List[str]],
    use_score: bool,
    config: dict,
    root_outdir: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Execute K-Fold cross-validation with BSS baseline comparison.

    This function:
    1. Splits data using StratifiedKFold (classification) or KFold (others)
    2. Trains and evaluates each fold
    3. Computes BSS baseline for each fold
    4. Aggregates results and saves summary

    Args:
        df: Complete filtered DataFrame.
        task: Task type.
        solver_cols: Detected solver columns.
        use_score: Whether to use score columns.
        config: Configuration dictionary.
        root_outdir: Root output directory for this run.

    Returns:
        Tuple of (fold_results_list, summary_dict).
        - fold_results_list: List of dicts with per-fold metrics
        - summary_dict: Aggregated statistics (mean, std)

    Design Decision:
        Uses stratified splitting for classification to maintain class balance.
        Adjusts number of folds if minority class is too small.
    """
    from .evaluation import evaluate_fold

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    folds = training_cfg.get("k_folds", 5)
    time_limit = data_cfg.get("time_limit_s", 60.0)
    seed = training_cfg.get("seed", 42)

    # Setup K-Fold splitter
    if task == "classification":
        labels = build_labels(df, solver_cols, task, use_score)
        binc = np.bincount(labels)
        min_class = binc.min()

        if min_class < folds:
            print(
                f"⚠️  Minority class has {min_class} samples < folds={folds}. "
                f"Adjusting folds to {max(2, min_class)}."
            )
            folds = max(2, int(min_class))

        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(df, labels)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(df)

    # Get solver names for reporting
    cols_for_names = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    solver_names = [
        c.replace("_Runtime_s", "").replace("_Score_S_rel", "") for c in cols_for_names
    ]

    # Train and evaluate each fold
    fold_results = []

    for i, (tr, va) in enumerate(splits, start=1):
        fold_dir = os.path.join(root_outdir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[tr].reset_index(drop=True)
        val_df = df.iloc[va].reset_index(drop=True)

        # Compute BSS baseline
        bss_idx_i = bss_index(train_df, solver_cols, use_score)
        baseline_metric, metric_name = compute_bss_baseline(
            val_df, bss_idx_i, solver_cols, task, use_score, time_limit
        )

        cols = (
            solver_cols["score"]
            if (use_score and solver_cols["score"])
            else solver_cols["runtime"]
        )
        bss_col = cols[bss_idx_i]

        print(f"\n{'='*60}")
        print(f"FOLD {i}/{folds}")
        print(f"{'='*60}")
        print(f"BSS: {bss_col} | BSS {metric_name}: {baseline_metric:.4f}")

        # Train fold
        model, train_metrics, y_true, y_pred = train_fold(
            train_df=train_df,
            val_df=val_df,
            solver_cols=solver_cols,
            task=task,
            use_score=use_score,
            config=config,
            fold_dir=fold_dir,
            fold_idx=i,
        )

        # Evaluate fold (compute metrics and generate visualizations)
        eval_metrics = evaluate_fold(
            y_true=y_true,
            y_pred=y_pred,
            task=task,
            solver_names=solver_names,
            fold_dir=fold_dir,
            fold_idx=i,
        )

        # Combine metrics
        fold_result = {
            "fold": i,
            "bss_solver": bss_col,
            "bss_baseline": baseline_metric,
            **train_metrics,
            **eval_metrics,
        }

        fold_results.append(fold_result)

        # Save fold metrics
        with open(os.path.join(fold_dir, f"fold{i}_metrics.json"), "w") as f:
            json.dump(fold_result, f, indent=2)

        print(f"[FOLD {i}] Validation metrics: {eval_metrics}")

    # Save per-fold results table
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(os.path.join(root_outdir, "metrics_per_fold.csv"), index=False)

    # Compute aggregated statistics
    metric_key = {
        "classification": "accuracy",
        "multilabel": "f1_micro",
        "regression": "mae",
    }[task]

    metric_values = [r[metric_key] for r in fold_results]
    summary = {
        "task": task,
        "metric": metric_key,
        "mean": float(np.mean(metric_values)),
        "std": float(np.std(metric_values)),
        "min": float(np.min(metric_values)),
        "max": float(np.max(metric_values)),
        "folds": folds,
    }

    # Save summary
    with open(os.path.join(root_outdir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({task})")
    print(f"{'='*60}")
    print(f"{metric_key.upper()}: {summary['mean']:.4f} ± {summary['std']:.4f}")
    print(f"Range: [{summary['min']:.4f}, {summary['max']:.4f}]")

    return fold_results, summary
