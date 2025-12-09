# -*- coding: utf-8 -*-
"""
Training loop for JSSP tensor-based solver selection.

This module reuses the high-level K-Fold logic from jssp_images but
builds datasets and models using the tensor-specific utilities:
- src/training/jssp_tensors/data_utils.py
- src/training/jssp_tensors/model_builder.py
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold

from .data_utils import bss_index, build_labels, make_dataset
from .model_builder import build_model_from_config
from ..jssp_images.training_loop import compute_bss_baseline
from ..jssp_images.evaluation import evaluate_fold


def train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    solver_cols: Dict[str, List[str]],
    task: Literal["classification", "multilabel"],
    use_score: bool,
    config: dict,
    fold_dir: str,
    fold_idx: int,
) -> Tuple[tf.keras.Model, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Train and evaluate a single fold for tensor inputs.

    This mirrors src/training/jssp_images/training_loop.train_fold, but uses:
    - jssp_tensors.make_dataset (tensors padded to [max_jobs, max_machines, n_channels])
    - jssp_tensors.build_model_from_config (CNN for 10x10x2)
    """
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    epochs = training_cfg.get("epochs", 25)
    batch_size = training_cfg.get("batch_size", 64)
    patience = training_cfg.get("early_stopping_patience", 6)

    # Time limit only matters for multilabel, but build_labels signature
    # always includes it for consistency with jssp_images.
    time_limit_s = data_cfg.get("time_limit_s", 60.0)

    # Build labels
    y_train = build_labels(train_df, solver_cols, task, use_score, time_limit_s)
    y_val = build_labels(val_df, solver_cols, task, use_score, time_limit_s)

    # Paths to tensors
    paths_train = train_df["Image_Npy_Path"].tolist()
    paths_val = val_df["Image_Npy_Path"].tolist()

    # Output dimension: number of solvers
    if task == "classification":
        cols = (
            solver_cols["score"]
            if (use_score and solver_cols["score"])
            else solver_cols["runtime"]
        )
        output_dim = len(cols)
    else:
        output_dim = len(solver_cols["runtime"])

    # Create datasets (tensor-specific make_dataset)
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

    # Build tensor model
    model = build_model_from_config(config, output_dim=output_dim, task=task)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    print(f"[FOLD {fold_idx}] Training model...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Inference on validation set
    print(f"[FOLD {fold_idx}] Running inference on validation set...")
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    y_score_list: list[np.ndarray] = []

    for xb, yb in ds_val:
        out = model(xb, training=False).numpy()
        if task == "classification":
            y_true_list.append(yb.numpy())
            y_pred_list.append(np.argmax(out, axis=1))
        else:  # multilabel
            y_true_list.append(yb.numpy())
            y_score_list.append(out)
            y_pred_list.append((out >= 0.5).astype(np.float32))

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # Save predictions
    os.makedirs(fold_dir, exist_ok=True)
    np.save(os.path.join(fold_dir, f"fold{fold_idx}_y_true.npy"), y_true)
    np.save(os.path.join(fold_dir, f"fold{fold_idx}_y_pred.npy"), y_pred)
    if task == "multilabel" and y_score_list:
        y_scores = np.concatenate(y_score_list, axis=0)
        np.save(os.path.join(fold_dir, f"fold{fold_idx}_y_scores.npy"), y_scores)

    metrics = {
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "epochs_trained": len(history.history["loss"]),
    }

    return model, metrics, y_true, y_pred


def run_kfold(
    df: pd.DataFrame,
    task: Literal["classification", "multilabel"],
    solver_cols: Dict[str, List[str]],
    use_score: bool,
    config: dict,
    root_outdir: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Execute K-Fold cross-validation for JSSP tensors.

    Mirrors src/training/jssp_images/training_loop.run_kfold, but uses:
    - tensor-specific build_labels
    - tensor-specific train_fold (which builds tensor datasets/models)
    while reusing compute_bss_baseline and evaluate_fold from jssp_images.
    """
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    folds = training_cfg.get("k_folds", 5)
    time_limit = data_cfg.get("time_limit_s", 60.0)
    seed = training_cfg.get("seed", 42)

    # Setup K-Fold splitter
    if task == "classification":
        labels = build_labels(df, solver_cols, task, use_score, time_limit)
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

    cols_for_names = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    solver_names = [
        c.replace("_Runtime_s", "").replace("_Score_S_rel", "") for c in cols_for_names
    ]

    fold_results: list[Dict] = []

    for i, (tr, va) in enumerate(splits, start=1):
        fold_dir = os.path.join(root_outdir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[tr].reset_index(drop=True)
        val_df = df.iloc[va].reset_index(drop=True)

        # BSS baseline
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

        # Train tensor model on this fold
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

        eval_metrics = evaluate_fold(
            y_true=y_true,
            y_pred=y_pred,
            task=task,
            solver_names=solver_names,
            fold_dir=fold_dir,
            fold_idx=i,
            val_df=val_df,
            solver_runtime_cols=cols_for_names,
            config=config,
        )

        fold_result = {
            "fold": i,
            "bss_solver": bss_col,
            "bss_baseline": baseline_metric,
            **train_metrics,
            **eval_metrics,
        }
        fold_results.append(fold_result)

        with open(os.path.join(fold_dir, f"fold{i}_metrics.json"), "w") as f:
            json.dump(fold_result, f, indent=2)
        print(f"[FOLD {i}] Validation metrics: {eval_metrics}")

    # Save per-fold table
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(os.path.join(root_outdir, "metrics_per_fold.csv"), index=False)
     
    # Primary metric to aggregate:
    # - accuracy   for classification
    # - f1_micro   for multilabel
    metric_key = {
        "classification": "accuracy",
        "multilabel": "f1_micro",
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
    
    # Optionally aggregate resolved_rate across folds (for tasks where it exists,
    # e.g. multilabel JSSP tensors using evaluate_fold from jssp_images)
    resolved_vals = [r["resolved_rate"] for r in fold_results if "resolved_rate" in r]
    if resolved_vals:
        summary["resolved_rate_mean"] = float(np.mean(resolved_vals))
        summary["resolved_rate_std"] = float(np.std(resolved_vals))
        summary["resolved_rate_min"] = float(np.min(resolved_vals))
        summary["resolved_rate_max"] = float(np.max(resolved_vals))

    # Optionally aggregate AST (average solving time) across folds when available.
    # This is the "Average Running Time" metric from the paper, measured on the
    # portfolio induced by the CNN's predictions.
    ast_vals = [r["AST_sec"] for r in fold_results if "AST_sec" in r]
    if ast_vals:
        summary["AST_sec_mean"] = float(np.mean(ast_vals))
        summary["AST_sec_std"] = float(np.std(ast_vals))
        summary["AST_sec_min"] = float(np.min(ast_vals))
        summary["AST_sec_max"] = float(np.max(ast_vals))

    # Optionally aggregate misclassification rate (1 - accuracy) across folds
    mis_vals = [
        r["misclassification_rate"]
        for r in fold_results
        if "misclassification_rate" in r
    ]
    if mis_vals:
        summary["misclassification_rate_mean"] = float(np.mean(mis_vals))
        summary["misclassification_rate_std"] = float(np.std(mis_vals))
        summary["misclassification_rate_min"] = float(np.min(mis_vals))
        summary["misclassification_rate_max"] = float(np.max(mis_vals))
     
    with open(os.path.join(root_outdir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({task})")
    print(f"{'='*60}")
    print(f"{metric_key.upper()}: {summary['mean']:.4f} ± {summary['std']:.4f}")
    print(f"Range: [{summary['min']:.4f}, {summary['max']:.4f}]")

    return fold_results, summary


__all__ = ["train_fold", "run_kfold", "compute_bss_baseline"]
