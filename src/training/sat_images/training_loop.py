# -*- coding: utf-8 -*-
"""
Training loop for SAT image-based solver selection.

Extends jssp_images training with:
- K-Fold repetitions (e.g., 5x5 cross-validation)
- SAT-specific metrics (resolved_rate, AST)
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

# Reuse base training from jssp_images
from ..jssp_images.training_loop import train_fold as _train_fold_base
from .data_utils import (
    bss_index,
    build_labels,
    compute_ast_bss,
    make_dataset,
)
from .evaluation import evaluate_fold
from .model_builder import build_model_from_config


def run_kfold_with_repeats(
    df: pd.DataFrame,
    task: str,
    solver_cols: Dict[str, List[str]],
    use_score: bool,
    config: dict,
    root_outdir: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Execute K-Fold cross-validation with optional repetitions.

    Supports multiple repetitions (e.g., 5x5) where each repetition
    uses a different random seed.

    Args:
        df: Complete DataFrame.
        task: Task type.
        solver_cols: Detected solver columns.
        use_score: Whether to use score columns.
        config: Configuration dictionary.
        root_outdir: Root output directory.

    Returns:
        Tuple of (all_fold_results, global_summary).
    """
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    folds = training_cfg.get("k_folds", 5)
    repeats = training_cfg.get("k_fold_repeats", 1)
    base_seed = training_cfg.get("seed", 42)
    time_limit = data_cfg.get("time_limit_s", 1800.0)
    feat_time_col = data_cfg.get("feat_time_column", "_feat_time_zero_")

    # If feat_time_col doesn't exist, create it with zeros
    if feat_time_col not in df.columns:
        df[feat_time_col] = 0.0

    metric_key = {
        "classification": "accuracy",
        "multilabel": "f1_micro",
        "regression": "mae",
    }[task]
 
    all_results = []
    per_rep_summary = []
    resolved_all: List[float] = []

    for rep in range(repeats):
        seed_rep = base_seed + rep
        rep_dir = os.path.join(root_outdir, f"rep_{rep+1}")
        os.makedirs(rep_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"REPETITION {rep+1}/{repeats} (seed={seed_rep})")
        print(f"{'='*60}")

        # Run K-Fold for this repetition
        fold_results, rep_summary = _run_single_kfold(
            df=df,
            task=task,
            solver_cols=solver_cols,
            use_score=use_score,
            config=config,
            outdir=rep_dir,
            seed=seed_rep,
            folds=folds,
            time_limit=time_limit,
            feat_time_col=feat_time_col,
        )

        all_results.extend([r[metric_key] for r in fold_results])

        # Per-repetition summary for the primary metric
        rep_row = {
            "rep": rep + 1,
            "mean": rep_summary["mean"],
            "std": rep_summary["std"],
        }

        # Collect resolved_rate statistics per repetition if available
        resolved_vals_rep = [r["resolved_rate"] for r in fold_results if "resolved_rate" in r]
        if resolved_vals_rep:
            rep_row["resolved_rate_mean"] = float(np.mean(resolved_vals_rep))
            rep_row["resolved_rate_std"] = float(np.std(resolved_vals_rep))
            rep_row["resolved_rate_min"] = float(np.min(resolved_vals_rep))
            rep_row["resolved_rate_max"] = float(np.max(resolved_vals_rep))
            # Also accumulate all resolved_rate values across repetitions
            resolved_all.extend(resolved_vals_rep)

        per_rep_summary.append(rep_row)

        print(
            f"[REP {rep+1}] {metric_key.upper()}: {rep_summary['mean']:.4f} ± {rep_summary['std']:.4f}"
        )

    # Global summary across all repetitions
    global_mean = float(np.mean(all_results))
    global_std = float(np.std(all_results))

    global_summary = {
        "task": task,
        "metric": metric_key,
        "mean": global_mean,
        "std": global_std,
        "min": float(np.min(all_results)),
        "max": float(np.max(all_results)),
        "folds": folds,
        "repeats": repeats,
    }

    # Global resolved_rate aggregated across all folds and repetitions
    if resolved_all:
        global_summary["resolved_rate_mean"] = float(np.mean(resolved_all))
        global_summary["resolved_rate_std"] = float(np.std(resolved_all))
        global_summary["resolved_rate_min"] = float(np.min(resolved_all))
        global_summary["resolved_rate_max"] = float(np.max(resolved_all))

    # Save per-repetition summary
    pd.DataFrame(per_rep_summary).to_csv(
        os.path.join(root_outdir, "metrics_summary_per_rep.csv"), index=False
    )

    # Save global summary
    with open(os.path.join(root_outdir, "metrics_summary_GLOBAL.json"), "w") as f:
        json.dump(global_summary, f, indent=2)

    return all_results, global_summary


def _run_single_kfold(
    df: pd.DataFrame,
    task: str,
    solver_cols: Dict[str, List[str]],
    use_score: bool,
    config: dict,
    outdir: str,
    seed: int,
    folds: int,
    time_limit: float,
    feat_time_col: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Run a single K-Fold cross-validation.

    Internal function called by run_kfold_with_repeats.
    """
    # Setup splitter
    if task == "classification":
        labels = build_labels(df, solver_cols, task, use_score, time_limit)
        binc = np.bincount(labels)
        min_class = binc.min()

        if min_class < folds:
            print(
                f"⚠️  Minority class: {min_class} samples < folds={folds}. Adjusting to {max(2, min_class)}"
            )
            folds = max(2, int(min_class))

        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(df, labels)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(df)

    # Get solver names and runtime columns.
    # For SAT-specific metrics that depend on runtimes (resolved_rate, AST, etc.),
    # we must always use *_Runtime_s columns, even if the model was trained
    # using scores for solver selection.
    cols_for_names = (
        solver_cols["score"]
        if (use_score and solver_cols["score"])
        else solver_cols["runtime"]
    )
    runtime_cols = solver_cols["runtime"]
    solver_names = [
        c.replace("_Runtime_s", "").replace("_Score_S_rel", "") for c in runtime_cols
    ]

    fold_results = []
    per_fold_rows = []

    for i, (tr, va) in enumerate(splits, start=1):
        fold_dir = os.path.join(outdir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        train_df = df.iloc[tr].reset_index(drop=True)
        val_df = df.iloc[va].reset_index(drop=True)

        # Compute BSS baseline
        bss_idx_i = bss_index(train_df, solver_cols, use_score)
        bss_col = cols_for_names[bss_idx_i]

        # BSS metrics
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

        if task == "classification":
            y_val_true = build_labels(val_df, solver_cols, task, use_score, time_limit)
            y_bss = np.full_like(y_val_true, bss_idx_i)
            bss_metric = accuracy_score(y_val_true, y_bss)
            # AST must be computed using true runtimes, not scores
            bss_ast = compute_ast_bss(
                val_df, runtime_cols, bss_idx_i, feat_time_col, time_limit
            )
            print(
                f"\n[FOLD {i}] BSS={bss_col} | BSS_acc={bss_metric:.4f} | BSS_AST={bss_ast:.1f}s"
            )
        elif task == "multilabel":
            from .data_utils import multilabel_targets

            rt_cols = solver_cols["runtime"]
            y_val_true = np.stack(
                [
                    multilabel_targets(r, rt_cols, time_limit)
                    for _, r in val_df.iterrows()
                ],
                axis=0,
            )
            y_bss = np.zeros_like(y_val_true)
            y_bss[:, bss_idx_i] = 1.0
            bss_metric = f1_score(
                y_val_true.flatten(), y_bss.flatten(), average="micro", zero_division=0
            )
            bss_ast = compute_ast_bss(
                val_df, rt_cols, bss_idx_i, feat_time_col, time_limit
            )
            print(
                f"\n[FOLD {i}] BSS={bss_col} | BSS_f1_micro={bss_metric:.4f} | BSS_AST={bss_ast:.1f}s"
            )
        else:  # regression
            rt_cols = solver_cols["runtime"]
            const_pred = (
                train_df[rt_cols[bss_idx_i]]
                .astype(float)
                .fillna(time_limit * 10)
                .mean()
            )
            y_val_true = val_df[rt_cols].astype(float).fillna(time_limit * 10).values
            y_bss = np.full_like(y_val_true, const_pred)
            bss_metric = mean_absolute_error(y_val_true, y_bss)
            bss_ast = None
            print(f"\n[FOLD {i}] BSS={bss_col} | BSS_mae={bss_metric:.4f}")

        # Train fold (reuse base training)
        print(f"[FOLD {i}] Training model...")
        model, train_metrics, y_true, y_pred = _train_fold_base(
            train_df=train_df,
            val_df=val_df,
            solver_cols=solver_cols,
            task=task,
            use_score=use_score,
            config=config,
            fold_dir=fold_dir,
            fold_idx=i,
        )

        # Evaluate with SAT-specific metrics
        eval_metrics = evaluate_fold(
            y_true=y_true,
            y_pred=y_pred,
            task=task,
            solver_names=solver_names,
            fold_dir=fold_dir,
            fold_idx=i,
            val_df=val_df,
            # For resolved_rate and AST we always use runtime columns.
            solver_runtime_cols=runtime_cols,
            config=config,
        )

        # Combine metrics
        fold_result = {
            "fold": i,
            "bss_solver": bss_col,
            "bss_baseline": bss_metric,
            **train_metrics,
            **eval_metrics,
        }

        if bss_ast is not None:
            fold_result["bss_ast"] = bss_ast

        fold_results.append(fold_result)

        # Save fold metrics
        with open(os.path.join(fold_dir, f"fold{i}_metrics.json"), "w") as f:
            json.dump(fold_result, f, indent=2)

        print(f"[FOLD {i}] Metrics: {eval_metrics}")

    # Save per-fold table
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(os.path.join(outdir, "metrics_per_fold.csv"), index=False)

    # Compute summary
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

    # Aggregate resolved_rate across folds (classification and multilabel where available)
    resolved_vals = [r["resolved_rate"] for r in fold_results if "resolved_rate" in r]
    if resolved_vals:
        summary["resolved_rate_mean"] = float(np.mean(resolved_vals))
        summary["resolved_rate_std"] = float(np.std(resolved_vals))
        summary["resolved_rate_min"] = float(np.min(resolved_vals))
        summary["resolved_rate_max"] = float(np.max(resolved_vals))
 
    # Save summary
    with open(os.path.join(outdir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plot metrics per fold
    from .visualization import plot_metrics_per_fold

    plot_metrics_per_fold(fold_results, metric_key, outdir)

    return fold_results, summary


# Alias for compatibility
run_kfold = run_kfold_with_repeats

__all__ = [
    "run_kfold",
    "run_kfold_with_repeats",
]
