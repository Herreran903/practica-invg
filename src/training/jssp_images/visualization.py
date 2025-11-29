# -*- coding: utf-8 -*-
"""
Visualization functions for JSSP image-based solver selection.

This module generates plots for:
- Classification: confusion matrix, per-class accuracy bars
- Multilabel: precision-recall curves, F1 bars per label
- Regression: scatter plots (predicted vs actual), MAE bars per solver
- Cross-fold: metric comparison across folds
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    fold_dir: str,
    fold_idx: int,
) -> None:
    """
    Generate and save confusion matrix heatmap for classification.

    Args:
        y_true: True class labels [N].
        y_pred: Predicted class labels [N].
        class_names: List of class names (solvers).
        fold_dir: Output directory.
        fold_idx: Fold index.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - Fold {fold_idx}")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(fold_dir, f"fold{fold_idx}_confusion.png"))
    plt.close(fig)


def plot_class_bars(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    fold_dir: str,
    fold_idx: int,
) -> None:
    """
    Generate bar plot of per-class accuracy (recall).

    Args:
        y_true: True class labels [N].
        y_pred: Predicted class labels [N].
        class_names: List of class names.
        fold_dir: Output directory.
        fold_idx: Fold index.
    """
    recalls = []
    for c in range(len(class_names)):
        idx = np.where(y_true == c)[0]
        if len(idx) == 0:
            recalls.append(0.0)
        else:
            recalls.append(float(np.mean(y_pred[idx] == c)))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars = ax.bar(class_names, recalls, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy (Recall)")
    ax.set_xlabel("Solver")
    ax.set_title(f"Per-Class Accuracy - Fold {fold_idx}")
    ax.axhline(
        y=np.mean(recalls), color="red", linestyle="--", linewidth=1, label="Mean"
    )
    plt.xticks(rotation=45, ha="right")
    ax.legend()

    # Add value labels on bars
    for bar, val in zip(bars, recalls):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(fold_dir, f"fold{fold_idx}_class_bars.png"))
    plt.close(fig)


def plot_pr_curves_multilabel(
    y_true: np.ndarray,
    class_names: List[str],
    fold_dir: str,
    fold_idx: int,
) -> None:
    """
    Generate precision-recall curves for multilabel classification.

    Requires y_scores.npy file to be present in fold_dir.

    Args:
        y_true: True binary labels [N, C].
        class_names: List of label names.
        fold_dir: Output directory.
        fold_idx: Fold index.
    """
    # Load scores if available
    scores_path = os.path.join(fold_dir, f"fold{fold_idx}_y_scores.npy")
    if not os.path.exists(scores_path):
        print(f"⚠️  Scores not found for PR curves: {scores_path}")
        return

    y_scores = np.load(scores_path)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    for j, name in enumerate(class_names):
        yt = y_true[:, j]
        ys = y_scores[:, j]

        # Skip if only one class present
        if np.unique(yt).size < 2:
            continue

        precision, recall, _ = precision_recall_curve(yt, ys)

        # Compute AP for legend
        from sklearn.metrics import average_precision_score

        ap = average_precision_score(yt, ys)

        ax.plot(recall, precision, label=f"{name} (AP={ap:.2f})", linewidth=1.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves - Fold {fold_idx}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(fold_dir, f"fold{fold_idx}_pr_curves.png"))
    plt.close(fig)


def plot_f1_bars_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    fold_dir: str,
    fold_idx: int,
) -> None:
    """
    Generate bar plot of F1-score per label for multilabel classification.

    Args:
        y_true: True binary labels [N, C].
        y_pred: Predicted binary labels [N, C].
        class_names: List of label names.
        fold_dir: Output directory.
        fold_idx: Fold index.
    """
    from sklearn.metrics import f1_score

    f1_per_label = []
    for j in range(y_true.shape[1]):
        f1_j = f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
        f1_per_label.append(f1_j)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars = ax.bar(class_names, f1_per_label, color="coral")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-Score")
    ax.set_xlabel("Solver")
    ax.set_title(f"F1-Score per Label - Fold {fold_idx}")
    ax.axhline(
        y=np.mean(f1_per_label), color="blue", linestyle="--", linewidth=1, label="Mean"
    )
    plt.xticks(rotation=45, ha="right")
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, f1_per_label):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(os.path.join(fold_dir, f"fold{fold_idx}_f1_bars.png"))
    plt.close(fig)


def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    fold_dir: str,
    fold_idx: int,
) -> None:
    """
    Generate scatter plots of predicted vs actual runtime for each solver.

    Args:
        y_true: True runtime values [N, C].
        y_pred: Predicted runtime values [N, C].
        class_names: List of solver names.
        fold_dir: Output directory.
        fold_idx: Fold index.
    """
    from sklearn.metrics import mean_absolute_error

    n_solvers = len(class_names)
    fig, axes = plt.subplots(
        1, n_solvers, figsize=(4 * n_solvers, 4), dpi=150, squeeze=False
    )

    for j, name in enumerate(class_names):
        ax = axes[0, j]

        # Scatter plot
        ax.scatter(y_true[:, j], y_pred[:, j], s=8, alpha=0.6, color="steelblue")

        # Perfect prediction line
        min_val = y_true[:, j].min()
        max_val = y_true[:, j].max()
        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1, label="Perfect")

        # Compute MAE
        mae = mean_absolute_error(y_true[:, j], y_pred[:, j])

        ax.set_xlabel("True Runtime (s)")
        ax.set_ylabel("Predicted Runtime (s)")
        ax.set_title(f"{name}\nMAE={mae:.2f}s")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Regression Scatter Plots - Fold {fold_idx}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(fold_dir, f"fold{fold_idx}_reg_scatter.png"))
    plt.close(fig)


def plot_metrics_per_fold(
    fold_results: List[dict],
    metric_key: str,
    root_outdir: str,
) -> None:
    """
    Generate bar plot comparing a metric across all folds.

    Args:
        fold_results: List of dictionaries with per-fold metrics.
        metric_key: Key of the metric to plot (e.g., 'accuracy', 'f1_micro', 'mae').
        root_outdir: Root output directory.
    """
    fold_df = pd.DataFrame(fold_results)

    if metric_key not in fold_df.columns:
        print(f"⚠️  Metric '{metric_key}' not found in fold results")
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    fold_ids = fold_df["fold"].astype(str)
    values = fold_df[metric_key].values

    bars = ax.bar(fold_ids, values, color="seagreen")
    ax.set_ylabel(metric_key.upper())
    ax.set_xlabel("Fold")
    ax.set_title(f"{metric_key.upper()} per Fold")

    # Add mean line
    mean_val = np.mean(values)
    ax.axhline(
        y=mean_val,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Mean={mean_val:.3f}",
    )
    ax.legend()

    # Add value labels
    for i, (fold_id, val) in enumerate(zip(fold_ids, values)):
        ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(root_outdir, f"{metric_key}_per_fold.png"))
    plt.close(fig)


def plot_training_history(
    history: dict,
    fold_dir: str,
    fold_idx: int,
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        history: Keras history.history dictionary.
        fold_dir: Output directory.
        fold_idx: Fold index.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training History - Fold {fold_idx}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(fold_dir, f"fold{fold_idx}_training_history.png"))
    plt.close(fig)
