# -*- coding: utf-8 -*-
"""
Command-line interface for JSSP image-based solver selection training.

This module provides a professional CLI for training CNN models on JSSP datasets.
It handles argument parsing, configuration loading, data preparation, and
orchestrates the complete training pipeline.

Usage:
    python -m src.training.jssp_images.cli \
        --config src/training/jssp_images/config.yaml \
        --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv \
        --task classification \
        --epochs 30 \
        --folds 5
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config_loader import load_config, merge_cli_args, resolve_paths
from .data_utils import detect_solver_cols, filter_valid_images, normalize_image_paths
from .training_loop import run_kfold
from .visualization import plot_metrics_per_fold


def setup_reproducibility(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train CNN for JSSP solver selection from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classification with default config
  python -m src.training.jssp_images.cli \\
    --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv \\
    --task classification

  # Multilabel with custom config and parameters
  python -m src.training.jssp_images.cli \\
    --config src/training/jssp_images/config.yaml \\
    --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv \\
    --task multilabel \\
    --epochs 50 \\
    --folds 10 \\
    --batch_size 32

  # Regression using score columns
  python -m src.training.jssp_images.cli \\
    --csv data/jssp/datasets/jsp_cnn_data_gen/ground_truth.csv \\
    --task regression \\
    --use_score \\
    --epochs 30
        """,
    )

    # Required arguments
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with Image_Npy_Path and solver performance columns",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "multilabel"],
        help="Task type: classification (best solver) or multilabel (viable solvers)",
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="src/training/jssp_images/config.yaml",
        help="Path to configuration YAML file (default: src/training/jssp_images/config.yaml)",
    )
    parser.add_argument(
        "--use_score",
        action="store_true",
        help="Use *_Score_S_rel columns (if available) instead of *_Runtime_s for 'best' solver",
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs (overrides config)"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument(
        "--folds", type=int, help="Number of K-Fold splits (overrides config)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for Adam optimizer (overrides config)",
    )
    parser.add_argument(
        "--time_limit",
        type=float,
        help="Time limit in seconds (overrides data.time_limit_s in config)",
    )
    parser.add_argument(
        "--out_parent",
        type=str,
        help="Parent directory for output (default: training/jssp/results/)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name prefix for this run (default: jssp_images_cnn)",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        help="Comma-separated solver names to use (e.g. 'solverA,solverB')",
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility (overrides config)"
    )

    return parser.parse_args()


def prepare_output_directory(config: dict, task: str) -> str:
    """
    Create timestamped output directory for this run.

    Args:
        config: Configuration dictionary.
        task: Task type.

    Returns:
        Path to output directory.
    """
    output_cfg = config.get("output", {})

    parent_dir = output_cfg.get("parent_dir", "training/jssp/results")
    run_name = output_cfg.get("run_name", "jssp_images_cnn")
    timestamp_fmt = output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S")
    append_task = output_cfg.get("append_task_to_dirname", True)

    timestamp = datetime.now().strftime(timestamp_fmt)

    if append_task:
        run_dirname = f"{run_name}_{task}_{timestamp}"
    else:
        run_dirname = f"{run_name}_{timestamp}"

    outdir = os.path.join(parent_dir, run_dirname)
    os.makedirs(outdir, exist_ok=True)

    return outdir


def save_run_info(
    outdir: str, config: dict, args: argparse.Namespace, df: pd.DataFrame
) -> None:
    """
    Save run information and configuration to output directory.

    Args:
        outdir: Output directory.
        config: Configuration dictionary.
        args: Parsed CLI arguments.
        df: Filtered DataFrame.
    """
    import json

    import yaml

    # Save configuration
    with open(os.path.join(outdir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Save run info
    run_info = {
        "task": args.task,
        "csv_path": args.csv,
        "use_score": args.use_score,
        "num_instances": len(df),
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(sys.argv),
    }

    with open(os.path.join(outdir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)


def main():
    """
    Main entry point for CLI.

    Workflow:
    1. Parse arguments and load configuration
    2. Setup reproducibility
    3. Load and validate CSV data
    4. Normalize and filter image paths
    5. Detect solver columns
    6. Create output directory
    7. Run K-Fold cross-validation
    8. Generate summary visualizations
    9. Save final report
    """
    # Parse arguments
    args = parse_arguments()

    # Load and merge configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    config = merge_cli_args(config, args)

    # Apply CLI overrides not handled by merge_cli_args
    if getattr(args, "time_limit", None) is not None:
        config.setdefault("data", {})["time_limit_s"] = args.time_limit

    config = resolve_paths(config)

    # Setup reproducibility
    seed = config.get("training", {}).get("seed", 42)
    setup_reproducibility(seed)
    print(f"Random seed set to: {seed}")

    # Load CSV
    print(f"\nLoading CSV: {args.csv}")
    if not os.path.exists(args.csv):
        print(f"❌ Error: CSV file not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")

    # Validate required column
    if "Image_Npy_Path" not in df.columns:
        print("❌ Error: CSV must contain 'Image_Npy_Path' column")
        sys.exit(1)

    # Normalize and filter image paths
    print("\nNormalizing image paths...")
    df = normalize_image_paths(df, "Image_Npy_Path")
 
    print("Filtering valid images...")
    df, missing_count, _total_rows = filter_valid_images(df)

    if missing_count > 0:
        print(f"⚠️  Removed {missing_count} rows with missing/invalid images")

    if len(df) == 0:
        print("❌ Error: No valid images found after filtering")
        sys.exit(1)

    print(f"✓ {len(df)} valid instances ready for training")

    # Detect solver columns
    print("\nDetecting solver columns...")
    try:
        solver_cols = detect_solver_cols(df)
        print(f"✓ Found {len(solver_cols['runtime'])} runtime columns")
        if solver_cols["score"]:
            print(f"✓ Found {len(solver_cols['score'])} score columns")
        else:
            print("  No score columns found")
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Optional solver filtering (similar to SAT training CLI)
    if getattr(args, "solvers", None):
        selected = [s.strip() for s in args.solvers.split(",") if s.strip()]
        print(f"\nFiltering to solvers: {selected}")

        keep_runtime = [
            f"{s}_Runtime_s" for s in selected if f"{s}_Runtime_s" in df.columns
        ]
        keep_score = [
            f"{s}_Score_S_rel" for s in selected if f"{s}_Score_S_rel" in df.columns
        ]

        if not keep_runtime:
            print("❌ Error: no matching runtime columns found for selected solvers")
            sys.exit(1)

        base_cols = ["Image_Npy_Path"]
        df = df[base_cols + keep_runtime + keep_score]
        solver_cols = {"runtime": keep_runtime, "score": keep_score}

        print(f"✓ Filtered to {len(keep_runtime)} solvers")

    # Create output directory
    outdir = prepare_output_directory(config, args.task)
    print(f"\n📁 Output directory: {outdir}")

    # Save run information
    save_run_info(outdir, config, args, df)

    # Print training configuration
    training_cfg = config.get("training", {})
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Use score: {args.use_score}")
    print(f"Epochs: {training_cfg.get('epochs', 25)}")
    print(f"Batch size: {training_cfg.get('batch_size', 64)}")
    print(f"K-Folds: {training_cfg.get('k_folds', 5)}")
    print(f"Learning rate: {training_cfg.get('learning_rate', 1e-3)}")
    print(f"Early stopping patience: {training_cfg.get('early_stopping_patience', 6)}")
    print(f"{'='*60}\n")

    # Run K-Fold training
    start_time = time.time()

    try:
        fold_results, summary = run_kfold(
            df=df,
            task=args.task,
            solver_cols=solver_cols,
            use_score=args.use_score,
            config=config,
            root_outdir=outdir,
        )
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    elapsed_time = time.time() - start_time

    # Generate cross-fold visualization
    metric_key = summary["metric"]
    plot_metrics_per_fold(fold_results, metric_key, outdir)

    # Save final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Metric: {metric_key.upper()}")
    print(f"Mean: {summary['mean']:.4f}")
    print(f"Std: {summary['std']:.4f}")
    print(f"Range: [{summary['min']:.4f}, {summary['max']:.4f}]")
    print(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
    print(f"{'='*60}\n")

    # Create README
    readme_path = os.path.join(outdir, "README.txt")
    with open(readme_path, "w") as f:
        f.write("JSSP Image-Based Solver Selection - Training Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Use score: {args.use_score}\n")
        f.write(f"K-Folds: {summary['folds']}\n")
        f.write(f"Metric: {metric_key.upper()}\n")
        f.write(f"Mean: {summary['mean']:.4f}\n")
        f.write(f"Std: {summary['std']:.4f}\n")
        f.write(f"Range: [{summary['min']:.4f}, {summary['max']:.4f}]\n")
        f.write(f"Training time: {elapsed_time:.2f}s\n\n")
        f.write("Files:\n")
        f.write("  - config.yaml: Configuration used for this run\n")
        f.write("  - run_info.json: Run metadata\n")
        f.write("  - metrics_per_fold.csv: Detailed metrics for each fold\n")
        f.write("  - metrics_summary.json: Aggregated statistics\n")
        f.write(
            f"  - {metric_key}_per_fold.png: Visualization of metric across folds\n"
        )
        f.write("  - fold_*/: Individual fold results and visualizations\n")

    print(f"✓ Results saved to: {outdir}")
    print(f"✓ Summary: {readme_path}")


if __name__ == "__main__":
    main()
