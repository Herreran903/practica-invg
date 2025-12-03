# -*- coding: utf-8 -*-
"""
CLI for SAT image-based solver selection training.

Extends jssp_images CLI with SAT-specific features:
- K-Fold repetitions (--repeats)
- Solver filtering (--solvers)
- Time limit configuration (--time_limit)
- Feature extraction time column (--feat_time_col)
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from .config_loader import load_config, merge_cli_args, resolve_paths
from .data_utils import detect_solver_cols, filter_valid_images, normalize_image_paths
from .training_loop import run_kfold_with_repeats
from .visualization import plot_metrics_per_fold


def setup_reproducibility(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train CNN for SAT solver selection from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.training.sat_images.cli --csv data.csv --task classification
  python -m src.training.sat_images.cli --csv data.csv --task classification --folds 5 --repeats 5
  python -m src.training.sat_images.cli --csv data.csv --task multilabel --time_limit 1800
        """,
    )

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "multilabel"],
    )
    parser.add_argument(
        "--config", type=str, default="src/training/sat_images/config.yaml"
    )
    parser.add_argument("--use_score", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument(
        "--repeats", type=int, help="K-Fold repetitions (e.g., 5 for 5x5)"
    )
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument(
        "--time_limit", type=float, help="Time limit in seconds (default: 1800)"
    )
    parser.add_argument(
        "--feat_time_col", type=str, help="Feature extraction time column"
    )
    parser.add_argument(
        "--solvers", type=str, help="Comma-separated solver names to use"
    )
    parser.add_argument("--out_parent", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--seed", type=int)

    return parser.parse_args()


def prepare_output_directory(config: dict, task: str) -> str:
    """Create output directory."""
    output_cfg = config.get("output", {})
    parent_dir = output_cfg.get("parent_dir", "training/sat/results")
    run_name = output_cfg.get("run_name", "sat_images_cnn")
    timestamp_fmt = output_cfg.get("timestamp_format", "%Y%m%d_%H%M%S")
    append_task = output_cfg.get("append_task_to_dirname", True)

    timestamp = datetime.now().strftime(timestamp_fmt)
    run_dirname = (
        f"{run_name}_{task}_{timestamp}" if append_task else f"{run_name}_{timestamp}"
    )
    outdir = os.path.join(parent_dir, run_dirname)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def save_run_info(
    outdir: str, config: dict, args: argparse.Namespace, df: pd.DataFrame
) -> None:
    """Save run information."""
    import json

    import yaml

    with open(os.path.join(outdir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

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
    """Main CLI entry point."""
    args = parse_arguments()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    config = merge_cli_args(config, args)

    # Apply CLI overrides
    if args.time_limit is not None:
        config.setdefault("data", {})["time_limit_s"] = args.time_limit
    if args.feat_time_col is not None:
        config.setdefault("data", {})["feat_time_column"] = args.feat_time_col
    if args.repeats is not None:
        config.setdefault("training", {})["k_fold_repeats"] = args.repeats

    config = resolve_paths(config)

    seed = config.get("training", {}).get("seed", 42)
    setup_reproducibility(seed)
    print(f"Random seed: {seed}")

    print(f"\nLoading CSV: {args.csv}")
    if not os.path.exists(args.csv):
        print(f"❌ CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")

    if "Image_Npy_Path" not in df.columns:
        print("❌ CSV must contain 'Image_Npy_Path'")
        sys.exit(1)

    print("\nNormalizing image paths...")
    df = normalize_image_paths(df, "Image_Npy_Path")

    print("Filtering valid images...")
    df, missing_count, _total_rows = filter_valid_images(df)

    if missing_count > 0:
        print(f"⚠️  Removed {missing_count} rows with missing images")

    if len(df) == 0:
        print("❌ No valid images found")
        sys.exit(1)

    print(f"✓ {len(df)} valid instances")

    print("\nDetecting solver columns...")
    try:
        solver_cols = detect_solver_cols(df)
        print(f"✓ {len(solver_cols['runtime'])} runtime columns")
        if solver_cols["score"]:
            print(f"✓ {len(solver_cols['score'])} score columns")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Optional solver filtering
    if args.solvers:
        selected = [s.strip() for s in args.solvers.split(",") if s.strip()]
        print(f"\nFiltering to solvers: {selected}")

        keep_runtime = [f"{s}_Runtime_s" for s in selected]
        keep_score = [
            f"{s}_Score_S_rel" for s in selected if f"{s}_Score_S_rel" in df.columns
        ]
        keep_status = [f"{s}_Status" for s in selected if f"{s}_Status" in df.columns]

        base_cols = ["Image_Npy_Path"]
        df = df[base_cols + keep_runtime + keep_score + keep_status]
        solver_cols = {"runtime": keep_runtime, "score": keep_score}

        print(f"✓ Filtered to {len(keep_runtime)} solvers")

    # Setup feature time column
    feat_time_col = config.get("data", {}).get("feat_time_column")
    if feat_time_col and feat_time_col not in df.columns:
        print(f"⚠️  Feature time column '{feat_time_col}' not found, using 0")
        df[feat_time_col] = 0.0
    elif not feat_time_col:
        df["_feat_time_zero_"] = 0.0
        config.setdefault("data", {})["feat_time_column"] = "_feat_time_zero_"

    outdir = prepare_output_directory(config, args.task)
    print(f"\n📁 Output: {outdir}")

    save_run_info(outdir, config, args, df)

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Use score: {args.use_score}")
    print(f"Epochs: {training_cfg.get('epochs', 25)}")
    print(f"Batch size: {training_cfg.get('batch_size', 64)}")
    print(f"K-Folds: {training_cfg.get('k_folds', 5)}")
    print(f"Repeats: {training_cfg.get('k_fold_repeats', 1)}")
    print(f"Time limit: {data_cfg.get('time_limit_s', 1800.0)}s")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        all_results, summary = run_kfold_with_repeats(
            df=df,
            task=args.task,
            solver_cols=solver_cols,
            use_score=args.use_score,
            config=config,
            root_outdir=outdir,
        )
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    elapsed_time = time.time() - start_time

    metric_key = summary["metric"]

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Metric: {metric_key.upper()}")
    print(f"Mean: {summary['mean']:.4f}")
    print(f"Std: {summary['std']:.4f}")
    print(f"Range: [{summary['min']:.4f}, {summary['max']:.4f}]")
    print(f"Folds: {summary['folds']}x{summary['repeats']}")
    print(f"Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
    print(f"{'='*60}\n")

    readme_path = os.path.join(outdir, "README.txt")
    with open(readme_path, "w") as f:
        f.write("SAT Image-Based Solver Selection - Training Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Use score: {args.use_score}\n")
        f.write(f"K-Folds: {summary['folds']}x{summary['repeats']}\n")
        f.write(f"Metric: {metric_key.upper()}\n")
        f.write(f"Mean: {summary['mean']:.4f}\n")
        f.write(f"Std: {summary['std']:.4f}\n")
        f.write(f"Range: [{summary['min']:.4f}, {summary['max']:.4f}]\n")
        f.write(f"Time limit: {data_cfg.get('time_limit_s', 1800.0)}s\n")
        f.write(f"Training time: {elapsed_time:.2f}s\n\n")
        f.write("Files:\n")
        f.write("  - config.yaml\n")
        f.write("  - run_info.json\n")
        f.write("  - metrics_summary_GLOBAL.json\n")
        f.write("  - metrics_summary_per_rep.csv\n")
        f.write("  - rep_*/\n")

    print(f"✓ Results: {outdir}")
    print(f"✓ Summary: {readme_path}")


if __name__ == "__main__":
    main()
