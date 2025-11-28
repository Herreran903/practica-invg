"""
Command-line interface for JSSP Tensors data generation.

This module provides a CLI for generating JSSP datasets with 2D tensor/matrix
representations. It supports both academic and generated modes.
"""

import argparse
import sys
from pathlib import Path

from .config_loader import load_config
from .prepare_academic_dataset import prepare_academic_dataset
from .prepare_generated_dataset import prepare_generated_dataset
from .tensor_converter import convert_dataset_to_tensors


def main():
    """
    Main entry point for the JSSP Tensors data generation CLI.
    
    Supports two modes:
    - academic: Uses JSPLIB benchmark instances
    - generated: Creates random balanced instances
    
    Both modes:
    1. Prepare dataset (run solvers, generate ground truth CSV)
    2. Convert instances to 2D tensors (.npy files)
    """
    parser = argparse.ArgumentParser(
        description="JSSP Tensors Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Generate academic dataset using default config
            python -m src.data_generation.jssp_tensors.cli --mode academic

            # Generate random instances with custom config
            python -m src.data_generation.jssp_tensors.cli --mode generated --config my_config.yaml

            # Only convert existing dataset to tensors (skip solver execution)
            python -m src.data_generation.jssp_tensors.cli --mode academic --skip-solvers --csv path/to/dataset.csv

            # Disable standardization
            python -m src.data_generation.jssp_tensors.cli --mode generated --no-standardize
                    """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["academic", "generated"],
        help="Dataset generation mode: 'academic' (JSPLIB benchmarks) or 'generated' (random instances)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/data_generation/jssp_tensors/config.yaml",
        help="Path to configuration YAML file relative to project root (default: src/data_generation/jssp_tensors/config.yaml)"
    )

    
    parser.add_argument(
        "--skip-solvers",
        action="store_true",
        help="Skip solver execution and only convert existing dataset to tensors"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to existing CSV file (required if --skip-solvers is used)"
    )
    
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable z-score standardization of tensors"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_solvers and not args.csv:
        parser.error("--csv is required when --skip-solvers is used")
    
    # Load configuration
    try:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        print("Configuration loaded successfully\n")
        # Diagnostic: show resolved paths to validate project_root and outputs
        print("Resolved paths:")
        print(f"  project_root: {config.project_root}")
        print(f"  cp_model_path: {config.cp_model_path} (exists={Path(config.cp_model_path).exists()})")
        print(f"  mip_model_path: {config.mip_model_path} (exists={Path(config.mip_model_path).exists() if config.mip_model_path else 'n/a'})")
        print(f"  academic_output_dir: {config.academic_output_dir}")
        print(f"  generated_output_dir: {config.generated_output_dir}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine standardization setting
    standardize = not args.no_standardize
    
    # Execute pipeline
    try:
        if args.skip_solvers:
            # Only convert to tensors
            csv_path = args.csv
            print(f"Skipping solver execution. Using existing CSV: {csv_path}\n")
        else:
            # Run solvers and generate dataset
            if args.mode == "academic":
                print("=" * 75)
                print("MODE: ACADEMIC (JSPLIB Benchmarks) - TENSORS")
                print("=" * 75)
                print()
                csv_path = prepare_academic_dataset(config)
            
            elif args.mode == "generated":
                print("=" * 75)
                print("MODE: GENERATED (Random Instances) - TENSORS")
                print("=" * 75)
                print()
                csv_path = prepare_generated_dataset(config)
            
            else:
                print(f"ERROR: Unknown mode '{args.mode}'", file=sys.stderr)
                sys.exit(1)
            
            print()
        
        # Convert to tensors
        print("=" * 75)
        print("STEP 2: Converting instances to 2D tensors")
        print("=" * 75)
        print()
        
        convert_dataset_to_tensors(csv_path, standardize=standardize)
        
        # Success summary
        print()
        print("=" * 75)
        print("✓ PIPELINE COMPLETE")
        print("=" * 75)
        print(f"Mode: {args.mode}")
        print(f"Output directory: {Path(csv_path).parent}")
        print(f"CSV file: {Path(csv_path).name}")
        print(f"Tensors directory: {Path(csv_path).parent / 'images'}")
        print(f"Standardization: {'enabled' if standardize else 'disabled'}")
        print()
        print("Files generated:")
        print(f"  - Ground truth CSV: {csv_path}")
        print(f"  - Instance files (.dzn): {Path(csv_path).parent}")
        print(f"  - Tensor files (.npy): {Path(csv_path).parent / 'images'}")
        print()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...", file=sys.stderr)
        sys.exit(130)
    
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()