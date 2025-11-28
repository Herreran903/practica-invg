"""
Command-line interface for SAT Images data generation.

This module provides a CLI for generating SAT datasets with grayscale image
representations from ASlib scenarios.
"""

import argparse
import sys
from pathlib import Path

from .config_loader import load_config
from .image_converter import convert_dataset_to_images
from .prepare_aslib_dataset import prepare_aslib_dataset


def main():
    """
    Main entry point for the SAT Images data generation CLI.
    
    Processes ASlib scenarios to generate:
    1. Ground truth CSV with solver performance data
    2. Grayscale images (.npy files) from SAT instance files
    """
    parser = argparse.ArgumentParser(
        description="SAT Images Data Generation Pipeline (ASlib)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process an ASlib scenario
  python -m src.data_generation.sat_images.cli \\
    --scenario-dir data/sat/aslib/sc2012-application \\
    --instances-dir data/sat/instances/sc2012-application

  # With custom output directory
  python -m src.data_generation.sat_images.cli \\
    --scenario-dir data/sat/aslib/sc2012-application \\
    --instances-dir data/sat/instances/sc2012-application \\
    --output-dir data/sat/datasets/my_custom_output

  # Only convert existing dataset to images (skip ASlib processing)
  python -m src.data_generation.sat_images.cli \\
    --skip-aslib \\
    --csv data/sat/datasets/sat_cnn_data_gen/ground_truth_aslib.csv \\
    --instances-dir data/sat/instances/sc2012-application

  # With custom image size
  python -m src.data_generation.sat_images.cli \\
    --scenario-dir data/sat/aslib/sc2012-application \\
    --instances-dir data/sat/instances/sc2012-application \\
    --image-size 256
        """
    )
    
    parser.add_argument(
        "--scenario-dir",
        type=str,
        help="Path to ASlib scenario directory (must contain algorithm_runs.arff)"
    )
    
    parser.add_argument(
        "--instances-dir",
        type=str,
        required=True,
        help="Path to directory containing raw instance files (CNF, XCSP, DZN, etc.)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for generated files (default: from config)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/data_generation/sat_images/config.yaml",
        help="Path to configuration YAML file relative to project root (default: src/data_generation/sat_images/config.yaml)"
    )
    
    parser.add_argument(
        "--skip-aslib",
        action="store_true",
        help="Skip ASlib processing and only convert existing dataset to images"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to existing CSV file (required if --skip-aslib is used)"
    )
    
    parser.add_argument(
        "--instance-map-csv",
        type=str,
        help="Optional CSV with instance_id,file_path columns for explicit path mapping"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        help="Timeout in seconds (overrides value from description.txt or config)"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        help="Override target image size from config (default: 128)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_aslib:
        if not args.csv:
            parser.error("--csv is required when --skip-aslib is used")
    else:
        if not args.scenario_dir:
            parser.error("--scenario-dir is required unless --skip-aslib is used")
    
    # Load configuration
    try:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        print("Configuration loaded successfully\n")
        # Diagnostic: show resolved paths to validate project_root and outputs
        print("Resolved paths:")
        print(f"  project_root: {config.project_root}")
        print(f"  default_output_dir: {config.default_output_dir}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine parameters
    image_size = args.image_size if args.image_size else config.image_target_size
    output_dir = args.output_dir if args.output_dir else config.default_output_dir
    
    # Execute pipeline
    try:
        if args.skip_aslib:
            # Only convert to images
            csv_path = args.csv
            print(f"Skipping ASlib processing. Using existing CSV: {csv_path}\n")
        else:
            # Process ASlib scenario
            print("=" * 75)
            print("STEP 1: Processing ASlib Scenario")
            print("=" * 75)
            print()
            
            import os
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, config.ground_truth_csv_name)
            
            csv_path = prepare_aslib_dataset(
                scenario_dir=args.scenario_dir,
                out_csv=csv_path,
                instances_dir=args.instances_dir,
                instance_map_csv=args.instance_map_csv,
                timeout_s=args.timeout,
                default_timeout_s=config.default_timeout_s
            )
            
            print()
        
        # Convert to images
        print("=" * 75)
        print("STEP 2: Converting instances to grayscale images")
        print("=" * 75)
        print()
        
        convert_dataset_to_images(
            csv_path=csv_path,
            instances_root=args.instances_dir,
            target_size=image_size,
            prefix_map=config.prefix_map
        )
        
        # Success summary
        print()
        print("=" * 75)
        print("✓ PIPELINE COMPLETE")
        print("=" * 75)
        print(f"Output directory: {Path(csv_path).parent}")
        print(f"CSV file: {Path(csv_path).name}")
        print(f"Images directory: {Path(csv_path).parent / 'images'}")
        print(f"Image size: {image_size}×{image_size}")
        print()
        print("Files generated:")
        print(f"  - Ground truth CSV: {csv_path}")
        print(f"  - Image files (.npy): {Path(csv_path).parent / 'images'}")
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