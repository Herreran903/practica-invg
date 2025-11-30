"""
ASlib dataset preparation for SAT problems.

This module handles the preparation of datasets from ASlib scenarios,
processing algorithm_runs.arff files and generating ground truth CSVs.
"""

import csv
import os
from typing import Optional, Dict

from .aslib_parser import (
    build_pivot_runtime_table,
    compute_winner_key,
    load_algorithm_runs_dataframe,
    try_read_timeout_from_description,
)
from .config_loader import SATImagesConfig
from .instance_resolver import (
    build_instance_path_map,
    load_instance_map_csv,
    resolve_raw_text_path,
    resolve_path_with_prefix_map,
)


def prepare_aslib_dataset(
    scenario_dir: str,
    out_csv: str,
    instances_dir: Optional[str] = None,
    instance_map_csv: Optional[str] = None,
    timeout_s: Optional[float] = None,
    default_timeout_s: float = 5000.0,
    prefix_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate a ground truth CSV from an ASlib scenario.

    Steps:
    1. Load algorithm_runs.arff and normalize columns
    2. Read description.txt to infer timeout (if not provided)
    3. Pivot runtimes/statuses by instance×solver and compute Winner_Key
    4. Resolve raw instance file paths (by ID or filename)
    5. Write final CSV with base columns, *_Runtime_s and *_Status

    Args:
        scenario_dir: ASlib scenario directory (contains ARFF and description.txt)
        out_csv: Output CSV path
        instances_dir: Directory with raw instance files (for filename mapping)
        instance_map_csv: CSV with instance_id,file_path columns (for ID mapping)
        timeout_s: Timeout to use; if None, tries description.txt, then default
        default_timeout_s: Default timeout if not found elsewhere
        prefix_map: Optional mapping used to resolve instance IDs to paths,
            mirroring the behavior of the image conversion step. If provided,
            it is consulted as a fallback when direct filename/ID resolution
            fails, ensuring Raw_Text_Path is fully populated before image
            generation.

    Returns:
        Absolute path to generated CSV

    Raises:
        FileNotFoundError: If algorithm_runs.arff doesn't exist
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # Validate algorithm_runs.arff exists
    arff_path = os.path.join(scenario_dir, "algorithm_runs.arff")
    if not os.path.exists(arff_path):
        raise FileNotFoundError(f"algorithm_runs.arff not found in {scenario_dir}")

    print("=" * 75)
    print("ASlib Dataset Preparation")
    print("=" * 75)
    print(f"Scenario: {os.path.abspath(scenario_dir)}")
    print(f"ARFF file: {arff_path}")

    # Load and normalize runs
    print("\n[1/3] Loading solver runs...")
    runs_df = load_algorithm_runs_dataframe(arff_path)

    # Infer timeout
    desc_path = os.path.join(scenario_dir, "description.txt")
    timeout_from_desc = try_read_timeout_from_description(desc_path)
    timeout_used = float(
        timeout_s
        if timeout_s is not None
        else (timeout_from_desc if timeout_from_desc else default_timeout_s)
    )
    print(f"[2/3] Timeout used: {timeout_used:.0f}s")

    # Build pivot table
    pivot_df, runtime_cols = build_pivot_runtime_table(runs_df, timeout_used)
    print(
        f"[2/3] Pivot complete: {pivot_df.shape[0]} instances × "
        f"{len(runtime_cols)} solvers"
    )

    # Resolve raw instance paths
    print("[3/3] Resolving instance file paths...")
    map_by_filename = build_instance_path_map(instances_dir)
    map_by_id = load_instance_map_csv(instance_map_csv)

    raw_paths = []
    for _, row in pivot_df.iterrows():
        instance_id = str(row["instance_id"])

        # 1) Try explicit ID/filename mapping (uncompressed preferred)
        path = resolve_raw_text_path(instance_id, map_by_filename, map_by_id)

        # 2) Fallback: use prefix-based resolution (same logic as image converter)
        if (not path or not os.path.exists(str(path))) and prefix_map and instances_dir:
            path = resolve_path_with_prefix_map(instances_dir, instance_id, prefix_map)

        # Validate path exists
        if path and os.path.exists(str(path)):
            raw_paths.append(os.path.abspath(str(path)))
        else:
            raw_paths.append("")

    # Enrich output columns
    pivot_df["Raw_Text_Path"] = raw_paths
    pivot_df["Time_Limit_s"] = timeout_used
    pivot_df["Winner_Key"] = [
        compute_winner_key(row, runtime_cols, timeout_used)
        for _, row in pivot_df.iterrows()
    ]
    pivot_df["Instance_Id"] = pivot_df["instance_id"]
    pivot_df["Instance_Name"] = pivot_df["instance_id"].apply(
        lambda s: os.path.splitext(os.path.basename(str(s)))[0]
    )

    # Define column order
    base_cols = [
        "Instance_Id",
        "Instance_Name",
        "Raw_Text_Path",
        "Time_Limit_s",
        "Winner_Key",
    ]
    status_cols = [c for c in pivot_df.columns if c.endswith("_Status")]
    final_cols = base_cols + runtime_cols + status_cols

    # Write CSV
    print("Writing CSV output...")
    pivot_df[final_cols].to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Ground truth saved: {out_csv}")
    print("=" * 75)
    print("ASlib Dataset Preparation Complete")
    print("=" * 75)

    return out_csv


def prepare_aslib_dataset_with_config(
    config: SATImagesConfig,
    scenario_dir: str,
    instances_dir: str,
    output_dir: Optional[str] = None,
    instance_map_csv: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> str:
    """
    Prepare ASlib dataset using configuration object.

    Args:
        config: Configuration object
        scenario_dir: ASlib scenario directory
        instances_dir: Directory with raw instance files
        output_dir: Output directory (uses config default if None)
        instance_map_csv: Optional CSV mapping instance IDs to paths
        timeout_s: Optional timeout override

    Returns:
        Path to generated CSV
    """
    # Use config defaults if not provided
    if output_dir is None:
        output_dir = config.default_output_dir

    if timeout_s is None:
        timeout_s = config.default_timeout_s

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, config.ground_truth_csv_name)

    return prepare_aslib_dataset(
        scenario_dir=scenario_dir,
        out_csv=out_csv,
        instances_dir=instances_dir,
        instance_map_csv=instance_map_csv,
        timeout_s=timeout_s,
        default_timeout_s=config.default_timeout_s,
        prefix_map=config.prefix_map,
    )
