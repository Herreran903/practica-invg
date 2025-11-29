"""
Generated dataset preparation for JSSP.

This module handles the preparation of datasets using randomly generated JSSP
instances. It creates balanced instances and benchmarks multiple solvers with
different time limits and random seeds.
"""

import csv
import os
from typing import Dict, List

from .config_loader import JSPPImagesConfig
from .jssp_instance_utils import generate_random_instance, save_instance_as_dzn
from .minizinc_solver import execute_minizinc_solver, filter_solver_candidates


def prepare_generated_dataset(config: JSPPImagesConfig) -> str:
    """
    Prepare generated JSSP dataset using random balanced instances.

    This function:
    1. Generates random JSSP instances with specified dimensions
    2. Converts them to .dzn format
    3. Runs multiple solvers with different time limits and seeds
    4. Identifies the best solver for each configuration
    5. Generates a CSV with ground truth data

    The CSV includes columns for each solver's runtime, makespan, and wall time,
    plus a "Winner_Key" column indicating which solver performed best.

    Args:
        config: Configuration object with all settings

    Returns:
        Path to the generated CSV file

    Raises:
        RuntimeError: If dataset preparation fails
    """
    output_dir = config.generated_output_dir
    csv_name = config.generated_csv_name
    cp_model_path = config.cp_model_path
    mip_model_path = config.mip_model_path
    time_limits_ms = config.generated_time_limits_ms
    solver_candidates = config.solver_candidates
    random_seeds = config.generated_random_seeds
    generation_cases = config.generated_cases
    duration_range = config.instance_duration_range
    generator_seed = config.instance_generation_seed

    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "solver_logs_temp")
    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 75)
    print("GENERATED DATASET PREPARATION (Random Instances)")
    print("=" * 75)

    # Generate all instances
    print("\n[1/3] Generating random JSSP instances...")
    print(f"Duration range: {duration_range}")
    print(f"Generator seed: {generator_seed}")

    all_instances = []
    instance_count = 0

    for num_jobs, num_machines, count in generation_cases:
        print(f"  Generating {count} instances of size {num_jobs}x{num_machines}...")

        for i in range(count):
            instance = generate_random_instance(
                num_jobs=num_jobs,
                num_machines=num_machines,
                duration_range=duration_range,
                generator_seed=generator_seed,
            )
            instance.name = f"GEN_{num_jobs}x{num_machines}_{i + 1}"
            all_instances.append(instance)
            instance_count += 1

    print(f"Total instances generated: {instance_count}")
    print("-" * 75)

    # Filter available solvers
    print("\n[2/3] Configuring available solvers...")
    solver_configs = filter_solver_candidates(solver_candidates, mip_model_path)
    config_keys = [key for _, key, _, _ in solver_configs]

    print(f"Available solvers: {', '.join(config_keys)}")
    print("-" * 75)

    # Prepare CSV header
    header = [
        "Instance_Name",
        "Raw_Text_Path",
        "N_Jobs",
        "N_Machines",
        "Time_Limit_s",
        "Seed",
        "Winner_Key",
    ]
    for key in config_keys:
        header += [f"{key}_Runtime_s", f"{key}_Makespan", f"{key}_Wall_s"]

    csv_rows: List[List[str]] = [header]

    # Run benchmarking
    print("\n[3/3] Running solver benchmarking...")
    print(f"Time limits: {[t/1000 for t in time_limits_ms]}s")
    print(f"Random seeds: {random_seeds}")
    print("-" * 75)

    for idx, instance in enumerate(all_instances):
        instance_name = instance.name
        print(f"\nInstance {idx + 1}/{instance_count}: {instance_name}")
        print(f"  Size: {instance.num_jobs}x{instance.num_machines}")

        try:
            # Save instance as .dzn
            dzn_path = save_instance_as_dzn(instance, instance_name, output_dir)
            print(f"  .dzn file: {os.path.basename(dzn_path)}")

            # Benchmark across all time limits and seeds
            for time_limit_ms in time_limits_ms:
                time_limit_s = time_limit_ms / 1000.0

                for seed in random_seeds:
                    print(f"\n  Configuration: {time_limit_s:.0f}s, seed={seed}")

                    results: Dict[str, Dict] = {}

                    # Run all solver configurations
                    for solver_id, key, solver_type, opts in solver_configs:
                        print(f"    Running {key}...", end=" ")

                        stats = execute_minizinc_solver(
                            solver_id=solver_id,
                            key=key,
                            solver_type=solver_type,
                            options=opts,
                            dzn_path=dzn_path,
                            time_limit_ms=time_limit_ms,
                            cp_model_path=cp_model_path,
                            mip_model_path=mip_model_path,
                            temp_dir=temp_dir,
                        )

                        stats["seed"] = seed
                        results[key] = stats

                        # Display result
                        makespan_str = (
                            f"{int(stats['makespan'])}"
                            if stats["makespan"] < float("inf")
                            else "inf"
                        )
                        solved = "✓" if stats["solved_binary"] == 1 else "✗"
                        print(
                            f"{solved} makespan={makespan_str}, "
                            f"runtime={stats['runtime']:.3f}s"
                        )

                    # Determine winner (solver with best runtime among those that solved)
                    solved_keys = [
                        k for k, s in results.items() if s.get("solved_binary", 0) == 1
                    ]

                    if solved_keys:
                        winner_key = min(
                            solved_keys, key=lambda k: results[k]["runtime"]
                        )
                        best_runtime = results[winner_key]["runtime"]
                        print(
                            f"    Winner: {winner_key} "
                            f"({len(solved_keys)}/{len(solver_configs)} solved, "
                            f"best runtime: {best_runtime:.3f}s)"
                        )
                    else:
                        winner_key = "NONE"
                        print(f"    Winner: NONE (0/{len(solver_configs)} solved)")

                    # Build CSV row
                    row = [
                        instance_name,
                        dzn_path,
                        str(instance.num_jobs),
                        str(instance.num_machines),
                        f"{time_limit_s:.0f}",
                        str(seed),
                        winner_key,
                    ]

                    for key in config_keys:
                        st = results[key]
                        makespan = (
                            st["makespan"] if st["makespan"] != float("inf") else "inf"
                        )
                        row += [
                            f"{st['runtime']:.3f}",
                            makespan,
                            f"{st['wall_time_s']:.3f}",
                        ]

                    csv_rows.append(row)

            print(f"  Instance complete ✓")

        except Exception as e:
            print(f"  ERROR: Failed to process instance '{instance_name}'")
            print(f"  Details: {e}")

        print("-" * 75)

    # Write CSV
    csv_path = os.path.join(output_dir, csv_name)
    print(f"\nWriting results to CSV: {csv_path}")

    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"Results saved successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to write CSV file '{csv_path}'. Error: {e}")

    print("\n" + "=" * 75)
    print("GENERATED DATASET PREPARATION COMPLETE")
    print("=" * 75)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"CSV file: {csv_name}")
    print(f"Total rows: {len(csv_rows) - 1}")  # Exclude header

    return csv_path
