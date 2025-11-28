"""
Academic dataset preparation for JSSP.

This module handles the preparation of datasets using benchmark JSSP instances
from JSPLIB. It runs MiniZinc solvers on fixed instances and generates ground
truth data for machine learning training.
"""

import csv
import os
import time
from typing import Dict, List, Tuple

from .config_loader import JSPPImagesConfig
from .jssp_instance_utils import (
    get_instance_optimum,
    load_academic_instance,
    save_instance_as_dzn
)
from .minizinc_solver import (
    calculate_relative_performance_score,
    execute_minizinc_solver
)


def prepare_academic_dataset(config: JSPPImagesConfig) -> str:
    """
    Prepare academic JSSP dataset using JSPLIB benchmark instances.
    
    This function:
    1. Loads benchmark instances from JSPLIB
    2. Converts them to .dzn format
    3. Runs configured solvers on each instance
    4. Calculates performance scores
    5. Generates a CSV with ground truth data
    
    Args:
        config: Configuration object with all settings
        
    Returns:
        Path to the generated CSV file
        
    Raises:
        RuntimeError: If dataset preparation fails
    """
    output_dir = config.academic_output_dir
    csv_name = config.academic_csv_name
    instance_names = config.academic_instances
    model_path = config.cp_model_path
    time_limit_ms = config.academic_time_limit_ms
    penalty_factor = config.academic_penalty_factor
    solver_strategies = config.academic_solver_strategies
    
    os.makedirs(output_dir, exist_ok=True)
    time_limit_s = time_limit_ms / 1000.0
    
    print("=" * 75)
    print("ACADEMIC DATASET PREPARATION (JSPLIB Benchmarks)")
    print("=" * 75)
    print(f"Instances: {', '.join(instance_names)}")
    print(f"Time limit: {time_limit_s:.0f}s per solver/instance")
    print(f"Penalty factor (K): {penalty_factor}")
    print(f"Output directory: {output_dir}")
    print("-" * 75)
    
    # Prepare CSV header
    solver_keys = [key for _, _, key in solver_strategies]
    header = [
        "Instance_Name",
        "Raw_Text_Path",
        "N_Jobs",
        "N_Machines",
        "Best_Makespan_Found",
        "Optimum"
    ]
    header += [f"{key}_Runtime_s" for key in solver_keys]
    header += [f"{key}_Score_S_rel" for key in solver_keys]
    
    csv_rows: List[List[str]] = [header]
    
    print("\n[1/2] Running solvers and collecting metrics...")
    print("-" * 75)
    
    total_instances = len(instance_names)
    
    for idx, instance_name in enumerate(instance_names):
        t_start = time.time()
        print(f"\nInstance {idx + 1}/{total_instances}: {instance_name}")
        
        try:
            # Load benchmark instance
            instance = load_academic_instance(instance_name)
            
            # Save as .dzn file
            dzn_path = save_instance_as_dzn(instance, instance_name, output_dir)
            
            # Get known optimum
            optimum = get_instance_optimum(instance)
            if optimum == float("inf"):
                print(f"  Warning: No known optimum. Using 1.0 for score calculation.")
                optimum = 1.0
            
            print(f"  Dimensions: {instance.num_jobs}x{instance.num_machines}")
            print(f"  Known optimum: {optimum:.0f}")
            print(f"  .dzn file: {os.path.basename(dzn_path)}")
            
            # Run all configured solvers
            all_results: Dict[str, Dict] = {}
            best_makespan = float("inf")
            
            for solver_id, strategy, key in solver_strategies:
                print(f"\n  Solver: {key}")
                
                # Execute solver
                stats = execute_minizinc_solver(
                    solver_id=solver_id,
                    key=key,
                    solver_type="cp",  # Academic mode uses CP solvers
                    options={"strategy": strategy, "inject_search": False},
                    dzn_path=dzn_path,
                    time_limit_ms=time_limit_ms,
                    cp_model_path=model_path,
                    mip_model_path=None,
                    temp_dir=os.path.join(output_dir, "temp")
                )
                
                # Calculate performance score
                stats["score"] = calculate_relative_performance_score(
                    stats, optimum, time_limit_s, penalty_factor
                )
                
                all_results[key] = stats
                
                # Display results
                makespan_str = (
                    f"{stats['makespan']:.0f}"
                    if stats['makespan'] < float("inf")
                    else "inf"
                )
                print(f"    Makespan: {makespan_str}")
                print(f"    Runtime: {stats['runtime']:.3f}s")
                print(f"    Score: {stats['score']:.2f}")
                
                # Track best makespan
                if stats['makespan'] < best_makespan:
                    best_makespan = stats['makespan']
            
            # Format best makespan for CSV
            best_makespan_str = (
                f"{best_makespan:.0f}"
                if best_makespan < float("inf")
                else "inf"
            )
            optimum_str = f"{optimum:.0f}" if optimum < float("inf") else "inf"
            
            # Build CSV row
            row = [
                instance_name,
                dzn_path,
                str(instance.num_jobs),
                str(instance.num_machines),
                best_makespan_str,
                optimum_str
            ]
            row += [f"{all_results[key]['runtime']:.3f}" for key in solver_keys]
            row += [f"{all_results[key]['score']:.2f}" for key in solver_keys]
            
            csv_rows.append(row)
            
            elapsed = time.time() - t_start
            print(f"\n  Instance complete. Best makespan: {best_makespan_str}")
            print(f"  Total time: {elapsed:.3f}s")
            print("-" * 75)
        
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"\n  ERROR: Failed to process instance '{instance_name}'")
            print(f"  Details: {e}")
            print(f"  Time elapsed: {elapsed:.3f}s")
            print("-" * 75)
    
    # Write CSV
    csv_path = os.path.join(output_dir, csv_name)
    print(f"\n[2/2] Writing results to CSV...")
    
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"Results saved: {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to write CSV file '{csv_path}'. Error: {e}")
    
    print("\n" + "=" * 75)
    print("ACADEMIC DATASET PREPARATION COMPLETE")
    print("=" * 75)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"CSV file: {csv_name}")
    print(f"Total instances: {len(csv_rows) - 1}")  # Exclude header
    
    return csv_path