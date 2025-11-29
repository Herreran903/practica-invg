"""
Utilities for handling JSSP instances.

This module provides functions to load, generate, and save JSSP instances
in the format required by MiniZinc (.dzn files).
"""

import os
from typing import List

try:
    from job_shop_lib import JobShopInstance
    from job_shop_lib.benchmarking import load_benchmark_instance
    from job_shop_lib.generation import GeneralInstanceGenerator
except ImportError as e:
    raise ImportError(
        "FATAL ERROR: Cannot import job_shop_lib. "
        "Please ensure 'job-shop-lib' is installed: pip install job-shop-lib"
    ) from e


def load_academic_instance(instance_name: str) -> JobShopInstance:
    """
    Load a benchmark JSSP instance from JSPLIB.

    Args:
        instance_name: Name of the benchmark instance (e.g., "ft06", "la01")

    Returns:
        JobShopInstance object

    Raises:
        ValueError: If instance cannot be loaded
    """
    try:
        return load_benchmark_instance(instance_name)
    except Exception as e:
        raise ValueError(
            f"Failed to load benchmark instance '{instance_name}'. "
            f"Ensure it exists in JSPLIB. Error: {e}"
        )


def generate_random_instance(
    num_jobs: int, num_machines: int, duration_range: tuple, generator_seed: int
) -> JobShopInstance:
    """
    Generate a random balanced JSSP instance.

    Args:
        num_jobs: Number of jobs
        num_machines: Number of machines
        duration_range: Tuple (min_duration, max_duration) for operations
        generator_seed: Random seed for reproducibility

    Returns:
        JobShopInstance object
    """
    generator = GeneralInstanceGenerator(
        duration_range=duration_range, seed=generator_seed
    )
    return generator.generate(num_jobs=num_jobs, num_machines=num_machines)


def save_instance_as_dzn(
    instance: JobShopInstance, instance_name: str, output_dir: str
) -> str:
    """
    Convert a JobShopInstance to a MiniZinc .dzn file.

    The .dzn file contains:
    - JOBS: number of jobs
    - MACHINES: number of machines
    - PROC_TIME: 2D array of processing times (flattened row-major)
    - MACHINE_OF_OP: 2D array of machine assignments (1-indexed, flattened row-major)

    Args:
        instance: JobShopInstance object to convert
        instance_name: Name for the output file (without extension)
        output_dir: Directory where the .dzn file will be saved

    Returns:
        Absolute path to the created .dzn file

    Raises:
        RuntimeError: If file writing fails
    """
    os.makedirs(output_dir, exist_ok=True)
    dzn_path = os.path.join(output_dir, f"{instance_name}.dzn")

    # Extract processing times and machine assignments
    proc_time_rows: List[List[int]] = []
    machine_of_op_rows: List[List[int]] = []

    try:
        for job_ops in instance.jobs:
            pt_row: List[int] = []
            mach_row: List[int] = []

            for op in job_ops:
                # Processing time
                pt_row.append(op.duration)

                # Machine ID (convert from 0-indexed to 1-indexed for MiniZinc)
                machine_info = op.machines[0]
                if isinstance(machine_info, int):
                    machine_id_0_based = machine_info
                elif hasattr(machine_info, "id"):
                    machine_id_0_based = machine_info.id
                else:
                    raise AttributeError(
                        f"Cannot determine machine ID for operation {op}. "
                        f"Expected int or object with 'id' attribute."
                    )

                mach_row.append(machine_id_0_based + 1)  # Convert to 1-indexed

            proc_time_rows.append(pt_row)
            machine_of_op_rows.append(mach_row)

    except Exception as e:
        raise RuntimeError(
            f"Failed to extract data from instance '{instance_name}'. Error: {e}"
        )

    # Helper function to flatten 2D array for MiniZinc format
    def flatten_array(matrix_2d: List[List[int]]) -> str:
        """Flatten a 2D matrix into MiniZinc array format."""
        flat_list = [item for row in matrix_2d for item in row]
        return f"[{', '.join(map(str, flat_list))}]"

    # Write .dzn file
    try:
        with open(dzn_path, "w") as f:
            f.write(f"JOBS = {instance.num_jobs};\n")
            f.write(f"MACHINES = {instance.num_machines};\n\n")
            f.write(
                f"PROC_TIME = array2d(SET_JOBS, SET_POS, {flatten_array(proc_time_rows)});\n"
            )
            f.write(
                f"MACHINE_OF_OP = array2d(SET_JOBS, SET_POS, {flatten_array(machine_of_op_rows)});\n"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to write .dzn file '{dzn_path}'. Error: {e}")

    return dzn_path


def get_instance_optimum(instance: JobShopInstance) -> float:
    """
    Get the known optimum makespan for an instance, if available.

    Args:
        instance: JobShopInstance object

    Returns:
        Optimum makespan value, or float('inf') if not known
    """
    optimum = instance.metadata.get("optimum", float("inf"))

    # Validate optimum value
    if not isinstance(optimum, (int, float)) or optimum <= 0:
        return float("inf")

    return float(optimum)
