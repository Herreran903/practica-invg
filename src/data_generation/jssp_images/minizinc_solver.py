"""
MiniZinc solver execution utilities for JSSP.

This module handles the execution of MiniZinc solvers and parsing of their output,
including support for both CP and MIP solvers with various configurations.
"""

import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple


def list_available_solvers() -> List[str]:
    """
    Detect available MiniZinc solvers on the system.

    Returns:
        List of solver IDs (lowercase) detected by `minizinc --solvers`
    """
    try:
        result = subprocess.run(
            ["minizinc", "--solvers"], capture_output=True, text=True, timeout=10
        )
        text = result.stdout.lower()
    except Exception:
        return []

    # Extract solver names from output
    tokens = set()
    for token in re.findall(r"[a-z0-9\-]+", text):
        tokens.add(token.strip())

    return list(tokens)


def filter_solver_candidates(
    solver_candidates: List[Tuple[str, str, str, Dict[str, Any]]],
    mip_model_path: Optional[str],
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """
    Filter solver candidates based on availability and model requirements.

    Args:
        solver_candidates: List of (solver_id, key, type, options) tuples
        mip_model_path: Path to MIP model (None if not available)

    Returns:
        Filtered list of usable solver configurations

    Raises:
        RuntimeError: If no usable solvers are found
    """
    available = set(list_available_solvers())
    configs: List[Tuple[str, str, str, Dict[str, Any]]] = []

    for solver_id, key, solver_type, opts in solver_candidates:
        # Skip if solver not installed
        if solver_id not in available:
            continue

        # Skip MIP solvers if no MIP model provided
        if solver_type == "mip" and not mip_model_path:
            continue

        configs.append((solver_id, key, solver_type, opts))

    if not configs:
        raise RuntimeError(
            "No usable solvers found. Please check:\n"
            "1. MiniZinc is installed and in PATH\n"
            "2. At least one solver is installed\n"
            "3. MIP model path is provided if using MIP solvers"
        )

    return configs


def patch_model_with_search_strategy(
    model_path: str, inject_search: bool, strategy: Optional[str], output_path: str
) -> str:
    """
    Create a modified version of a MiniZinc model with injected search strategy.

    For CP models with inject_search=True, replaces the solve statement with
    one that includes an int_search annotation using the specified strategy.

    Args:
        model_path: Path to original .mzn model file
        inject_search: Whether to inject search strategy
        strategy: Search strategy name (e.g., "first_fail", "input_order")
        output_path: Path where modified model will be saved

    Returns:
        Path to the output file

    Raises:
        RuntimeError: If model file cannot be read
    """
    try:
        with open(model_path, "r") as f:
            mzn_content = f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Model file not found: {model_path}")

    # Inject search strategy if requested
    if inject_search and strategy:
        # Find the solve statement
        match = re.search(
            r"^\s*solve\s+minimize\s+END_MAKESPAN\s*;", mzn_content, re.MULTILINE
        )

        if match:
            # Replace with search-annotated version
            replacement = (
                f"solve :: int_search(S_FLAT, {strategy}, indomain_min, complete) "
                f"minimize END_MAKESPAN;"
            )
            mzn_content = mzn_content.replace(match.group(0), replacement)

    # Write modified model
    with open(output_path, "w") as f:
        f.write(mzn_content)

    return output_path


def parse_minizinc_output(output_text: str, time_limit_s: float) -> Dict[str, Any]:
    """
    Parse MiniZinc solver output to extract statistics.

    Extracts:
    - makespan: Best makespan found (from END=<value> lines)
    - runtime: Solver runtime in seconds (from %%%mzn-stat lines)
    - solved_binary: 1 if solution found, 0 otherwise
    - had_time_tag: Whether runtime was explicitly reported

    Args:
        output_text: Raw stdout from MiniZinc
        time_limit_s: Time limit used (fallback if no time reported)

    Returns:
        Dictionary with parsed statistics
    """
    stats = {
        "makespan": float("inf"),
        "runtime": time_limit_s,
        "had_time_tag": False,
        "solved_binary": 0,
    }

    try:
        # Extract all makespan values (END=<number>)
        all_makespans = re.findall(r"END=\s*(\d+)", output_text)
        if all_makespans:
            # Take the minimum (best) makespan found
            stats["makespan"] = min(float(m) for m in all_makespans)
            stats["solved_binary"] = 1

        # Extract runtime from statistics
        time_match = re.search(r"%%%mzn-stat:\s*solveTime=([0-9\.]+)", output_text)
        if not time_match:
            time_match = re.search(r"%%%mzn-stat:\s*time=([0-9\.]+)", output_text)

        if time_match:
            stats["runtime"] = float(time_match.group(1))
            stats["had_time_tag"] = True

    except Exception:
        # If parsing fails, use defaults
        pass

    return stats


def execute_minizinc_solver(
    solver_id: str,
    key: str,
    solver_type: str,
    options: Dict[str, Any],
    dzn_path: str,
    time_limit_ms: int,
    cp_model_path: str,
    mip_model_path: Optional[str],
    temp_dir: str,
) -> Dict[str, Any]:
    """
    Execute a MiniZinc solver on a JSSP instance.

    Args:
        solver_id: MiniZinc solver identifier (e.g., "gecode", "cplex")
        key: Unique key for this solver configuration
        solver_type: "cp" or "mip"
        options: Solver options dict (strategy, supports_seed, inject_search)
        dzn_path: Path to instance .dzn file
        time_limit_ms: Time limit in milliseconds
        cp_model_path: Path to CP model .mzn file
        mip_model_path: Path to MIP model .mzn file (or None)
        temp_dir: Directory for temporary files

    Returns:
        Dictionary with solver statistics and metadata
    """
    time_limit_s = time_limit_ms / 1000.0
    is_cp = solver_type == "cp"
    inject = bool(options.get("inject_search", False))
    strategy = options.get("strategy")

    # Select appropriate model
    model_path = cp_model_path if is_cp else mip_model_path
    if not model_path or not os.path.exists(model_path):
        return {
            "makespan": float("inf"),
            "runtime": time_limit_s,
            "wall_time_s": 0.0,
            "solved_binary": 0,
            "had_time_tag": False,
            "solver": solver_id,
            "key": key,
            "type": solver_type,
            "time_limit_s": time_limit_s,
            "returncode": -1,
        }

    # Create patched model if needed
    os.makedirs(temp_dir, exist_ok=True)
    temp_model = os.path.join(temp_dir, f"__tmp_{key}_T{time_limit_ms}.mzn")
    patched_model = patch_model_with_search_strategy(
        model_path, inject and is_cp, strategy, temp_model
    )

    # Build command with solver-specific options
    extra_args: List[str] = []

    # Handle solver-specific DLL paths (macOS examples)
    if solver_id == "cplex":
        dll = os.environ.get("CPLEX_DLL") or os.path.join(
            os.environ.get("CPLEX_STUDIO_DIR", "/Applications/CPLEX_Studio2211"),
            "cplex",
            "bin",
            "arm64_osx",
            "libcplex.dylib",
        )
        if os.path.exists(dll):
            extra_args += ["--cplex-dll", dll]

    elif solver_id == "highs":
        dll = (
            os.environ.get("HIGHS_DLL") or "/opt/homebrew/opt/highs/lib/libhighs.dylib"
        )
        if os.path.exists(dll):
            extra_args += ["--highs-dll", dll]

    # Build full command
    cmd = [
        "minizinc",
        "--solver",
        solver_id,
        "--statistics",
        "--time-limit",
        str(time_limit_ms),
        patched_model,
        dzn_path,
    ]

    # Insert extra args after minizinc but before --solver
    if extra_args:
        cmd[1:1] = extra_args

    # Execute solver
    wall_start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit_s + 10,  # Extra buffer for timeout
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode

    except subprocess.TimeoutExpired:
        stdout = f"%%%mzn-stat: solveTime={time_limit_s}\n"
        stderr = "Timeout expired"
        returncode = 124

    except FileNotFoundError:
        raise RuntimeError(
            "MiniZinc not found in PATH. Please install MiniZinc and ensure it's accessible."
        )

    finally:
        wall_time = time.time() - wall_start
        # Clean up temporary model file
        try:
            os.unlink(patched_model)
        except Exception:
            pass

    # Parse output
    stats = parse_minizinc_output(stdout, time_limit_s)

    # If no explicit time tag, use wall time as fallback
    if not stats.get("had_time_tag", False):
        stats["runtime"] = min(stats.get("runtime", time_limit_s), wall_time)

    # Add metadata
    stats.update(
        {
            "solver": solver_id,
            "key": key,
            "type": solver_type,
            "time_limit_s": time_limit_s,
            "wall_time_s": wall_time,
            "returncode": returncode,
        }
    )

    # Log warnings for non-zero return codes
    if returncode != 0:
        err_snippet = (stderr or "").strip().splitlines()
        err_snippet = " | ".join(err_snippet[:3])[:200]
        print(
            f"     Note: returncode={returncode}, "
            f"had_time_tag={stats['had_time_tag']}, "
            f"stderr='{err_snippet}'"
        )

    return stats


def calculate_relative_performance_score(
    stats: Dict[str, Any], optimum: float, time_limit_s: float, penalty_factor_k: float
) -> float:
    """
    Calculate relative performance score (PAR-style metric).

    Score combines:
    - Runtime of the solver
    - Quality gap relative to known optimum

    Lower score is better. Unsolved instances get penalty of K * time_limit.

    Args:
        stats: Solver statistics dictionary
        optimum: Known optimum makespan (or best known)
        time_limit_s: Time limit in seconds
        penalty_factor_k: Penalty multiplier for unsolved instances

    Returns:
        Performance score (lower is better)
    """
    runtime = stats["runtime"]
    makespan_found = stats["makespan"]

    # Penalize if no solution found or invalid optimum
    if makespan_found == float("inf") or optimum <= 0 or optimum == float("inf"):
        return time_limit_s * penalty_factor_k

    # Calculate quality gap
    gap = (makespan_found - optimum) / optimum

    # Score = runtime + (time_limit * gap)
    score = runtime + (time_limit_s * gap)

    # Cap at maximum penalty
    return min(score, time_limit_s * penalty_factor_k)
