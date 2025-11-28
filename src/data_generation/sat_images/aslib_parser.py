"""
ASlib scenario parser utilities.

This module provides functions to parse ASlib scenario files, particularly
algorithm_runs.arff and description.txt files.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


def read_text_file(path: str) -> str:
    """
    Read a text file with UTF-8 encoding, ignoring errors.
    
    Args:
        path: Path to the text file
        
    Returns:
        Content of the file as string
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_arff_minimal(arff_path: str) -> Tuple[List[str], List[List[str]]]:
    """
    Minimal ARFF parser for dense format (no sparse support).
    
    Extracts attribute names and data rows, ignoring comments and empty lines.
    Does not support sparse format records with `{...}`.
    
    Args:
        arff_path: Path to the ARFF file
        
    Returns:
        Tuple of (attribute_names, data_rows)
        
    Raises:
        ValueError: If @data section is not found
        NotImplementedError: If sparse format is detected
    """
    # Read and preprocess lines: clean, remove empty and comments
    lines = []
    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            lines.append(stripped)
    
    # Parse header: collect @attribute and locate @data
    attr_names: List[str] = []
    data_idx = None
    
    for i, line in enumerate(lines):
        if line.lower().startswith("@attribute"):
            match = re.match(r"@attribute\s+([^\s]+)\s+(.+)", line, re.IGNORECASE)
            if match:
                # Remove quotes from attribute name
                attr = match.group(1).strip("'").strip('"')
                attr_names.append(attr)
        
        elif line.lower().startswith("@data"):
            data_idx = i + 1
            break
    
    if data_idx is None:
        raise ValueError(f"@data section not found in {arff_path}")
    
    # Parse data rows (dense format only, comma-separated)
    rows: List[List[str]] = []
    for line in lines[data_idx:]:
        if line.startswith("{"):
            raise NotImplementedError(
                "Sparse ARFF format not supported. "
                "This parser only handles dense format."
            )
        
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(attr_names):
            # Skip malformed rows
            continue
        
        rows.append(parts)
    
    return attr_names, rows


def load_algorithm_runs_dataframe(arff_path: str) -> pd.DataFrame:
    """
    Load algorithm_runs.arff into a normalized DataFrame.
    
    Renames common columns to a standard schema: instance_id, algorithm,
    runtime, runstatus.
    
    Args:
        arff_path: Path to algorithm_runs.arff
        
    Returns:
        DataFrame with normalized columns and appropriate types
        
    Raises:
        ValueError: If required columns are missing
    """
    # Parse ARFF and create base DataFrame
    names, rows = parse_arff_minimal(arff_path)
    df = pd.DataFrame(rows, columns=names)
    
    # Normalize column names to standard schema
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        
        if col_lower in ("instance_id", "instance"):
            rename_map[col] = "instance_id"
        elif col_lower in ("algorithm", "solver"):
            rename_map[col] = "algorithm"
        elif col_lower in ("runtime", "run_time", "runtime_secs", "rtime"):
            rename_map[col] = "runtime"
        elif col_lower in ("runstatus", "status", "run_status"):
            rename_map[col] = "runstatus"
    
    df = df.rename(columns=rename_map)
    
    # Validate required columns
    required = {"instance_id", "algorithm", "runtime", "runstatus"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns {required} in {arff_path}. "
            f"Found columns: {set(df.columns)}"
        )
    
    # Type conversion and cleaning
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["runstatus"] = df["runstatus"].astype(str).str.strip()
    
    return df


def try_read_timeout_from_description(desc_path: str) -> Optional[float]:
    """
    Attempt to extract timeout from description.txt file.
    
    Searches for common patterns like 'cutoff time', 'cpu limit', 'time limit'.
    
    Args:
        desc_path: Path to description.txt
        
    Returns:
        Timeout in seconds if found, None otherwise
    """
    if not os.path.exists(desc_path):
        return None
    
    text = read_text_file(desc_path)
    
    # Try common patterns for timeout specification
    patterns = [
        r"cutoff[_\s]*time\s*[:=]\s*([0-9]+(\.[0-9]+)?)",
        r"cpu[_\s]*limit\s*[:=]\s*([0-9]+(\.[0-9]+)?)",
        r"time[_\s]*limit\s*[:=]\s*([0-9]+(\.[0-9]+)?)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                # If parsing fails, try next pattern
                continue
    
    return None


def normalize_runstatus(status: str) -> str:
    """
    Normalize a runstatus string.
    
    Some statuses come as 'TIMEOUT: ...' - we extract just the label.
    
    Args:
        status: Raw status string
        
    Returns:
        Normalized status string
    """
    return (status or "").split(":")[0].strip()


def build_pivot_runtime_table(
    runs_df: pd.DataFrame,
    timeout_s: float
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a pivot table with runtimes and statuses per instance/solver.
    
    - Fills missing runtimes with timeout_s
    - Normalizes statuses and generates *_Status columns
    - Takes minimum runtime per (instance, solver) in case of duplicates
    
    Args:
        runs_df: Normalized DataFrame of algorithm runs
        timeout_s: Timeout to use for missing values
        
    Returns:
        Tuple of (pivoted_dataframe, runtime_column_names)
    """
    # Defensive copy and normalizations
    runs = runs_df.copy()
    runs["algorithm"] = runs["algorithm"].astype(str)
    runs["runstatus_norm"] = runs["runstatus"].map(normalize_runstatus)
    
    # Sort and keep best (first) run per instance/solver
    runs = runs.sort_values(["instance_id", "algorithm", "runtime"]).drop_duplicates(
        ["instance_id", "algorithm"], keep="first"
    )
    
    # Create separate pivots for runtime and status
    pivot_runtime = runs.pivot(
        index="instance_id",
        columns="algorithm",
        values="runtime"
    ).fillna(timeout_s)
    
    pivot_status = runs.pivot(
        index="instance_id",
        columns="algorithm",
        values="runstatus_norm"
    ).fillna("TIMEOUT")
    
    # Rename columns to *_Runtime_s and *_Status
    runtime_cols = []
    new_runtime_names = {}
    new_status_names = {}
    
    for alg in pivot_runtime.columns:
        runtime_col = f"{alg}_Runtime_s"
        status_col = f"{alg}_Status"
        
        runtime_cols.append(runtime_col)
        new_runtime_names[alg] = runtime_col
        new_status_names[alg] = status_col
    
    pivot_runtime = pivot_runtime.rename(columns=new_runtime_names)
    pivot_status = pivot_status.rename(columns=new_status_names)
    
    # Merge both pivots
    merged = pivot_runtime.merge(
        pivot_status,
        left_index=True,
        right_index=True,
        how="left"
    ).reset_index()
    
    return merged, runtime_cols


def compute_winner_key(
    row: pd.Series,
    runtime_cols: List[str],
    timeout_s: float
) -> str:
    """
    Determine the winning solver (minimum time < timeout) for a pivot row.
    
    Args:
        row: Row from pivot table with *_Runtime_s columns
        runtime_cols: List of runtime column names
        timeout_s: Timeout in seconds
        
    Returns:
        Name of winning solver (without suffix), or "NONE" if no solver solved
    """
    # Filter solvers that solved (runtime < timeout)
    solved = [
        (col, float(row[col]))
        for col in runtime_cols
        if float(row[col]) < timeout_s
    ]
    
    if not solved:
        return "NONE"
    
    # Choose solver with minimum time and remove suffix
    winner_col, _ = min(solved, key=lambda kv: kv[1])
    return winner_col.replace("_Runtime_s", "")