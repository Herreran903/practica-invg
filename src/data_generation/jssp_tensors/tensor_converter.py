"""
Tensor conversion utilities for JSSP instances.

This module converts JSSP instance .dzn files into 2D tensor/matrix representations
stored as NumPy arrays (.npy files). The representation follows the CONVJSSP style:
- Each instance becomes a 2D matrix of shape (num_jobs, num_machines)
- Each cell contains the processing time (duration) for that operation
- Optional z-score standardization: (x - mean) / std
"""

import os
import re
from typing import List

import numpy as np
import pandas as pd


def parse_dzn_int(name: str, text: str) -> int:
    """
    Extract an integer assignment from .dzn text.
    
    Looks for patterns like: NAME = 10;
    
    Args:
        name: Variable name to search for
        text: Content of the .dzn file
        
    Returns:
        Integer value
        
    Raises:
        ValueError: If the variable is not found
    """
    pattern = rf"{name}\s*=\s*([0-9]+)\s*;"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Variable '{name}' not found in .dzn file")
    return int(match.group(1))


def parse_dzn_array2d(name: str, text: str) -> List[int]:
    """
    Extract a list of integers from an array2d definition in .dzn text.
    
    Looks for patterns like:
        NAME = array2d(SET_JOBS, SET_POS, [v1, v2, ..., vk]);
    
    Args:
        name: Array variable name to search for
        text: Content of the .dzn file
        
    Returns:
        List of integer values
        
    Raises:
        ValueError: If the array is not found or cannot be parsed
    """
    pattern = rf"{name}\s*=\s*array2d\([^,]+,[^,]+,\s*\[(.*?)\]\s*\)\s*;"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Array '{name}' not found as array2d in .dzn file")
    
    # Extract values inside brackets
    inside = match.group(1)
    tokens = [t.strip() for t in inside.replace("\n", " ").split(",") if t.strip()]
    
    try:
        values = [int(tok) for tok in tokens]
    except ValueError as e:
        raise ValueError(f"Cannot convert values of '{name}' to integers") from e
    
    return values


def convert_dzn_to_tensor(
    dzn_path: str,
    standardize: bool = True
) -> np.ndarray:
    """
    Convert a .dzn file to a 2D tensor/matrix representation.
    
    The resulting matrix has shape (num_jobs, num_machines) where each cell
    contains the processing time for that operation.
    
    Args:
        dzn_path: Path to the .dzn file
        standardize: If True, apply z-score normalization (x - mean) / std
        
    Returns:
        2D numpy array of shape (num_jobs, num_machines) with float32 dtype
        
    Raises:
        FileNotFoundError: If .dzn file doesn't exist
        ValueError: If .dzn file is malformed or has inconsistent dimensions
    """
    if not os.path.exists(dzn_path):
        raise FileNotFoundError(f".dzn file not found: {dzn_path}")
    
    # Read file content
    with open(dzn_path, "r") as f:
        text = f.read()
    
    # Parse dimensions
    num_jobs = parse_dzn_int("JOBS", text)
    num_machines = parse_dzn_int("MACHINES", text)
    
    # Parse processing times
    proc_flat = parse_dzn_array2d("PROC_TIME", text)
    expected_len = num_jobs * num_machines
    
    if len(proc_flat) != expected_len:
        raise ValueError(
            f"PROC_TIME length inconsistent with JOBS*MACHINES. "
            f"PROC_TIME has {len(proc_flat)} elements, but JOBS*MACHINES = {expected_len}"
        )
    
    # Reshape to 2D matrix (jobs x machines)
    proc_matrix = np.array(proc_flat, dtype=np.float32).reshape((num_jobs, num_machines))
    
    # Apply standardization if requested
    if standardize:
        mean = proc_matrix.mean()
        std = proc_matrix.std()
        
        if std == 0.0:
            raise ValueError(
                f"Standard deviation is zero in PROC_TIME of {dzn_path}. "
                f"Cannot standardize. All values are identical."
            )
        
        proc_matrix = (proc_matrix - mean) / std
    
    return proc_matrix


def convert_dataset_to_tensors(
    csv_path: str,
    standardize: bool = True,
    instance_name_col: str = "Instance_Name",
    text_path_col: str = "Raw_Text_Path",
    output_col: str = "Tensor_Npy_Path"
) -> None:
    """
    Convert all instances in a dataset CSV to tensor representations.
    
    Reads a CSV file containing instance information, converts each instance's
    .dzn file to a 2D tensor, saves as .npy, and updates the CSV with tensor paths.
    
    Args:
        csv_path: Path to the dataset CSV file
        standardize: If True, apply z-score normalization to tensors
        instance_name_col: Name of column containing instance names
        text_path_col: Name of column containing paths to .dzn files
        output_col: Name of column to add with tensor paths
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If required columns are missing from CSV
    """
    print("=" * 75)
    print("Tensor Conversion: .dzn to 2D Matrix (CONVJSSP Style)")
    print("=" * 75)
    print(f"Input CSV: {os.path.abspath(csv_path)}")
    
    # Read CSV
    print("\n[1/3] Reading CSV...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Validate required columns
    if instance_name_col not in df.columns:
        raise KeyError(f"Column '{instance_name_col}' not found in CSV")
    if text_path_col not in df.columns:
        raise KeyError(f"Column '{text_path_col}' not found in CSV")
    
    total_instances = len(df)
    print(f"Found {total_instances} instances to convert")
    
    # Create output directory for tensors
    output_dir = os.path.dirname(csv_path)
    tensors_dir = os.path.join(output_dir, "images")  # Keep "images" for consistency
    os.makedirs(tensors_dir, exist_ok=True)
    print(f"Output directory: {tensors_dir}")
    
    # Convert each instance
    standardization_str = "with" if standardize else "without"
    print(f"\n[2/3] Converting instances to 2D tensors ({standardization_str} standardization)...")
    tensor_paths = []
    
    for idx, row in df.iterrows():
        instance_name = row[instance_name_col]
        dzn_path = row[text_path_col]
        
        print(f"  [{idx + 1}/{total_instances}] {instance_name}...", end=" ")
        
        try:
            # Convert to tensor
            tensor = convert_dzn_to_tensor(dzn_path, standardize=standardize)
            
            # Save as .npy
            npy_filename = f"{instance_name}_tensor.npy"
            npy_path = os.path.join(tensors_dir, npy_filename)
            np.save(npy_path, tensor)
            
            tensor_paths.append(npy_path)
            print(f"✓ shape={tensor.shape}")
        
        except Exception as e:
            print(f"✗ ERROR: {e}")
            tensor_paths.append(None)
    
    # Update CSV with tensor paths
    print("\n[3/3] Updating CSV with tensor paths...")
    df[output_col] = tensor_paths
    df.to_csv(csv_path, index=False)
    
    successful = sum(1 for p in tensor_paths if p is not None)
    print(f"Conversion complete: {successful}/{total_instances} successful")
    print("\n" + "=" * 75)
    print("Tensor Conversion Complete")
    print("=" * 75 + "\n")


def generate_all_images(csv_path: str, standardize: bool = True) -> None:
    """
    Legacy wrapper function for backward compatibility.
    
    Converts all instances in a CSV to tensor representations.
    Note: Despite the name "images", this generates tensors/matrices.
    
    Args:
        csv_path: Path to the dataset CSV file
        standardize: If True, apply z-score normalization
    """
    convert_dataset_to_tensors(csv_path, standardize)