"""
Instance path resolution utilities for SAT problems.

This module provides functions to resolve paths to SAT instance files
(CNF, XCSP, DZN, etc.) from instance IDs or partial paths.
"""

import os
from typing import Dict, Optional

import pandas as pd


def build_instance_path_map(instances_dir: Optional[str]) -> Dict[str, str]:
    """
    Create a mapping 'filename' → 'absolute_path' by walking a directory tree.
    
    Args:
        instances_dir: Root directory to search for instance files
        
    Returns:
        Dictionary mapping filename to absolute path
    """
    mapping: Dict[str, str] = {}
    
    # Return empty if no valid directory
    if not instances_dir or not os.path.isdir(instances_dir):
        return mapping
    
    # Recursively walk directory tree
    for root, _, files in os.walk(instances_dir):
        for filename in files:
            mapping[filename] = os.path.join(root, filename)
    
    return mapping


def load_instance_map_csv(instance_map_csv: Optional[str]) -> Dict[str, str]:
    """
    Load a CSV with explicit mapping 'instance_id' → 'file_path'.
    
    Args:
        instance_map_csv: Path to CSV mapping file
        
    Returns:
        Dictionary mapping instance_id to file_path
        
    Raises:
        ValueError: If required columns are missing
    """
    if not instance_map_csv or not os.path.exists(instance_map_csv):
        return {}
    
    df = pd.read_csv(instance_map_csv)
    
    required = {"instance_id", "file_path"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"instance_map_csv must have columns: {required}. "
            f"Found: {set(df.columns)}"
        )
    
    return dict(zip(
        df["instance_id"].astype(str),
        df["file_path"].astype(str)
    ))


def resolve_raw_text_path(
    instance_id: str,
    map_by_filename: Dict[str, str],
    map_by_id: Dict[str, str]
) -> Optional[str]:
    """
    Resolve the raw file path for an instance by ID or filename.
    
    Priority:
    1. Map by ID (instance_id → file_path)
    2. Map by filename (filename → path)
    
    Args:
        instance_id: Instance identifier (or filename)
        map_by_filename: Mapping filename → path
        map_by_id: Mapping instance_id → path
        
    Returns:
        Resolved path if found, None otherwise
    """
    # Try explicit ID mapping first
    if instance_id in map_by_id and os.path.exists(map_by_id[instance_id]):
        return map_by_id[instance_id]
    
    # Try exact filename match
    if instance_id in map_by_filename and os.path.exists(map_by_filename[instance_id]):
        return map_by_filename[instance_id]
    
    return None


def resolve_path_with_prefix_map(
    instances_root: str,
    instance_id: str,
    prefix_map: Dict[str, str]
) -> Optional[str]:
    """
    Resolve instance path using prefix mapping.
    
    This is useful for ASlib scenarios where instance IDs contain prefixes
    that map to specific directory structures.
    
    Args:
        instances_root: Root directory for instances
        instance_id: Instance identifier or partial path
        prefix_map: Mapping of prefixes to directory paths
        
    Returns:
        Resolved absolute path if found, None otherwise
    """
    # Validate input
    if not isinstance(instance_id, str) or not instance_id.strip():
        return None
    
    # Try direct path first
    direct_path = os.path.join(instances_root, instance_id)
    if os.path.exists(direct_path):
        return direct_path
    
    # Try with prefix mapping
    parts = instance_id.split("/", 1)
    if len(parts) == 2 and parts[0] in prefix_map:
        mapped_path = os.path.join(
            instances_root,
            prefix_map[parts[0]],
            parts[1]
        )
        if os.path.exists(mapped_path):
            return mapped_path
    
    # Exhaustive search by basename
    basename = os.path.basename(instance_id)
    for root, _, files in os.walk(instances_root):
        if basename in files:
            return os.path.join(root, basename)
    
    return None