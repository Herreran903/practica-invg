"""
Instance path resolution utilities for SAT problems.

This module provides functions to resolve paths to SAT instance files
(CNF, XCSP, DZN, etc.) from instance IDs or partial paths.
"""

import os
import warnings
from typing import Dict, Optional

import pandas as pd

# Prefer working with uncompressed plaintext files. Compressed files are exceptional.
COMPRESSED_EXTS = (".gz", ".bz2", ".xz", ".zip", ".lzma", ".izma")


def _is_compressed(path: str) -> bool:
    p = str(path).lower()
    return any(p.endswith(ext) for ext in COMPRESSED_EXTS)


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

    # Recursively walk directory tree (only index UNCOMPRESSED files)
    for root, _, files in os.walk(instances_dir):
        for filename in files:
            if _is_compressed(filename):
                # Skip compressed files; resolution will treat them as exceptional fallbacks
                continue
            path = os.path.join(root, filename)
            mapping[filename] = path

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

    return dict(zip(df["instance_id"].astype(str), df["file_path"].astype(str)))


def resolve_raw_text_path(
    instance_id: str, map_by_filename: Dict[str, str], map_by_id: Dict[str, str]
) -> Optional[str]:
    """
    Resolve the raw file path for an instance by ID or filename.

    Standard flow prefers UNCOMPRESSED plaintext files. Using compressed
    archives is an exceptional fallback and will emit a warning recommending
    to decompress with scripts/decompress_instances.py.

    Priority:
    1. Map by ID (instance_id → file_path), preferring uncompressed sibling
    2. Map by filename (filename → path) [index includes only uncompressed]
    3. If instance_id carries a compression suffix, try base name in map

    Args:
        instance_id: Instance identifier (or filename)
        map_by_filename: Mapping filename → path (should contain only uncompressed)
        map_by_id: Mapping instance_id → path

    Returns:
        Resolved path if found, None otherwise
    """
    # 1) Try explicit ID mapping first
    if instance_id in map_by_id and os.path.exists(map_by_id[instance_id]):
        p = map_by_id[instance_id]
        if _is_compressed(p):
            # Prefer uncompressed sibling if it exists
            plain = p
            for ext in COMPRESSED_EXTS:
                if p.lower().endswith(ext):
                    plain = p[: -len(ext)]
                    break
            if os.path.exists(plain):
                return plain
            warnings.warn(
                f"[sat_images] Exceptional path resolution: using COMPRESSED archive via instance_map: {p}. "
                f"Decompression recommended (python3 scripts/decompress_instances.py ...)."
            )
        return p

    # 2) Try exact filename (uncompressed index)
    if instance_id in map_by_filename and os.path.exists(map_by_filename[instance_id]):
        return map_by_filename[instance_id]

    # 3) If provided id has compression suffix, try its base in the filename map
    for ext in COMPRESSED_EXTS:
        if instance_id.lower().endswith(ext):
            base = instance_id[: -len(ext)]
            if base in map_by_filename and os.path.exists(map_by_filename[base]):
                return map_by_filename[base]
            break

    # Not found
    return None


def resolve_path_with_prefix_map(
    instances_root: str, instance_id: str, prefix_map: Dict[str, str]
) -> Optional[str]:
    """
    Resolve instance path using prefix mapping.

    Standard flow prefers UNCOMPRESSED plaintext files. Using compressed
    archives is an exceptional fallback and will emit a warning.

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

    # 1) Try direct path (prefer uncompressed)
    direct_path = os.path.join(instances_root, instance_id)
    if os.path.exists(direct_path):
        if _is_compressed(direct_path):
            # Prefer uncompressed sibling if available
            plain = direct_path
            for ext in COMPRESSED_EXTS:
                if direct_path.lower().endswith(ext):
                    plain = direct_path[: -len(ext)]
                    break
            if os.path.exists(plain):
                return plain
            warnings.warn(
                f"[sat_images] Exceptional path resolution: using COMPRESSED archive: {direct_path}. "
                f"Decompression recommended (python3 scripts/decompress_instances.py ...)."
            )
        return direct_path

    # 2) Try compressed variants of direct path (exceptional)
    for ext in COMPRESSED_EXTS:
        cand = direct_path + ext
        if os.path.exists(cand):
            warnings.warn(
                f"[sat_images] Exceptional path resolution: using COMPRESSED archive: {cand}. "
                f"Decompression recommended (python3 scripts/decompress_instances.py ...)."
            )
            return cand

    # 3) Try with prefix mapping (prefer uncompressed)
    parts = instance_id.split("/", 1)
    if len(parts) == 2 and parts[0] in prefix_map:
        mapped_path = os.path.join(instances_root, prefix_map[parts[0]], parts[1])
        if os.path.exists(mapped_path):
            if _is_compressed(mapped_path):
                plain = mapped_path
                for ext in COMPRESSED_EXTS:
                    if mapped_path.lower().endswith(ext):
                        plain = mapped_path[: -len(ext)]
                        break
                if os.path.exists(plain):
                    return plain
                warnings.warn(
                    f"[sat_images] Exceptional path resolution: using COMPRESSED archive: {mapped_path}. "
                    f"Decompression recommended (python3 scripts/decompress_instances.py ...)."
                )
            return mapped_path

        # Compressed variants for mapped path (exceptional)
        for ext in COMPRESSED_EXTS:
            cand = mapped_path + ext
            if os.path.exists(cand):
                warnings.warn(
                    f"[sat_images] Exceptional path resolution: using COMPRESSED archive: {cand}. "
                    f"Decompression recommended (python3 scripts/decompress_instances.py ...)."
                )
                return cand

    # 4) Exhaustive search by basename, prefer uncompressed
    basename = os.path.basename(instance_id)
    candidate_plain: Optional[str] = None
    candidate_comp: Optional[str] = None
    for root, _, files in os.walk(instances_root):
        for fn in files:
            full = os.path.join(root, fn)
            if fn == basename:
                if _is_compressed(fn):
                    candidate_comp = full
                else:
                    candidate_plain = full
            else:
                # Check "basename + compression" pattern
                for ext in COMPRESSED_EXTS:
                    if fn.endswith(ext) and fn[: -len(ext)] == basename:
                        candidate_comp = full
                        break

    if candidate_plain:
        return candidate_plain
    if candidate_comp:
        warnings.warn(
            f"[sat_images] Exceptional path resolution: using COMPRESSED archive: {candidate_comp}. "
            f"Decompression recommended (python3 scripts/decompress_instances.py ...)."
        )
        return candidate_comp

    return None
