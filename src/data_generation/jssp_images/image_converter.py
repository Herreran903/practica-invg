"""
Image conversion utilities for JSSP instances.

This module converts JSSP instance files (.dzn) into structured grayscale image
representations derived from the problem matrices and stores them as NumPy arrays
(.npy files).

Default behavior:
- Build the image from the JSSP data matrices (not ASCII).
- Use PROC_TIME (JOBS×MACHINES) as a single-channel image, min–max normalized
  to [0, 255], then resized to the target size. Output dtype: float32.
- Optionally include MACHINE_OF_OP as a second channel (see parameters).
- Optional z-score standardization can be enabled via normalize=True.

Pipeline:
1) Parse .dzn to extract: JOBS, MACHINES, PROC_TIME[J×M], MACHINE_OF_OP[J×M]
2) Normalize selected matrix/matrices to [0, 255] (min–max; MACHINE_OF_OP scaled
   over machine IDs)
3) Resize each channel to target_size × target_size (LANCZOS)
4) Stack channels if configured (default: one channel = PROC_TIME)
"""

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


def _parse_dzn(text: str) -> tuple[int, int, np.ndarray, Optional[np.ndarray]]:
    """
    Parse a MiniZinc .dzn for JSSP and extract JOBS, MACHINES, PROC_TIME and MACHINE_OF_OP.

    Returns:
        jobs, machines, proc_time[J,M], machine_of_op[J,M or None if missing]
    """

    def _find_int(name: str) -> int:
        # Tolerant to whitespace/newlines and case
        pattern = re.compile(rf"\b{name}\s*=\s*(\d+)\s*;", re.IGNORECASE | re.MULTILINE)
        m = pattern.search(text)
        if not m:
            raise ValueError(f"Missing integer '{name}' in .dzn")
        return int(m.group(1))

    def _find_array(name: str) -> np.ndarray:
        """
        Robustly extract the flat list inside array2d(...) for the given variable name.
        Tolerant to:
          - Arbitrary whitespace/newlines
          - Any set names or extra arguments before the list (e.g., array2d(SET_JOBS, SET_POS, [ ... ]);)
        """
        # 1) Find the start of "name = array2d("
        start_pat = re.compile(
            rf"\b{name}\s*=\s*array2d\s*\(", re.IGNORECASE | re.MULTILINE
        )
        m = start_pat.search(text)
        if not m:
            raise ValueError(f"Missing array2d for '{name}' in .dzn")

        # 2) Find the first '[' after array2d( ... to locate the list of numbers
        list_start = text.find("[", m.end())
        if list_start == -1:
            raise ValueError(f"array2d for '{name}' has no '[' list section")

        # 3) Find the closing ']' of that list (numbers lists in our .dzn contain only digits/commas/spaces/newlines)
        list_end = text.find("]", list_start + 1)
        if list_end == -1:
            raise ValueError(
                f"array2d for '{name}' has no closing ']' for list section"
            )

        nums_str = text[list_start + 1 : list_end]
        vals = [int(s) for s in re.findall(r"-?\d+", nums_str)]
        return np.asarray(vals, dtype=np.int32)

    jobs = _find_int("JOBS")
    machines = _find_int("MACHINES")

    pt_flat = _find_array("PROC_TIME")
    if pt_flat.size != jobs * machines:
        raise ValueError(
            f"PROC_TIME has {pt_flat.size} elements, expected {jobs*machines}"
        )
    proc_time = pt_flat.reshape(jobs, machines)

    mo = None
    try:
        mo_flat = _find_array("MACHINE_OF_OP")
        if mo_flat.size != jobs * machines:
            raise ValueError(
                f"MACHINE_OF_OP has {mo_flat.size} elements, expected {jobs*machines}"
            )
        mo = mo_flat.reshape(jobs, machines)
    except ValueError:
        mo = None

    return jobs, machines, proc_time, mo


def _minmax_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale arr to uint8 [0..255] using per-channel min–max."""
    a = arr.astype(np.float32)
    mn = float(np.min(a))
    mx = float(np.max(a))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(a, dtype=np.uint8)
    scaled = (a - mn) / (mx - mn)
    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def _resize_uint8_channel(channel_uint8: np.ndarray, target_size: int) -> np.ndarray:
    """Resize single-channel uint8 image to target size; return float32 array."""
    pil = Image.fromarray(channel_uint8, mode="L")
    pil_resized = pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return np.asarray(pil_resized, dtype=np.float32)


def convert_model_and_instance_to_grayscale_image(
    cp_model_path: str,
    dzn_path: str,
    target_size: int = 128,
    normalize: bool = False,
) -> np.ndarray:
    """
    Text-to-Image conversion for JSSP instances.

    This function implements the Text-to-Image approach described in the paper:
    instead of parsing PROC_TIME / MACHINE_OF_OP as numeric matrices, it treats
    the MiniZinc model (.mzn) and instance data (.dzn) as raw text, concatenates
    them, and maps the resulting byte sequence to a single-channel grayscale image.

    Steps:
        1) Read CP model (.mzn) and instance (.dzn) as UTF-8 text (errors ignored)
        2) Concatenate: <model_text> + two newlines + <instance_text>
        3) Encode to bytes and interpret as a uint8 vector
        4) Reshape into the largest possible square matrix
        5) Resize to target_size × target_size using LANCZOS
        6) Optionally apply per-image z-score normalization

    Args:
        cp_model_path: Path to the MiniZinc CP model (.mzn)
        dzn_path: Path to the MiniZinc data file (.dzn) for the instance
        target_size: Output square size (default: 128)
        normalize: If True, apply z-score normalization after resizing

    Returns:
        np.ndarray float32 of shape (target_size, target_size).
        - If normalize=False (default): values ~ [0, 255]
        - If normalize=True: mean≈0, std≈1 (when std > 0)
    """
    # Read model and instance as text, then encode to bytes
    try:
        with open(cp_model_path, "r", encoding="utf-8", errors="ignore") as f:
            model_text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"CP model file not found: {cp_model_path}")

    try:
        with open(dzn_path, "r", encoding="utf-8", errors="ignore") as f:
            instance_text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Instance .dzn file not found: {dzn_path}")

    combined_text = model_text + "\n\n" + instance_text
    data_bytes = combined_text.encode("utf-8", errors="ignore")

    # Convert to uint8 vector
    vector = np.frombuffer(data_bytes, dtype=np.uint8)
    n_bytes = vector.size

    if n_bytes == 0:
        # Return empty matrix for empty/degenerate inputs
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Compute largest usable square from the byte vector
    side_length = int(np.floor(np.sqrt(n_bytes)))
    usable = side_length * side_length

    if usable <= 0:
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Reshape into square matrix and resize
    square = vector[:usable].reshape(side_length, side_length).astype(np.uint8)
    resized = _resize_uint8_channel(square, target_size)  # float32

    if normalize:
        std = float(resized.std())
        if std > 0:
            resized = (resized - float(resized.mean())) / std

    return resized.astype(np.float32)


def convert_text_to_grayscale_image(
    text_file_path: str,
    target_size: int = 128,
    include_machine_channel: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    """
    Convert a .dzn JSSP file into a structured grayscale image.

    - Channel 0 (always): PROC_TIME[J×M], min–max normalized to [0, 255]
    - Channel 1 (optional): MACHINE_OF_OP[J×M] scaled over machine IDs to [0, 255]
      (1-indexed in .dzn; mapped to [0..M-1] then to [0..255])

    The resulting channel(s) are resized to target_size×target_size using LANCZOS.
    If normalize=True, an additional z-score standardization is applied per channel.

    Args:
        text_file_path: Path to the input .dzn file
        target_size: Output square size (default: 128)
        include_machine_channel: Whether to add the MACHINE_OF_OP channel
        normalize: If True, apply z-score standardization per channel

    Returns:
        np.ndarray float32 with shape:
          - (target_size, target_size) for single-channel
          - (target_size, target_size, 2) if include_machine_channel is True

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If .dzn parsing fails or shapes are inconsistent
    """
    # Read file content
    try:
        with open(text_file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Text file not found: {text_file_path}")

    # Parse .dzn structured data
    jobs, machines, proc_time, machine_of_op = _parse_dzn(content)

    # Build channels as uint8
    channels: list[np.ndarray] = []

    # PROC_TIME → min–max → [0..255]
    pt_u8 = _minmax_to_uint8(proc_time)
    channels.append(pt_u8)

    # MACHINE_OF_OP → scale IDs (1..M) to [0..255]
    if include_machine_channel and machine_of_op is not None:
        if machines > 1:
            mo_zero = (machine_of_op.astype(np.float32) - 1.0) / float(machines - 1)
            mo_u8 = np.clip(np.round(mo_zero * 255.0), 0, 255).astype(np.uint8)
        else:
            mo_u8 = np.zeros_like(machine_of_op, dtype=np.uint8)
        channels.append(mo_u8)

    # Resize each channel and stack (if any)
    resized = [_resize_uint8_channel(c, target_size) for c in channels]
    if len(resized) == 1:
        out = resized[0]  # (H, W)
    else:
        out = np.stack(resized, axis=-1)  # (H, W, C)

    # Optional z-score per channel
    if normalize:
        if out.ndim == 2:
            std = float(out.std())
            if std > 0:
                out = (out - float(out.mean())) / std
        else:
            for ch in range(out.shape[-1]):
                ch_std = float(out[..., ch].std())
                if ch_std > 0:
                    out[..., ch] = (out[..., ch] - float(out[..., ch].mean())) / ch_std

    return out.astype(np.float32)


def convert_dataset_to_images(
    csv_path: str,
    cp_model_path: str,
    target_size: int = 128,
    instance_name_col: str = "Instance_Name",
    text_path_col: str = "Raw_Text_Path",
    output_col: str = "Image_Npy_Path",
    normalize: bool = False,
) -> None:
    """
    Convert all instances in a dataset CSV to grayscale images using Text-to-Image.

    This function implements the Text-to-Image approach for JSSP:
    for each instance, it concatenates the MiniZinc CP model (.mzn) specified
    by `cp_model_path` with the instance data file (.dzn) referenced in the CSV,
    and converts the resulting text into a single-channel grayscale image.

    Default behavior (paper-like):
        - No z-score normalization. Pixel intensities reflect raw byte/ASCII values
          in the range [0, 255], resized to the target size and stored as float32.

    Optional:
        - If normalize=True, apply z-score normalization (x - mean) / std per image.

    Args:
        csv_path: Path to the dataset CSV file
        cp_model_path: Path to the MiniZinc CP model (.mzn) used for all instances
        target_size: Target size for generated images (default: 128)
        instance_name_col: Name of column containing instance names
        text_path_col: Name of column containing paths to .dzn files
        output_col: Name of column to add with image paths
        normalize: If True, apply z-score normalization to images

    Raises:
        FileNotFoundError: If CSV file or model file doesn't exist
        KeyError: If required columns are missing from CSV
    """
    print("=== Image Conversion: JSSP Text-to-Image (Model + Instance) ===")
    print(f"Input CSV: {os.path.abspath(csv_path)}")
    print(f"CP model: {os.path.abspath(cp_model_path)}")
 
    # Read CSV
    print("[1/3] Reading CSV...")
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
 
    # Create output directory for images
    output_dir = os.path.dirname(csv_path)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"Output directory: {images_dir}")
 
    # Convert each instance
    print(
        f"[2/3] Converting instances to {target_size}x{target_size} grayscale images (Text-to-Image)..."
    )
    image_paths = []
 
    for idx, row in df.iterrows():
        instance_name = row[instance_name_col]
        text_path = row[text_path_col]
 
        try:
            # Convert concatenated model + instance text to grayscale image
            image_matrix = convert_model_and_instance_to_grayscale_image(
                cp_model_path=cp_model_path,
                dzn_path=text_path,
                target_size=target_size,
                normalize=normalize,
            )
 
            # Save as .npy
            npy_filename = f"{instance_name}_image.npy"
            npy_path = os.path.join(images_dir, npy_filename)
            np.save(npy_path, image_matrix.astype(np.float32))
 
            image_paths.append(npy_path)
            print(f"  [{idx + 1}/{total_instances}] {instance_name} -> {npy_filename}")
 
        except Exception as e:
            print(
                f"  [{idx + 1}/{total_instances}] ERROR processing {instance_name}: {e}"
            )
            image_paths.append(None)
 
    # Update CSV with image paths
    print("[3/3] Updating CSV with image paths...")
    df[output_col] = image_paths
    df.to_csv(csv_path, index=False)
 
    successful = sum(1 for p in image_paths if p is not None)
    print(f"Conversion complete: {successful}/{total_instances} successful")
    print("=== Image Conversion Complete ===\n")


def generate_all_images(csv_path: str, cp_model_path: str, target_size: int = 128) -> None:
    """
    Legacy wrapper function for backward compatibility.
 
    Converts all instances in a CSV to grayscale images using Text-to-Image.
 
    Args:
        csv_path: Path to the dataset CSV file
        cp_model_path: Path to the MiniZinc CP model (.mzn)
        target_size: Target size for generated images (default: 128)
    """
    convert_dataset_to_images(csv_path, cp_model_path, target_size)
