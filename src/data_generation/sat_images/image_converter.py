"""
Image conversion utilities for SAT instances.

This module converts SAT instance files (CNF, XCSP, DZN, etc.) into grayscale
image representations stored as NumPy arrays (.npy files).
"""

import hashlib
import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from .config_loader import SATImagesConfig
from .instance_resolver import resolve_path_with_prefix_map


def convert_raw_text_to_grayscale_image(
    raw_text_path: str,
    target_size: int = 128
) -> np.ndarray:
    """
    Convert a text file into a normalized grayscale image matrix.
    
    The conversion process:
    1. Read file as binary buffer
    2. Convert bytes to uint8 vector
    3. Reshape into largest possible square matrix
    4. Resize to target_size × target_size using LANCZOS interpolation
    5. Normalize using z-score (mean=0, std=1)
    
    Args:
        raw_text_path: Path to the input file
        target_size: Target size for the square output image (default: 128)
        
    Returns:
        Normalized grayscale image as float32 numpy array of shape (target_size, target_size)
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    # Read file as binary
    with open(raw_text_path, "rb") as f:
        data = f.read()
    
    # Convert to uint8 vector
    vector = np.frombuffer(data, dtype=np.uint8)
    n_bytes = vector.size
    
    if n_bytes == 0:
        # Return empty matrix for empty files
        return np.zeros((target_size, target_size), dtype=np.float32)
    
    # Calculate largest usable square
    side_length = int(np.floor(np.sqrt(n_bytes)))
    usable = side_length * side_length
    
    if usable <= 0:
        return np.zeros((target_size, target_size), dtype=np.float32)
    
    # Create square matrix
    initial_image = vector[:usable].reshape(side_length, side_length).astype(np.uint8)
    
    # Resize to target dimensions using high-quality LANCZOS resampling
    pil_image = Image.fromarray(initial_image, mode="L")
    resized_image = pil_image.resize(
        (target_size, target_size),
        Image.Resampling.LANCZOS
    )
    
    # Convert to numpy array
    image_array = np.asarray(resized_image, dtype=np.float32)
    
    # Normalize using z-score (only if std > 0)
    std_dev = float(image_array.std())
    if std_dev > 0:
        image_array = (image_array - float(image_array.mean())) / std_dev
    
    return image_array


def convert_dataset_to_images(
    csv_path: str,
    instances_root: str,
    target_size: int = 128,
    prefix_map: Optional[dict] = None,
    instance_id_col: str = "Instance_Id",
    instance_name_col: str = "Instance_Name",
    raw_path_col: str = "Raw_Text_Path",
    output_col: str = "Image_Npy_Path"
) -> None:
    """
    Convert all instances in a dataset CSV to grayscale images.
    
    Reads a CSV file containing instance information, converts each instance's
    file to a grayscale image, saves as .npy, and updates the CSV with image paths.
    
    Args:
        csv_path: Path to the dataset CSV file
        instances_root: Root directory for instance files
        target_size: Target size for generated images (default: 128)
        prefix_map: Optional mapping of prefixes to directory paths
        instance_id_col: Name of column containing instance IDs
        instance_name_col: Name of column containing instance names
        raw_path_col: Name of column containing paths to raw files
        output_col: Name of column to add with image paths
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    print("=" * 75)
    print("Image Conversion: SAT Instances to Grayscale")
    print("=" * 75)
    print(f"Input CSV: {os.path.abspath(csv_path)}")
    print(f"Instances root: {os.path.abspath(instances_root)}")
    
    # Read CSV
    print("\n[1/3] Reading CSV...")
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Found {total} instances")
    
    # Create output directory for images
    output_dir = os.path.abspath(os.path.dirname(csv_path) or ".")
    images_dir = os.path.abspath(os.path.join(output_dir, "images"))
    os.makedirs(images_dir, exist_ok=True)
    print(f"Output directory: {images_dir}")
    
    # Initialize lists and counters
    raw_paths = []
    image_paths = []
    success_count = 0
    missing_count = 0
    
    print(f"\n[2/3] Converting instances to {target_size}×{target_size} grayscale images...")
    
    for idx, row in df.iterrows():
        # Resolve raw file path
        raw_path = row.get(raw_path_col)
        
        if not (isinstance(raw_path, str) and os.path.exists(raw_path)):
            # Try to resolve using instance ID and prefix map
            instance_id = str(row.get(instance_id_col, ""))
            if prefix_map:
                raw_path = resolve_path_with_prefix_map(
                    instances_root,
                    instance_id,
                    prefix_map
                )
            else:
                raw_path = None
        
        if raw_path and os.path.exists(raw_path):
            raw_paths.append(raw_path)
            
            try:
                # Convert to image
                image_array = convert_raw_text_to_grayscale_image(
                    raw_path,
                    target_size
                )
                
                # Generate unique filename using hash
                instance_id = str(row.get(instance_id_col, row.get(instance_name_col, "")))
                hash_suffix = hashlib.md5(
                    instance_id.encode("utf-8", errors="ignore")
                ).hexdigest()[:12]
                
                basename = os.path.splitext(os.path.basename(instance_id))[0]
                npy_filename = f"{basename}__{hash_suffix}.npy"
                npy_path = os.path.abspath(os.path.join(images_dir, npy_filename))
                
                # Save as .npy
                np.save(npy_path, image_array.astype(np.float32))
                image_paths.append(npy_path)
                success_count += 1
                
                if (idx + 1) % 100 == 0 or idx == 0:
                    print(f"  [{idx + 1}/{total}] Processed...")
            
            except Exception as e:
                print(f"  Warning: Failed to convert {raw_path}. Error: {e}")
                image_paths.append("")
                missing_count += 1
        else:
            # File not found
            raw_paths.append("" if raw_path is None else raw_path)
            image_paths.append("")
            missing_count += 1
    
    # Update CSV
    print("\n[3/3] Updating CSV with image paths...")
    df[raw_path_col] = raw_paths
    df[output_col] = image_paths
    df.to_csv(csv_path, index=False)
    
    print(f"Conversion complete: {success_count} successful, {missing_count} missing/failed")
    print("=" * 75)
    print("Image Conversion Complete")
    print("=" * 75 + "\n")


def generate_all_images(
    csv_path: str,
    instances_root: str,
    target_size: int = 128,
    prefix_map: Optional[dict] = None
) -> None:
    """
    Legacy wrapper function for backward compatibility.
    
    Converts all instances in a CSV to grayscale images.
    
    Args:
        csv_path: Path to the dataset CSV file
        instances_root: Root directory for instance files
        target_size: Target size for generated images (default: 128)
        prefix_map: Optional mapping of prefixes to directory paths
    """
    convert_dataset_to_images(
        csv_path,
        instances_root,
        target_size,
        prefix_map
    )