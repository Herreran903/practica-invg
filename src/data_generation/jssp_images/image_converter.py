"""
Image conversion utilities for JSSP instances.

This module converts JSSP instance text files (.dzn) into grayscale image representations
stored as NumPy arrays (.npy files). The conversion process:
1. Reads the text content as ASCII values
2. Reshapes into a square matrix
3. Resizes to target dimensions
4. Normalizes using z-score normalization
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


def convert_text_to_grayscale_image(
    text_file_path: str,
    target_size: int = 128
) -> np.ndarray:
    """
    Convert a text file into a normalized grayscale image matrix.
    
    The conversion process:
    1. Read file content as plain text
    2. Convert each character to its ASCII value
    3. Reshape into the largest possible square matrix
    4. Resize to target_size x target_size using LANCZOS interpolation
    5. Normalize using z-score (mean=0, std=1)
    
    Args:
        text_file_path: Path to the input text file (.dzn)
        target_size: Target size for the square output image (default: 128)
        
    Returns:
        Normalized grayscale image as float32 numpy array of shape (target_size, target_size)
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    # Read file content
    try:
        with open(text_file_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Text file not found: {text_file_path}")
    
    # Convert characters to ASCII values
    ascii_values = [ord(char) for char in content]
    n_values = len(ascii_values)
    
    # Calculate largest square that fits the data
    side_length = int(np.sqrt(n_values))
    n_usable = side_length * side_length
    
    # Handle edge case: empty or very small files
    if n_usable == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)
    
    # Reshape to square matrix
    ascii_values_square = ascii_values[:n_usable]
    initial_image = np.array(ascii_values_square, dtype=np.uint8).reshape(
        (side_length, side_length)
    )
    
    # Resize to target dimensions using high-quality LANCZOS resampling
    pil_image = Image.fromarray(initial_image)
    resized_image = pil_image.resize(
        (target_size, target_size),
        Image.Resampling.LANCZOS
    )
    
    # Convert back to numpy array
    image_array = np.array(resized_image, dtype=np.float32)
    
    # Normalize using z-score (only if std > 0)
    std_dev = np.std(image_array)
    if std_dev > 0:
        normalized_image = (image_array - np.mean(image_array)) / std_dev
    else:
        normalized_image = image_array
    
    return normalized_image


def convert_dataset_to_images(
    csv_path: str,
    target_size: int = 128,
    instance_name_col: str = "Instance_Name",
    text_path_col: str = "Raw_Text_Path",
    output_col: str = "Image_Npy_Path"
) -> None:
    """
    Convert all instances in a dataset CSV to grayscale images.
    
    Reads a CSV file containing instance information, converts each instance's
    text file to a grayscale image, saves as .npy, and updates the CSV with
    image paths.
    
    Args:
        csv_path: Path to the dataset CSV file
        target_size: Target size for generated images (default: 128)
        instance_name_col: Name of column containing instance names
        text_path_col: Name of column containing paths to text files
        output_col: Name of column to add with image paths
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        KeyError: If required columns are missing from CSV
    """
    print("=== Image Conversion: Text to Grayscale ===")
    print(f"Input CSV: {os.path.abspath(csv_path)}")
    
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
    print(f"[2/3] Converting instances to {target_size}x{target_size} grayscale images...")
    image_paths = []
    
    for idx, row in df.iterrows():
        instance_name = row[instance_name_col]
        text_path = row[text_path_col]
        
        try:
            # Convert to image
            image_matrix = convert_text_to_grayscale_image(text_path, target_size)
            
            # Save as .npy
            npy_filename = f"{instance_name}_image.npy"
            npy_path = os.path.join(images_dir, npy_filename)
            np.save(npy_path, image_matrix)
            
            image_paths.append(npy_path)
            print(f"  [{idx + 1}/{total_instances}] {instance_name} -> {npy_filename}")
        
        except Exception as e:
            print(f"  [{idx + 1}/{total_instances}] ERROR processing {instance_name}: {e}")
            image_paths.append(None)
    
    # Update CSV with image paths
    print("[3/3] Updating CSV with image paths...")
    df[output_col] = image_paths
    df.to_csv(csv_path, index=False)
    
    successful = sum(1 for p in image_paths if p is not None)
    print(f"Conversion complete: {successful}/{total_instances} successful")
    print("=== Image Conversion Complete ===\n")


def generate_all_images(csv_path: str, target_size: int = 128) -> None:
    """
    Legacy wrapper function for backward compatibility.
    
    Converts all instances in a CSV to grayscale images.
    
    Args:
        csv_path: Path to the dataset CSV file
        target_size: Target size for generated images (default: 128)
    """
    convert_dataset_to_images(csv_path, target_size)