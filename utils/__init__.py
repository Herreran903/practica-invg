"""
Utilities package for visualization and helper scripts.

Provides:
- visualize_image: Display standard image files via a consistent CLI.
- visualize_tensor: Display 2D/3D arrays/tensors from NumPy/PyTorch files.
- visualize_common: Shared helpers (loading, normalization, channel order, rendering).

Both visualize_image and visualize_tensor support running:
- As a module:  python -m utils.visualize_image  /  python -m utils.visualize_tensor
- As a script:  python utils/visualize_image.py  /  python utils/visualize_tensor.py
"""
