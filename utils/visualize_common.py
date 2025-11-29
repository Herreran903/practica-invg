"""
Common helpers for visualization utilities.

Provides loading, normalization, channel-order inference, and rendering helpers
shared by visualize_image.py and visualize_tensor.py.

Docstring style: NumPy style.
"""

from __future__ import annotations

import io
import logging
import os
import pickle
from typing import Iterable, Mapping, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ---------- Logging ----------

logger = logging.getLogger(__name__)


# ---------- File type helpers ----------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def is_supported_image_ext(path: str) -> bool:
    """
    Check whether the file path has a supported image extension.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    bool
        True if extension is one of supported image formats.
    """
    return _ext(path) in _IMAGE_EXTS


# ---------- Loading helpers ----------

def load_pil_image(path: str) -> np.ndarray:
    """
    Load an image using Pillow and return as numpy array in either HxW (grayscale)
    or HxWxC (RGB/RGBA).

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image array.
    """
    with Image.open(path) as img:
        # Preserve existing mode: if RGB/RGBA keep as is; if L keep grayscale
        if img.mode in ("RGB", "RGBA", "L"):
            arr = np.array(img)
        else:
            # Convert unknown modes to RGB for display consistency
            arr = np.array(img.convert("RGB"))
    return arr


def _lazy_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception as e:
        raise ImportError(
            "PyTorch is required to load .pt/.pth files but is not installed. "
            "Install with: pip install torch"
        ) from e


def _to_numpy_from_torch_tensor(t):
    torch = _lazy_import_torch()
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return None


def _select_from_mapping(mapping: Mapping, key: Optional[str]) -> Union[np.ndarray, None]:
    """
    Try to select a tensor/array from a mapping (e.g., state dict) using an optional key.
    If no key is given, and exactly one candidate tensor/array exists, return it.
    Otherwise return None.
    """
    candidates = []
    for k, v in mapping.items():
        # Torch Tensor?
        try:
            arr = _to_numpy_from_torch_tensor(v)
            if arr is not None:
                candidates.append((k, arr))
                continue
        except Exception:
            pass
        # NumPy array?
        if isinstance(v, np.ndarray):
            candidates.append((k, v))

    if key is not None:
        if key in mapping:
            v = mapping[key]
            arr = _to_numpy_from_torch_tensor(v)
            if arr is not None:
                return arr
            if isinstance(v, np.ndarray):
                return v
        return None

    if len(candidates) == 1:
        return candidates[0][1]

    return None


def load_tensor_like(
    path: str,
    key: Optional[str] = None,
) -> np.ndarray:
    """
    Load a tensor-like object from .npy, .npz, .pt/.pth, or .pkl.

    Parameters
    ----------
    path : str
        File path.
    key : str, optional
        Optional key for selecting an array/tensor from .npz or mapping-like .pt/.pth/.pkl.

    Returns
    -------
    np.ndarray
        Loaded array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unsupported or cannot be interpreted.
    ImportError
        If torch is required but missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = _ext(path)

    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        if not isinstance(arr, np.ndarray):
            raise ValueError(f".npy file did not contain a numpy array: {path}")
        return arr

    if ext == ".npz":
        with np.load(path) as data:
            if key:
                if key not in data.files:
                    raise ValueError(
                        f"Key '{key}' not found in npz file. Available keys: {list(data.files)}"
                    )
                return data[key]
            if len(data.files) == 1:
                return data[data.files[0]]
            raise ValueError(
                f".npz contains multiple arrays {list(data.files)}. "
                f"Provide --key to select one."
            )

    if ext in (".pt", ".pth"):
        torch = _lazy_import_torch()
        obj = torch.load(path, map_location="cpu")

        # Direct tensor
        arr = _to_numpy_from_torch_tensor(obj)
        if arr is not None:
            return arr

        # Mapping-like (e.g., state dict)
        if isinstance(obj, Mapping):
            selected = _select_from_mapping(obj, key)
            if selected is not None:
                return selected
            raise ValueError(
                "Could not infer which tensor to visualize from .pt/.pth. "
                "Provide --key to select one. Available keys might include "
                f"{list(obj.keys())[:10]}..."
            )

        # Single object wrapping tensor?
        if hasattr(obj, "state_dict"):
            state = obj.state_dict()
            selected = _select_from_mapping(state, key)
            if selected is not None:
                return selected
            raise ValueError(
                "Object state_dict did not yield a unique tensor. Provide --key."
            )

        raise ValueError("Unsupported .pt/.pth content: expected a Tensor or mapping.")

    if ext == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)

        # NumPy
        if isinstance(obj, np.ndarray):
            return obj

        # Torch Tensor
        try:
            arr = _to_numpy_from_torch_tensor(obj)
            if arr is not None:
                return arr
        except Exception:
            pass

        # Mapping-like
        if isinstance(obj, Mapping):
            selected = _select_from_mapping(obj, key)
            if selected is not None:
                return selected
            raise ValueError(
                "Pickle contains multiple arrays/tensors. Provide --key to select one."
            )

        raise ValueError("Unsupported object in pickle; expected ndarray or torch.Tensor.")

    raise ValueError(
        f"Unsupported tensor file extension: '{ext}'. "
        "Supported: .npy, .npz, .pt, .pth, .pkl"
    )


# ---------- Normalization ----------

def minmax_normalize(
    arr: np.ndarray,
    per_channel: bool = False,
) -> np.ndarray:
    """
    Min-max normalize array to [0, 1]. Safe for constant arrays.

    Parameters
    ----------
    arr : np.ndarray
        Input array (2D or 3D).
    per_channel : bool
        If True and arr is 3D, normalize each channel independently.

    Returns
    -------
    np.ndarray
        Normalized float32 array in [0, 1].
    """
    x = arr.astype(np.float32, copy=False)

    if x.ndim == 2 or not per_channel:
        mn = float(np.nanmin(x))
        mx = float(np.nanmax(x))
        rng = mx - mn
        if rng == 0.0:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / rng

    # 3D per-channel
    if x.ndim == 3:
        # Assume channels is either first or last; we do not transpose here,
        # call-site should ensure consistent channel order if needed.
        # We normalize along spatial dims.
        # Choose last axis as channel by default; caller should pass HWC here for per-channel.
        out = np.empty_like(x, dtype=np.float32)
        if x.shape[-1] in (1, 3, 4):
            # Channels last (HWC)
            for c in range(x.shape[-1]):
                ch = x[..., c]
                mn = float(np.nanmin(ch))
                mx = float(np.nanmax(ch))
                rng = mx - mn
                out[..., c] = 0.0 if rng == 0.0 else (ch - mn) / rng
            return out
        else:
            # Channels first (CHW)
            for c in range(x.shape[0]):
                ch = x[c, ...]
                mn = float(np.nanmin(ch))
                mx = float(np.nanmax(ch))
                rng = mx - mn
                out[c, ...] = 0.0 if rng == 0.0 else (ch - mn) / rng
            return out

    # Fallback
    return x


# ---------- Channel order inference ----------

def infer_channel_order(arr: np.ndarray) -> str:
    """
    Infer channel order for 3D arrays.

    Heuristic:
    - If shape[0] in {1,3,4} and shape[-1] not in {1,3,4} -> CHW
    - Else if shape[-1] in {1,3,4} -> HWC
    - Else -> fallback HWC (warn)

    Parameters
    ----------
    arr : np.ndarray
        Input 3D array.

    Returns
    -------
    str
        'chw' or 'hwc'
    """
    if arr.ndim != 3:
        raise ValueError("infer_channel_order expects a 3D array")

    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        return "chw"
    if arr.shape[-1] in (1, 3, 4):
        return "hwc"

    logger.warning(
        "Could not confidently infer channel order for shape %s. Falling back to HWC.",
        arr.shape,
    )
    return "hwc"


def ensure_hwc(arr: np.ndarray, channel_order: str) -> np.ndarray:
    """
    Ensure array is HxWxC for 3D arrays.

    Parameters
    ----------
    arr : np.ndarray
        Input array (3D).
    channel_order : str
        'chw', 'hwc', or 'auto'.

    Returns
    -------
    np.ndarray
        HWC array.
    """
    if arr.ndim != 3:
        return arr

    co = channel_order.lower()
    if co == "auto":
        co = infer_channel_order(arr)

    if co == "chw":
        return np.transpose(arr, (1, 2, 0))
    elif co == "hwc":
        return arr
    else:
        raise ValueError(f"Invalid channel order: {channel_order}")


# ---------- Rendering helpers ----------

def prepare_for_display(
    arr: np.ndarray,
    normalize: bool,
    channel_order: str,
    normalize_per_channel: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    Prepare array for display.

    Parameters
    ----------
    arr : np.ndarray
        Input array, 2D or 3D.
    normalize : bool
        Apply min-max normalization to [0, 1].
    channel_order : str
        'auto', 'chw', or 'hwc'. Only relevant for 3D arrays.
    normalize_per_channel : bool
        For 3D, normalize each channel separately when True.

    Returns
    -------
    (np.ndarray, bool)
        Tuple of (display_array, is_rgb_like).
        is_rgb_like is True if array looks like RGB/RGBA in HWC with channel last in {3,4}.
    """
    x = np.asarray(arr)

    if x.ndim == 2:
        y = x.astype(np.float32, copy=False)
        if normalize:
            y = minmax_normalize(y, per_channel=False)
        return y, False

    if x.ndim == 3:
        # Bring to HWC for display
        y = ensure_hwc(x, channel_order=channel_order)
        y = y.astype(np.float32, copy=False)
        if normalize:
            y = minmax_normalize(y, per_channel=normalize_per_channel)
        is_rgb = y.shape[-1] in (3, 4)
        return y, is_rgb

    raise ValueError(f"Tensor cannot be interpreted as 2D or 3D image (ndim={x.ndim}).")


def render_figure(
    img: np.ndarray,
    *,
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    dpi: int = 120,
    save: Optional[str] = None,
    no_show: bool = False,
) -> None:
    """
    Render and optionally save/show a figure.

    Parameters
    ----------
    img : np.ndarray
        Image array. 2D for grayscale or HWC for color.
    title : str, optional
        Figure title.
    cmap : str, optional
        Matplotlib colormap name for 2D images. Ignored for RGB unless explicitly requested.
    dpi : int
        Figure DPI.
    save : str, optional
        Output path to save the figure.
    no_show : bool
        If True, do not open interactive window.
    """
    plt.figure(dpi=dpi)

    if img.ndim == 2:
        plt.imshow(img, cmap=cmap or "gray")
    elif img.ndim == 3:
        # If explicitly set cmap and channels==1, allow plotting as 2D
        if img.shape[-1] == 1:
            plt.imshow(img[..., 0], cmap=cmap or "gray")
        else:
            # For RGB/RGBA, ignore cmap by default
            if cmap and cmap.lower() != "none":
                logger.info("Colormap specified for RGB image; matplotlib will ignore it.")
            # Ensure values in [0,1] for float images
            to_show = img
            if to_show.dtype.kind == "f":
                # Clip to [0,1] to avoid warnings
                to_show = np.clip(to_show, 0.0, 1.0)
            plt.imshow(to_show)
    else:
        raise ValueError("render_figure expects 2D or 3D array")

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        plt.savefig(save, bbox_inches="tight", dpi=dpi)

    if not no_show and not os.environ.get("CI", "").lower() == "true":
        plt.show()
    else:
        plt.close()