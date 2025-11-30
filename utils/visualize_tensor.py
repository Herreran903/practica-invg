# -*- coding: utf-8 -*-
"""
Unified visualization CLI for the project.

This single script replaces/absorbs the functionality that previously lived in:
- utils/visualize_common.py
- utils/visualize_tensor.py
- utils/visualizador.py

Capabilities
------------

- Visualize **NumPy / tensor-like artifacts** (`.npy`, `.npz`, `.pt`, `.pth`, `.pkl`)
  produced by SAT/JSSP data generation and training pipelines.
- Visualize **standard image files** (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`)
  directly via Pillow.
- Normalize values for display (min-max to [0, 1]) with per-channel support.
- Handle 2D (grayscale) and 3D (multi-channel) arrays, infiriendo el orden de canales.

Examples
--------

# Visualizar un .npy de tensores/imágenes
python -m utils.visualize_tensor path/to/file.npy

# Visualizar un .png / .jpg
python -m utils.visualize_tensor path/to/image.png

# Especificar key para .npz o .pt/.pth/.pkl (state dict)
python -m utils.visualize_tensor checkpoints/weights.pt --key features --normalize

# No mostrar ventana (sólo guardar)
python -m utils.visualize_tensor file.npy --save out.png --no-show

CLI
---

Posicional:
- path: Ruta al archivo a visualizar (.npy, .npz, .pt, .pth, .pkl, .png, .jpg, ...)

Opciones:
- --key NAME                 Key opcional para .npz o mappings .pt/.pth/.pkl
- --normalize / --no-normalize
                            Normalización min-max a [0, 1] (por defecto: activada)
- --channel-order {auto,chw,hwc}
                            Orden de canales para tensores 3D (por defecto: auto)
- --cmap {gray,viridis,magma,none}
                            Colormap (default: gray para 2D; none para RGB)
- --title TEXT              Título de la figura
- --save OUTPUT_PATH        Guardar figura renderizada en esta ruta
- --dpi INT                 DPI de la figura (default: 120)
- --no-show                 No abrir ventana interactiva (útil en CI)
- --verbose                 Logging a nivel INFO

"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from typing import Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File / image helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def is_supported_image_ext(path: str) -> bool:
    """Return True if file extension looks like a standard image."""
    return _ext(path) in _IMAGE_EXTS


def load_pil_image(path: str) -> np.ndarray:
    """
    Load an image using Pillow and return as numpy array in either HxW (grayscale)
    or HxWxC (RGB/RGBA).
    """
    with Image.open(path) as img:
        if img.mode in ("RGB", "RGBA", "L"):
            arr = np.array(img)
        else:
            # Convert unknown modes to RGB for display consistency
            arr = np.array(img.convert("RGB"))
    return arr


# ---------------------------------------------------------------------------
# Tensor loaders (.npy, .npz, .pt/.pth, .pkl)
# ---------------------------------------------------------------------------


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


def _select_from_mapping(
    mapping: Mapping, key: Optional[str]
) -> Optional[np.ndarray]:
    """
    Try to select a tensor/array from a mapping (e.g., state dict) using an optional key.
    If no key is given and exactly one candidate tensor/array exists, return it.
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


def load_tensor_like(path: str, key: Optional[str] = None) -> np.ndarray:
    """
    Load a tensor-like object from .npy, .npz, .pt/.pth, or .pkl.

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
                        f"Key '{key}' not found in npz file. "
                        f"Available keys: {list(data.files)}"
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
        obj = torch.load(path, map_location="cpu")  # type: ignore[name-defined]

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

        # Object with state_dict
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

        raise ValueError(
            "Unsupported object in pickle; expected ndarray or torch.Tensor."
        )

    raise ValueError(
        f"Unsupported tensor file extension: '{ext}'. "
        "Supported: .npy, .npz, .pt, .pth, .pkl"
    )


# ---------------------------------------------------------------------------
# Normalization and channel order helpers
# ---------------------------------------------------------------------------


def minmax_normalize(arr: np.ndarray, per_channel: bool = False) -> np.ndarray:
    """
    Min-max normalize array to [0, 1]. Safe for constant arrays.
    """
    x = arr.astype(np.float32, copy=False)

    if x.ndim == 2 or not per_channel:
        mn = float(np.nanmin(x))
        mx = float(np.nanmax(x))
        rng = mx - mn
        if rng == 0.0:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / rng

    if x.ndim == 3:
        out = np.empty_like(x, dtype=np.float32)
        if x.shape[-1] in (1, 3, 4):  # HWC
            for c in range(x.shape[-1]):
                ch = x[..., c]
                mn = float(np.nanmin(ch))
                mx = float(np.nanmax(ch))
                rng = mx - mn
                out[..., c] = 0.0 if rng == 0.0 else (ch - mn) / rng
            return out
        else:  # CHW
            for c in range(x.shape[0]):
                ch = x[c, ...]
                mn = float(np.nanmin(ch))
                mx = float(np.nanmax(ch))
                rng = mx - mn
                out[c, ...] = 0.0 if rng == 0.0 else (ch - mn) / rng
            return out

    return x


def infer_channel_order(arr: np.ndarray) -> str:
    """
    Infer channel order for 3D arrays.

    Heuristic:
    - If shape[0] in {1,3,4} and shape[-1] not in {1,3,4} -> CHW
    - Else if shape[-1] in {1,3,4} -> HWC
    - Else -> fallback HWC (warn)
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


def prepare_for_display(
    arr: np.ndarray,
    normalize: bool,
    channel_order: str,
    normalize_per_channel: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    Prepare array for display.

    Returns
    -------
    (np.ndarray, bool)
        (display_array, is_rgb_like)
    """
    x = np.asarray(arr)

    if x.ndim == 2:
        y = x.astype(np.float32, copy=False)
        if normalize:
            y = minmax_normalize(y, per_channel=False)
        return y, False

    if x.ndim == 3:
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
    """
    plt.figure(dpi=dpi)

    if img.ndim == 2:
        plt.imshow(img, cmap=cmap or "gray")
    elif img.ndim == 3:
        if img.shape[-1] == 1:
            plt.imshow(img[..., 0], cmap=cmap or "gray")
        else:
            if cmap and cmap.lower() != "none":
                logger.info(
                    "Colormap specified for RGB image; matplotlib may ignore it."
                )
            to_show = img
            if to_show.dtype.kind == "f":
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

    if not no_show and os.environ.get("CI", "").lower() != "true":
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validate_cmap(cmap: str) -> str:
    allowed = {"gray", "viridis", "magma", "none"}
    v = cmap.lower()
    if v not in allowed:
        raise argparse.ArgumentTypeError(
            f"Invalid cmap '{cmap}'. Allowed: {sorted(allowed)}"
        )
    return v


def _validate_channel_order(order: str) -> str:
    allowed = {"auto", "chw", "hwc"}
    v = order.lower()
    if v not in allowed:
        raise argparse.ArgumentTypeError(
            f"Invalid channel order '{order}'. Allowed: {sorted(allowed)}"
        )
    return v


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    # BooleanOptionalAction allows --normalize / --no-normalize, default True
    try:
        action_bool_opt = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        class action_bool_opt(argparse.Action):  # type: ignore
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(
                    namespace, self.dest, option_string and "no-" not in option_string
                )

    parser = argparse.ArgumentParser(
        description="Visualize a tensor-like or image file as an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=str, help="Path to file to visualize.")
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Optional key for .npz or mapping-like .pt/.pth/.pkl files.",
    )
    parser.add_argument(
        "--normalize",
        action=action_bool_opt,
        default=True,
        help="Apply min-max normalization to [0, 1] before display.",
    )
    parser.add_argument(
        "--channel-order",
        type=_validate_channel_order,
        default="auto",
        help="Channel order for 3D tensors.",
    )
    parser.add_argument(
        "--cmap",
        type=_validate_cmap,
        default="none",
        help="Colormap for single-channel images (ignored for RGB unless explicitly set).",
    )
    parser.add_argument("--title", type=str, default=None, help="Figure title.")
    parser.add_argument(
        "--save", type=str, default=None, help="Save the rendered figure to this path."
    )
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window (useful in CI).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable INFO-level logging output."
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if not os.path.exists(args.path):
        print(f"ERROR: File does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Branch: standard image file
        if is_supported_image_ext(args.path):
            arr = load_pil_image(args.path)
            logger.info(
                "Loaded image %s with shape %s",
                args.path,
                getattr(arr, "shape", None),
            )
        else:
            # Tensor-like file
            arr = load_tensor_like(args.path, key=args.key)
            logger.info(
                "Loaded tensor %s with shape %s",
                args.path,
                getattr(arr, "shape", None),
            )

        if not isinstance(arr, np.ndarray):
            raise ValueError("Loaded object is not a NumPy array")

        if arr.ndim not in (2, 3):
            raise ValueError(
                f"Tensor cannot be interpreted as a 2D or 3D image (ndim={arr.ndim})."
            )

        disp, _ = prepare_for_display(
            arr,
            normalize=args.normalize,
            channel_order=args.channel_order,
            normalize_per_channel=True,
        )

        # Decide cmap
        cmap = None
        if disp.ndim == 2:
            cmap = None if args.cmap == "none" else args.cmap
            if args.cmap == "none":
                cmap = "gray"
        else:
            if args.cmap != "none":
                cmap = args.cmap

        render_figure(
            disp,
            title=args.title,
            cmap=cmap,
            dpi=args.dpi,
            save=args.save,
            no_show=args.no_show,
        )

    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except (FileNotFoundError, ImportError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected failure: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
