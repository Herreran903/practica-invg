"""
Visualize tensor-like files (.npy, .npz, .pt, .pth, .pkl) with a consistent CLI.

Supported formats:
- NumPy: .npy, .npz
- PyTorch: .pt, .pth
- Pickle: .pkl (containing a NumPy array or torch.Tensor)

CLI
----
Positional:
- path: Path to the tensor file to visualize.

Options:
- --key NAME                 Optional key for .npz or mapping-like .pt/.pth/.pkl files.
- --normalize / --no-normalize
                            Min-max normalization to [0, 1] (default: on).
- --channel-order {auto,chw,hwc}
                            Channel order handling for 3D tensors (default: auto).
- --cmap {gray,viridis,magma,none}
                            Colormap (default: gray for 2D; none for RGB).
- --title TEXT              Figure title.
- --save OUTPUT_PATH        Save the rendered figure to this path.
- --dpi INT                 Figure DPI (default: 120).
- --no-show                 Do not open interactive window (useful for CI).
- --verbose                 Enable INFO logging.

Examples
--------
python -m utils.visualize_tensor runs/feat.npy --normalize --channel-order chw
python -m utils.visualize_tensor runs/weights.pt --key features --normalize --no-show
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np

from .visualize_common import (
    load_tensor_like,
    prepare_for_display,
    render_figure,
)

logger = logging.getLogger(__name__)


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
    except Exception:
        # Fallback for very old Python (not expected per project requirements)
        class action_bool_opt(argparse.Action):  # type: ignore
            def __call__(self, parser, namespace, values, option_string=None):
                setattr(
                    namespace, self.dest, option_string and "no-" not in option_string
                )

    parser = argparse.ArgumentParser(
        description="Visualize a tensor-like file (.npy/.npz/.pt/.pth/.pkl) as an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=str, help="Path to the tensor file to visualize.")
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
        help="Apply min-max normalization to [0, 1] before display (default: on).",
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

    # Logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # Validation
    if not os.path.exists(args.path):
        print(f"ERROR: File does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)

    try:
        arr = load_tensor_like(args.path, key=args.key)
        logger.info(
            "Loaded tensor %s with shape %s", args.path, getattr(arr, "shape", None)
        )

        # Check dimensionality
        if not isinstance(arr, np.ndarray):
            raise ValueError("Loaded object is not a NumPy array")

        if arr.ndim not in (2, 3):
            raise ValueError(
                f"Tensor cannot be interpreted as a 2D or 3D image (ndim={arr.ndim})."
            )

        # Prepare for display
        disp, is_rgb = prepare_for_display(
            arr,
            normalize=args.normalize,
            channel_order=args.channel_order,
            normalize_per_channel=True,
        )

        # Colormap default behavior:
        # - If 2D image: default to gray unless user specified otherwise.
        # - If RGB: ignore cmap unless explicitly set (we keep 'none' by default).
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
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected failure: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
