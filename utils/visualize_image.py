"""
Visualize common image files with a consistent CLI.

Supported formats: .png, .jpg, .jpeg, .bmp, .tif, .tiff

CLI
----
Positional:
- path: Path to the image file.

Options:
- --normalize               Apply min-max normalization to [0, 1] before display (default: off).
- --channel-order {auto,chw,hwc}
                            Channel order handling for 3D images (default: auto).
                            For typical PIL-loaded images this is already HWC.
- --cmap {gray,viridis,magma,none}
                            Colormap (default: none for RGB; gray for 2D).
- --title TEXT              Figure title.
- --save OUTPUT_PATH        Save the rendered figure to this path.
- --dpi INT                 Figure DPI (default: 120).
- --no-show                 Do not open interactive window (useful for CI).
- --verbose                 Enable INFO logging.

Examples
--------
python -m utils.visualize_image data/samples/cat.jpg --title "Sample" --save out.png --no-show
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

# numpy not required directly here

# Support both "python -m utils.visualize_image" (package context)
# and "python utils/visualize_image.py" (script context from repo root).
try:
    # Prefer absolute import when running as a script from the repo root
    from utils.visualize_common import (
        is_supported_image_ext,
        load_pil_image,
        prepare_for_display,
        render_figure,
    )
except Exception:
    # Fallback when executed as a module within the package
    from .visualize_common import (  # type: ignore
        is_supported_image_ext,
        load_pil_image,
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
    parser = argparse.ArgumentParser(
        description="Visualize an image file with a consistent CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=str, help="Path to the image file to visualize.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply min-max normalization to [0, 1] before display (default: off).",
    )
    parser.add_argument(
        "--channel-order",
        type=_validate_channel_order,
        default="auto",
        help="Channel order for 3D images.",
    )
    parser.add_argument(
        "--cmap",
        type=_validate_cmap,
        default="none",
        help="Colormap (ignored for RGB unless explicitly set).",
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

    if not is_supported_image_ext(args.path):
        print(
            "ERROR: Unsupported image format. Supported: .png, .jpg, .jpeg, .bmp, .tif, .tiff",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        arr = load_pil_image(args.path)
        logger.info("Loaded image %s with shape %s", args.path, getattr(arr, "shape", None))

        # Prepare
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
            # 3D: if user provided a specific cmap and last channel==1, we'll respect in renderer,
            # otherwise RGB will ignore colormap.
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
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()