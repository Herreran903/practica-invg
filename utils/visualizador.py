"""
Deprecated wrapper for visualization.

This script is deprecated. Please use the new standardized CLIs instead:

- Images:  python -m utils.visualize_image <path_to_image>
- Tensors: python -m utils.visualize_tensor <path_to_tensor>

Behavior:
- If the provided path looks like a standard image (.png, .jpg, .jpeg, .bmp, .tif, .tiff),
  this wrapper forwards arguments to utils.visualize_image.
- Otherwise, it forwards to utils.visualize_tensor.

Exit codes:
- Non-zero on error consistent with the underlying tool.
"""

from __future__ import annotations

import logging
import os
import sys

try:
    # Absolute imports require utils to be a package; ensured by utils/__init__.py
    from utils.visualize_common import is_supported_image_ext
    from utils.visualize_image import main as image_main
    from utils.visualize_tensor import main as tensor_main
except Exception:
    # Fallback to relative imports if executed as part of the package
    try:
        from .visualize_common import is_supported_image_ext  # type: ignore
        from .visualize_image import main as image_main  # type: ignore
        from .visualize_tensor import main as tensor_main  # type: ignore
    except Exception as e:
        print(f"ERROR: Failed to import visualization modules: {e}", file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Print deprecation notice (stderr) but continue
    print(
        "DEPRECATION: utils/visualizador.py is deprecated.\n"
        "Use one of the new CLIs instead:\n"
        "  - Images:  python -m utils.visualize_image <path_to_image>\n"
        "  - Tensors: python -m utils.visualize_tensor <path_to_tensor>\n",
        file=sys.stderr,
    )

    # Route based on the first positional argument if present
    target = None
    for a in argv:
        # First non-option is assumed to be the path
        if not a.startswith("-"):
            target = a
            break

    # Default routing: if extension is an image, go to image_main; else tensor_main
    if target and is_supported_image_ext(target):
        image_main(argv)
    else:
        tensor_main(argv)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
