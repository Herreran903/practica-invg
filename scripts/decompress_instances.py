#!/usr/bin/env python3
"""
Recursive decompressor for SAT/ASlib instance trees.

Goal:
- Walk a root directory recursively, find compressed instance files
  (*.cnf.gz, *.bz2, *.xz, *.zip, *.lzma, etc.)
- Produce a plaintext file without the compression extension next to the archive
  (e.g., foo.cnf.xz -> foo.cnf)
- Preserve the directory structure (operate in-place)
- Optionally delete the original archive after successful decompression
- Avoid overwriting existing plaintext files unless --overwrite is explicitly set

Usage examples:
  # Dry-run first (recommended): only prints what would be done
  python3 scripts/decompress_instances.py data/sat/instances/sc2012-application --dry-run

  # Decompress without deleting archives, skip files whose target already exists
  python3 scripts/decompress_instances.py data/sat/instances/sc2012-application

  # Decompress and delete original archives afterwards
  python3 scripts/decompress_instances.py data/sat/instances/sc2012-application --delete

  # Allow overwriting existing plaintext targets (use with care)
  python3 scripts/decompress_instances.py data/sat/instances/sc2012-application --overwrite

  # Handle .zip with multiple members by extracting the whole tree (rarely needed)
  python3 scripts/decompress_instances.py data/sat/instances/sc2012-application --extract-zip-multi

Precautions to avoid accidental overwrites:
- By default, if "foo.cnf" already exists, the tool will SKIP decompressing "foo.cnf.gz|bz2|xz|zip"
- Use --overwrite only if you know you want to replace existing targets
- Always consider running with --dry-run first to see actions without changing files
"""

from __future__ import annotations

import argparse
import bz2
import contextlib
import errno
import gzip
import lzma
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Tuple

COMPRESSED_EXTS = (".gz", ".bz2", ".xz", ".zip", ".lzma")
STREAM_EXTS = (".gz", ".bz2", ".xz", ".lzma")
ZIP_EXT = ".zip"


def iter_compressed_files(root: Path, include_exts: Iterable[str]) -> Iterable[Path]:
    exts = tuple(e.lower() for e in include_exts)
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if any(fn.lower().endswith(ext) for ext in exts):
                yield p


def safe_target_for_archive(archive: Path) -> Path:
    # Remove just the last compression suffix: foo.cnf.xz -> foo.cnf
    return archive.with_suffix("")


def make_parent_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_bytes(dest: Path, data_stream, chunk_size: int = 1024 * 1024) -> None:
    """
    Write bytes from data_stream to dest atomically using a temporary file.
    """
    make_parent_dirs(dest)
    with tempfile.NamedTemporaryFile(dir=str(dest.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            shutil.copyfileobj(data_stream, tmp, length=chunk_size)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            tmp_path.replace(dest)
        except Exception:
            try:
                tmp.close()
            finally:
                with contextlib.suppress(Exception):
                    tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            raise


def decompress_stream_file(src: Path, dest: Path, kind: str) -> None:
    """
    Decompress stream-based formats: gz, bz2, xz, lzma.
    """
    opener_map = {
        "gz": gzip.open,
        "bz2": bz2.open,
        "xz": lzma.open,
        "lzma": lzma.open,
    }
    opener = opener_map[kind]
    with opener(src, "rb") as f_in:
        atomic_write_bytes(dest, f_in)


def zip_single_member_info(zf: zipfile.ZipFile) -> Tuple[zipfile.ZipInfo | None, int]:
    members = [info for info in zf.infolist() if not info.is_dir()]
    return (members[0] if len(members) == 1 else None, len(members))


def decompress_zip(
    src: Path, dest: Path, overwrite: bool, dry_run: bool, extract_zip_multi: bool
) -> Tuple[int, int]:
    """
    Handle .zip files.
    - If the zip has exactly one file (not a directory), write it to 'dest'.
    - If multiple members:
        * If extract_zip_multi=True, extract all to parent directory (preserving internal structure)
        * Otherwise skip and warn
    Returns (extracted_files_count, skipped_files_count)
    """
    extracted, skipped = 0, 0
    with zipfile.ZipFile(src, "r") as zf:
        single, total_files = zip_single_member_info(zf)
        if single is not None:
            # Single-file zip; write to 'dest'
            if dest.exists() and not overwrite:
                print(f"SKIP (exists): {dest}")
                return (0, 1)
            action = f"unzip(single) -> {dest}"
            if dry_run:
                print(f"DRY-RUN: {action}")
                return (0, 0)
            # Stream extract
            with zf.open(single, "r") as f_in:
                atomic_write_bytes(dest, f_in)
            print(f"DONE: {action}")
            extracted += 1
        else:
            # Multiple files; either extract all, or skip.
            if not extract_zip_multi:
                print(
                    f"SKIP (.zip multi-member): {src} contains {total_files} files. Use --extract-zip-multi to extract the full tree."
                )
                return (0, 1)
            # Extract all into parent directory (preserving internal paths)
            action = f"unzip(multi) -> {src.parent} (members={total_files})"
            if dry_run:
                print(f"DRY-RUN: {action}")
                return (0, 0)
            zf.extractall(src.parent)
            print(f"DONE: {action}")
            extracted += total_files
    return (extracted, skipped)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recursively decompress SAT/ASlib instances in-place.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "root",
        type=Path,
        help="Root directory containing (possibly nested) compressed instances",
    )
    p.add_argument(
        "--delete",
        "--delete-original",
        action="store_true",
        help="Delete original archives after successful decompression",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target files if they already exist (use with care)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Show actions without writing files"
    )
    p.add_argument(
        "--extract-zip-multi",
        action="store_true",
        help="Extract .zip with multiple members (preserves internal folder structure)",
    )
    p.add_argument(
        "--ext",
        dest="exts",
        nargs="*",
        default=list(COMPRESSED_EXTS),
        help="Compression extensions to search (case-insensitive). Examples: .gz .bz2 .xz .zip .lzma",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root: Path = args.root

    if not root.exists() or not root.is_dir():
        print(
            f"ERROR: Root directory does not exist or is not a directory: {root}",
            file=sys.stderr,
        )
        return errno.ENOENT

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.exts)

    total_found = 0
    total_done = 0
    total_skipped = 0
    total_errors = 0

    print("=" * 75)
    print(f"Decompress recursively: root={root}")
    print(f"Extensions: {', '.join(exts)}")
    print(
        f"Options: delete={args.delete} | overwrite={args.overwrite} | dry_run={args.dry_run} | extract_zip_multi={args.extract_zip_multi}"
    )
    print("=" * 75)

    for archive in iter_compressed_files(root, exts):
        total_found += 1

        # Decide target for stream formats; zip handling may differ
        dest = safe_target_for_archive(archive)

        # If an uncompressed file already exists and overwrite is false, skip for safety
        if (
            archive.suffix.lower() in STREAM_EXTS
            and dest.exists()
            and not args.overwrite
        ):
            print(f"SKIP (exists): {dest}")
            total_skipped += 1
            continue

        try:
            if archive.suffix.lower() in (".gz",):
                if args.dry_run:
                    print(f"DRY-RUN: gunzip -> {dest}")
                else:
                    decompress_stream_file(archive, dest, "gz")
                    print(f"DONE: gunzip -> {dest}")
                total_done += 1

            elif archive.suffix.lower() in (".bz2",):
                if args.dry_run:
                    print(f"DRY-RUN: bunzip2 -> {dest}")
                else:
                    decompress_stream_file(archive, dest, "bz2")
                    print(f"DONE: bunzip2 -> {dest}")
                total_done += 1

            elif archive.suffix.lower() in (".xz",):
                if args.dry_run:
                    print(f"DRY-RUN: unxz -> {dest}")
                else:
                    decompress_stream_file(archive, dest, "xz")
                    print(f"DONE: unxz -> {dest}")
                total_done += 1

            elif archive.suffix.lower() in (".lzma",):
                if args.dry_run:
                    print(f"DRY-RUN: unlzma -> {dest}")
                else:
                    decompress_stream_file(archive, dest, "lzma")
                    print(f"DONE: unlzma -> {dest}")
                total_done += 1

            elif archive.suffix.lower() == ZIP_EXT:
                if not args.overwrite and dest.exists():
                    # Special case: if .zip has single member, target would be 'dest' (archive without .zip)
                    # Respect overwrite safety.
                    print(f"SKIP (exists): {dest}")
                    total_skipped += 1
                    continue
                done, skipped = decompress_zip(
                    src=archive,
                    dest=dest,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    extract_zip_multi=args.extract_zip_multi,
                )
                total_done += done
                total_skipped += skipped

            else:
                # Unknown extension (shouldn't happen due to filter)
                print(f"SKIP (unknown extension): {archive}")
                total_skipped += 1
                continue

            # Optionally delete the original archive after successful action
            if not args.dry_run and args.delete:
                try:
                    archive.unlink()
                    print(f"DELETED: {archive}")
                except Exception as de:
                    print(f"WARNING: Could not delete archive {archive}: {de}")

        except Exception as e:
            total_errors += 1
            print(f"ERROR: Failed to decompress {archive}: {e}")

    # Summary
    print("\n" + "=" * 75)
    print("Summary")
    print("=" * 75)
    print(f"Archives found:      {total_found}")
    print(f"Decompressed done:   {total_done}")
    print(f"Skipped (safe/exist):{total_skipped}")
    print(f"Errors:              {total_errors}")

    # Quick check: how many compressed files still remain?
    remaining = sum(1 for _ in iter_compressed_files(root, exts))
    print(f"Remaining compressed (by extensions): {remaining}")
    if remaining > 0:
        print(
            "Hint: If many compressed files remain, rerun without --dry-run and consider --delete to remove archives."
        )
    print("=" * 75)

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
