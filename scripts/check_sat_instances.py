#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check ASlib-style ARFF instance IDs against a directory of SAT instances.

Usage (from project root):

  python scripts/check_sat_instances.py \
      --arff data/sat/aslib/algorithm_runs.arff \
      --instances-root data/sat/instances

The script will:
  - Parse the ARFF file and extract the instance identifier column
  - Recursively scan the instances directory
  - Match each ARFF instance ID to files under the instances root
  - Print how many instances were found, how many were missing,
    and list the missing IDs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class ArffInstanceInfo:
    """Holds parsed information from an ARFF file."""

    arff_path: str
    id_attribute: str
    instance_ids: List[str]


def parse_arff_instance_ids(
    arff_path: str,
    id_attr: Optional[str] = None,
) -> ArffInstanceInfo:
    """
    Parse an ARFF file and extract instance IDs from the given attribute.

    If id_attr is None, tries to guess a suitable attribute name such as
    'instance_id', 'instance', or 'problem' (case-insensitive).
    """
    header_attrs: List[str] = []
    id_index: Optional[int] = None
    in_data = False
    instance_ids: List[str] = []

    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower = line.lower()

            if lower.startswith("@attribute"):
                # @ATTRIBUTE name type
                parts = line.split()
                if len(parts) < 3:
                    continue
                attr_name = parts[1].strip("'\"")
                header_attrs.append(attr_name)
                continue

            if lower.startswith("@data"):
                in_data = True
                # Decide which attribute to use as instance ID
                if id_index is None:
                    if id_attr is not None:
                        # Explicit attribute name
                        try:
                            id_index = [
                                i
                                for i, n in enumerate(header_attrs)
                                if n.lower() == id_attr.lower()
                            ][0]
                        except IndexError:
                            raise ValueError(
                                f"Attribute '{id_attr}' not found in {arff_path}. "
                                f"Available attributes: {header_attrs}"
                            )
                    else:
                        # Heuristic search
                        candidates = [
                            "instance_id",
                            "instance",
                            "problem",
                            "id",
                        ]
                        lowered = [n.lower() for n in header_attrs]
                        for cand in candidates:
                            if cand in lowered:
                                id_index = lowered.index(cand)
                                break
                        if id_index is None:
                            raise ValueError(
                                "Could not infer instance ID attribute. "
                                f"Available attributes: {header_attrs}"
                            )
                continue

            if in_data:
                # Data line: comma-separated values
                # ARFF in ASlib scenarios typically does not use commas in IDs.
                if line.startswith("{"):
                    # Sparse ARFF not expected for SAT12-INDU; bail out early.
                    raise ValueError(
                        "Sparse ARFF format detected; this helper expects dense rows."
                    )

                parts = [p.strip() for p in line.split(",")]
                if id_index is None or id_index >= len(parts):
                    continue
                raw_id = parts[id_index]
                # Strip quotes and whitespace
                full_id = raw_id.strip().strip("'\"")
                if not full_id or full_id == "?":
                    continue

                # In SAT12-ALL / SAT12-INDU the instance_id is often a *path* like:
                #   SAT_Competition2007/random/LargeSize/3SAT/v10000/unif-k3-r4.2-v10000-c42000-S421554531-04.cnf
                # We only want to use the *last component without extension* as ID:
                #   unif-k3-r4.2-v10000-c42000-S421554531-04
                last_seg = full_id.replace("\\", "/").split("/")[-1]
                base_name, _ext = os.path.splitext(last_seg)
                inst_id = base_name if base_name else full_id

                instance_ids.append(inst_id)

    if id_index is None:
        raise ValueError(
            f"Failed to determine instance ID attribute in {arff_path}."
        )

    return ArffInstanceInfo(
        arff_path=os.path.abspath(arff_path),
        id_attribute=header_attrs[id_index],
        instance_ids=instance_ids,
    )


def build_instance_name_index(instances_root: str) -> Dict[str, List[str]]:
    """
    Recursively scan instances_root and build an index:

        base_name (without extension) -> list of relative file paths
    """
    index: Dict[str, List[str]] = {}
    root_abs = os.path.abspath(instances_root)

    for dirpath, _dirnames, filenames in os.walk(root_abs):
        for fname in filenames:
            base, _ext = os.path.splitext(fname)
            rel_dir = os.path.relpath(dirpath, root_abs)
            rel_path = os.path.join(rel_dir, fname) if rel_dir != "." else fname
            index.setdefault(base, []).append(rel_path)

    return index


@dataclass
class MatchSummary:
    total_rows: int
    unique_ids: int
    matched_ids: int
    missing_ids: int
    matched_list: List[str]
    missing_list: List[str]


def match_instances(
    arff_info: ArffInstanceInfo,
    name_index: Dict[str, List[str]],
) -> MatchSummary:
    """Match ARFF instance IDs against the instances directory index."""
    ids = arff_info.instance_ids
    total_rows = len(ids)
    unique_ids_set: Set[str] = set(ids)

    matched: Set[str] = set()
    missing: Set[str] = set()

    for inst_id in unique_ids_set:
        if inst_id in name_index:
            matched.add(inst_id)
        else:
            missing.add(inst_id)

    return MatchSummary(
        total_rows=total_rows,
        unique_ids=len(unique_ids_set),
        matched_ids=len(matched),
        missing_ids=len(missing),
        matched_list=sorted(matched),
        missing_list=sorted(missing),
    )


def print_summary(
    arff_info: ArffInstanceInfo,
    summary: MatchSummary,
    instances_root: str,
    total_files_scanned: int,
    out_path: Optional[str] = None,
) -> None:
    """
    Pretty-print results and optionally save them to a text file.
    """
    lines: List[str] = []

    lines.append(f"Instances root       : {os.path.abspath(instances_root)}")
    lines.append(f"Total files scanned  : {total_files_scanned}")
    lines.append("")
    lines.append("=== ARFF / Instances Consistency Check ===")
    lines.append(f"ARFF file            : {arff_info.arff_path}")
    lines.append(f"Instance ID attribute: {arff_info.id_attribute}")
    lines.append(f"Total ARFF rows      : {summary.total_rows}")
    lines.append(f"Unique instance IDs  : {summary.unique_ids}")
    lines.append(f"Matched IDs          : {summary.matched_ids}")
    lines.append(f"Missing IDs          : {summary.missing_ids}")
    if summary.unique_ids:
        ratio = summary.matched_ids / summary.unique_ids
        lines.append(f"Match ratio          : {ratio:.3f}")
    lines.append("")

    if summary.matched_list:
        lines.append("Matched instance IDs (at least one file under instances root):")
        for inst_id in summary.matched_list:
            lines.append(f"  - {inst_id}")
    else:
        lines.append("No ARFF instance IDs could be matched to files.")
    lines.append("")

    if summary.missing_list:
        lines.append("Missing instance IDs (no matching file under instances root):")
        for inst_id in summary.missing_list:
            lines.append(f"  - {inst_id}")
    else:
        lines.append("All ARFF instance IDs have at least one matching file.")

    text = "\n".join(lines)
    print(text)

    if out_path:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"\nSummary written to: {os.path.abspath(out_path)}")
        except OSError as e:
            print(f"\n[WARN] Could not write summary to {out_path}: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that ASlib-style ARFF instance IDs have corresponding files "
            "under a given instances directory."
        )
    )
    parser.add_argument(
        "--arff",
        required=True,
        help="Path to algorithm_runs.arff (or similar) file.",
    )
    parser.add_argument(
        "--instances-root",
        default="data/sat/instances",
        help=(
            "Root directory containing SAT instances (searched recursively). "
            "Default: data/sat/instances"
        ),
    )
    parser.add_argument(
        "--id-attr",
        default=None,
        help=(
            "Name of the ARFF attribute holding the instance ID "
            "(default: auto-detect, e.g. instance_id or instance)."
        ),
    )
    parser.add_argument(
        "--out",
        default="check_sat_instances_result.txt",
        help=(
            "Optional path to a .txt file where the summary will be saved. "
            "Default: check_sat_instances_result.txt in the current directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    arff_info = parse_arff_instance_ids(args.arff, id_attr=args.id_attr)
    name_index = build_instance_name_index(args.instances_root)

    total_files = sum(len(v) for v in name_index.values())
    summary = match_instances(arff_info, name_index)

    print_summary(
        arff_info=arff_info,
        summary=summary,
        instances_root=args.instances_root,
        total_files_scanned=total_files,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()