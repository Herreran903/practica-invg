# -*- coding: utf-8 -*-
"""
Dataset imbalance inspection utilities.

This script helps analyze possible label/solver imbalance in the CSVs
produced by the SAT and JSSP data-generation pipelines.

It focuses on solver-related columns, typically:
  - *_Runtime_s
  - *_Score_S_rel
  - *_Status
  - Winner_Key (for some JSSP generated datasets)

Now it also takes solver status into account:
  - For classification-style argmin, solvers with invalid status are treated
    as having infinite runtime (i.e., nunca ganan).
  - For multilabel viability, a solver is viable sólo si:
        runtime < time_limit  AND  status NO es inválido.

Usage examples:

  python -m utils.dataset_imbalance \
      --csv data/jssp/datasets/jsp_cnn_data_images/ground_truth_jsp_generated_dataset.csv

  python -m utils.dataset_imbalance \
      --csv data/sat/datasets/sat_ground_truth.csv \
      --time-limit 1200

You can customize what counts as valid/invalid with:
  --valid-statuses "ok,sat,unsat"
  --invalid-statuses "timeout,time_out,timedout,memout,crash,error,fail"
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------


def detect_solver_cols(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect solver-related columns in a CSV.

    Returns:
        {
          "runtime": [cols ending with '_Runtime_s'],
          "score":   [cols ending with '_Score_S_rel'],
          "status":  [cols ending with '_Status'],
        }
    """
    runtime_cols = sorted([c for c in df.columns if c.endswith("_Runtime_s")])
    score_cols = sorted([c for c in df.columns if c.endswith("_Score_S_rel")])
    status_cols = sorted([c for c in df.columns if c.endswith("_Status")])

    return {"runtime": runtime_cols, "score": score_cols, "status": status_cols}


def status_lists_from_args(
    valid_statuses_arg: str, invalid_statuses_arg: str
) -> Tuple[List[str], List[str]]:
    """Parse CLI strings into canonical lowercase status lists."""
    def _split(s: str) -> List[str]:
        return [t.strip().lower() for t in s.split(",") if t.strip()]

    valid = _split(valid_statuses_arg)
    invalid = _split(invalid_statuses_arg)
    return valid, invalid


def status_column_for_runtime(runtime_col: str) -> str:
    """Map 'solver_Runtime_s' -> 'solver_Status'."""
    return runtime_col.replace("_Runtime_s", "_Status")


def build_status_masks(
    df: pd.DataFrame,
    runtime_cols: List[str],
    valid_statuses: List[str],
    invalid_statuses: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build boolean masks [N, C] for valid / invalid statuses.

    valid_mask[i, j]   = True  if row i, solver j has status in valid_statuses
    invalid_mask[i, j] = True  if row i, solver j has status in invalid_statuses
    """
    n = len(df)
    c = len(runtime_cols)
    valid_mask = np.zeros((n, c), dtype=bool)
    invalid_mask = np.zeros((n, c), dtype=bool)

    for j, rt_col in enumerate(runtime_cols):
        st_col = status_column_for_runtime(rt_col)
        if st_col not in df.columns:
            continue

        # Convert to lowercase strings; NaN -> "nan"
        svals = df[st_col].astype(str).str.strip().str.lower()

        valid_mask[:, j] = svals.isin(valid_statuses).to_numpy()
        invalid_mask[:, j] = svals.isin(invalid_statuses).to_numpy()

    return valid_mask, invalid_mask


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize_winner_key(df: pd.DataFrame) -> None:
    """If 'Winner_Key' exists, print its frequency distribution."""
    if "Winner_Key" not in df.columns:
        return

    print("\n=== Winner_Key distribution ===")
    counts = df["Winner_Key"].value_counts(dropna=False)
    total = len(df)
    for key, cnt in counts.items():
        pct = 100.0 * cnt / total if total > 0 else 0.0
        print(f"  {repr(key):15s} : {cnt:6d} ({pct:6.2f}%)")


def summarize_best_solver_classification(
    df: pd.DataFrame,
    runtime_cols: List[str],
    valid_statuses: List[str],
    invalid_statuses: List[str],
) -> None:
    """
    Treat each instance as a classification example:
    - label = argmin over *_Runtime_s columns
    - rows where all effective runtimes are inf are skipped
    - invalid statuses (timeout, crash, etc.) are treated as inf
    """
    if not runtime_cols:
        print("\nNo *_Runtime_s columns found; skipping classification-like summary.")
        return

    print("\n=== Classification-style label distribution (argmin runtime, invalid=∞) ===")

    # Raw runtimes: [N, C]
    runtimes = df[runtime_cols].astype(float).values
    n, c = runtimes.shape

    # Build status masks
    _, invalid_mask = build_status_masks(df, runtime_cols, valid_statuses, invalid_statuses)

    # Effective runtimes: NaN / invalid -> inf
    effective = runtimes.copy()
    # NaN / non-finite -> inf
    non_finite_mask = ~np.isfinite(effective)
    effective[non_finite_mask] = np.inf
    # Invalid statuses -> inf
    effective[invalid_mask] = np.inf

    finite_mask = np.isfinite(effective)
    valid_rows = finite_mask.any(axis=1)
    if not valid_rows.any():
        print("No rows with finite effective runtimes; nothing to summarize.")
        return

    valid_effective = effective[valid_rows]
    best_idx = np.argmin(valid_effective, axis=1)
    total = best_idx.size

    unique, counts = np.unique(best_idx, return_counts=True)
    for idx, cnt in zip(unique, counts):
        col = runtime_cols[int(idx)]
        pct = 100.0 * cnt / total
        print(f"  {col:30s} : {cnt:6d} ({pct:6.2f}%)")


def summarize_multilabel_viability(
    df: pd.DataFrame,
    runtime_cols: List[str],
    time_limit_s: float,
    valid_statuses: List[str],
    invalid_statuses: List[str],
) -> None:
    """
    Multilabel-style view:
      - A solver is "viable" for an instance if:
            runtime < time_limit_s AND status not in invalid_statuses
      - Report per-solver viability frequency
      - Report distribution of '#viable solvers per instance'
    """
    if not runtime_cols:
        print("\nNo *_Runtime_s columns found; skipping multilabel-like summary.")
        return

    print(
        f"\n=== Multilabel-style viability (runtime < {time_limit_s:.1f}s "
        f"AND status not invalid) ==="
    )

    runtimes = df[runtime_cols].astype(float).values  # [N, C]
    finite_mask = np.isfinite(runtimes)

    valid_mask, invalid_mask = build_status_masks(
        df, runtime_cols, valid_statuses, invalid_statuses
    )

    # A solver is "usable" if:
    #   - runtime is finite
    #   - NOT invalid (status not in invalid_statuses)
    usable_mask = finite_mask & ~invalid_mask

    # Viable = usable and runtime < time_limit
    viable_mask = usable_mask & (runtimes < time_limit_s)

    total_instances = runtimes.shape[0]

    print("\nPer-solver viability:")
    solver_viable_counts = viable_mask.sum(axis=0)
    for col, cnt in zip(runtime_cols, solver_viable_counts):
        pct = 100.0 * cnt / total_instances if total_instances > 0 else 0.0
        print(f"  {col:30s} : {cnt:6d} ({pct:6.2f}%)")

    # Distribution of number of viable solvers per instance
    num_viable_per_row = viable_mask.sum(axis=1)
    unique_k, counts_k = np.unique(num_viable_per_row, return_counts=True)

    print("\n#viable solvers per instance:")
    for k, cnt in zip(unique_k, counts_k):
        pct = 100.0 * cnt / total_instances if total_instances > 0 else 0.0
        print(f"  k = {int(k):2d} : {cnt:6d} ({pct:6.2f}%)")

    # Extra: status summary per solver (valid / invalid / other / missing)
    print("\nPer-solver status summary:")
    for j, rt_col in enumerate(runtime_cols):
        st_col = status_column_for_runtime(rt_col)
        if st_col not in df.columns:
            print(f"  {rt_col:30s} : (no status column '{st_col}')")
            continue

        svals = df[st_col]
        is_na = svals.isna()
        s_norm = svals.astype(str).str.strip().str.lower()

        is_valid = s_norm.isin(valid_statuses) & ~is_na
        is_invalid = s_norm.isin(invalid_statuses) & ~is_na
        is_other = ~is_na & ~is_valid & ~is_invalid

        n_total = len(df)
        n_valid = int(is_valid.sum())
        n_invalid = int(is_invalid.sum())
        n_other = int(is_other.sum())
        n_missing = int(is_na.sum())

        def pct(x: int) -> float:
            return 100.0 * x / n_total if n_total > 0 else 0.0

        print(
            f"  {rt_col:30s} : "
            f"valid={n_valid:4d} ({pct(n_valid):5.1f}%), "
            f"invalid={n_invalid:4d} ({pct(n_invalid):5.1f}%), "
            f"other={n_other:4d} ({pct(n_other):5.1f}%), "
            f"missing={n_missing:4d} ({pct(n_missing):5.1f}%)"
        )


def summarize_basic_info(df: pd.DataFrame, csv_path: str) -> None:
    """Print basic dataset information (rows, columns, missing values)."""
    print("============================================================")
    print("DATASET SUMMARY")
    print("============================================================")
    print(f"CSV: {os.path.abspath(csv_path)}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Quick overview of missing values (top 10 columns)
    missing = df.isna().sum()
    if missing.any():
        print("\nMissing values (top 10 columns by count):")
        missing_sorted = missing.sort_values(ascending=False)
        for col, cnt in missing_sorted.head(10).items():
            if cnt == 0:
                continue
            pct = 100.0 * cnt / len(df) if len(df) > 0 else 0.0
            print(f"  {col:30s} : {cnt:6d} ({pct:6.2f}%)")
    else:
        print("\nNo missing values detected.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect solver/label imbalance in SAT/JSSP CSV datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=60.0,
        help="Time limit in seconds used for multilabel-style viability analysis",
    )
    parser.add_argument(
        "--valid-statuses",
        type=str,
        default="ok,sat,unsat",
        help=(
            "Comma-separated list of status values considered as 'success'. "
            "Case-insensitive."
        ),
    )
    parser.add_argument(
        "--invalid-statuses",
        type=str,
        default="timeout,time_out,timedout,memout,crash,error,fail",
        help=(
            "Comma-separated list of status values considered as 'failure' "
            "(timeouts, crashes, etc.). Case-insensitive."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)

    summarize_basic_info(df, args.csv)

    solver_cols = detect_solver_cols(df)
    runtime_cols = solver_cols["runtime"]
    score_cols = solver_cols["score"]
    status_cols = solver_cols["status"]

    print("\nDetected solver columns:")
    print(f"  Runtime columns ({len(runtime_cols)}):")
    for c in runtime_cols:
        print(f"    - {c}")
    print(f"  Score columns ({len(score_cols)}):")
    for c in score_cols:
        print(f"    - {c}")
    print(f"  Status columns ({len(status_cols)}):")
    for c in status_cols:
        print(f"    - {c}")

    # Parse status sets
    valid_statuses, invalid_statuses = status_lists_from_args(
        args.valid_statuses, args.invalid_statuses
    )
    print("\nStatus interpretation:")
    print(f"  Valid statuses   : {valid_statuses}")
    print(f"  Invalid statuses : {invalid_statuses}")

    # Winner_Key distribution (if present)
    summarize_winner_key(df)

    # Classification-style imbalance (with invalids treated as ∞)
    summarize_best_solver_classification(
        df, runtime_cols, valid_statuses, invalid_statuses
    )

    # Multilabel-style viability (runtime + status)
    summarize_multilabel_viability(
        df, runtime_cols, args.time_limit, valid_statuses, invalid_statuses
    )


if __name__ == "__main__":
    main()
