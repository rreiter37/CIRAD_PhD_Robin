#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect the content of .meta.parquet and .arrays.parquet files
produced per dataset.

The script:
  - Loads both parquet files
  - Displays shape, columns, dtypes
  - Prints a short preview of the content
  - Helps identify available metrics, splits, folds, predictions, etc.
"""

import argparse
from pathlib import Path
import pandas as pd


# ===================== CLI ===================== #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing {dataset}.meta.parquet and {dataset}.arrays.parquet"
    )

    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of rows to display for preview"
    )

    return parser.parse_args()


# ===================== Utilities ===================== #

def inspect_parquet(path: Path, head: int):
    """
    Load a parquet file and print a structured summary.
    """
    print("=" * 100)
    print(f"File: {path.name}")
    print("=" * 100)

    df = pd.read_parquet(path)

    print("\n[Shape]")
    print(df.shape)

    print("\n[Columns]")
    for col in df.columns:
        print(f"  - {col}")

    print("\n[Data types]")
    print(df.dtypes)

    print(f"\n[Head: first {head} rows]")
    print(df.head(head))

    print("\n[Describe (numeric columns)]")
    try:
        print(df.describe())
    except Exception:
        print("No numeric columns available for describe().")

    print("\n[Unique values per column (up to 10)]")
    for col in df.columns:
        uniques = df[col].unique()
        print(f"  - {col}: {uniques[:10]}{' ...' if len(uniques) > 10 else ''}")

    return df


# ===================== Main ===================== #

def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    meta_files = list(dataset_dir.glob("*.meta.parquet"))
    array_files = list(dataset_dir.glob("*.arrays.parquet"))

    if not meta_files and not array_files:
        raise RuntimeError("No .meta.parquet or .arrays.parquet files found.")

    # ---------- META FILES ----------
    for meta_path in meta_files:
        inspect_parquet(meta_path, args.head)

    # ---------- ARRAY FILES ----------
    for array_path in array_files:
        inspect_parquet(array_path, args.head)

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    main()
