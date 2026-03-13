#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_preds_csv_to_parquet.py

Convert many prediction CSV files saved by the TabPFN pipeline into a single Parquet per dataset.

Expected input layout:
    <results_dir>/preds/<dataset_name>/{valid,test}/*.csv

Expected CSV filename pattern (from your pipeline):
    <preproc_tag>_fold_<k>_preds.csv

Expected CSV content:
- Usually columns: y_true; y_pred (separator=';', decimal='.')
- Sometimes only: y_pred (then y_true will be NaN)

Output:
    <results_dir>/preds/<dataset_name>/predictions.parquet

Parquet schema (per row):
- dataset (string)
- split (string)        # "valid" or "test"
- fold (int32)
- preproc_tag (string)
- row_id (int64)
- y_true (float64)
- y_pred (float64)

Notes:
- Uses pyarrow ParquetWriter to stream-write chunks (memory friendly).
- Compression: zstd
- Optional: delete original CSV after successful conversion.

Install dependency:
    pip install pyarrow
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# Helpers
# -----------------------------

def require_pyarrow() -> None:
    """Fail fast with a clear message if pyarrow is missing."""
    if pa is None or pq is None:
        raise ImportError("pyarrow is required. Install with: pip install pyarrow")


def iter_dataset_dirs(preds_root: Path) -> List[Path]:
    """Return all dataset directories under preds/."""
    if not preds_root.exists():
        return []
    return sorted([p for p in preds_root.iterdir() if p.is_dir()])


def iter_csv_files(dataset_dir: Path) -> Iterable[Path]:
    """Yield all CSV files under valid/ and test/ subfolders."""
    for split in ("valid", "test"):
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for fp in sorted(split_dir.glob("*.csv")):
            yield fp


def infer_split_from_path(fp: Path) -> str:
    """Infer split name based on parent folder."""
    parent = fp.parent.name.lower()
    if parent.startswith("v"):
        return "valid"
    if parent.startswith("t"):
        return "test"
    return parent


def infer_tag_and_fold_from_filename(fp: Path) -> Tuple[str, int]:
    """
    Infer preproc_tag and fold id from filename.

    Expected:
        <preproc_tag>_fold_<k>_preds.csv

    If parsing fails:
        fold = -1
        tag = filename stem
    """
    stem = fp.stem  # without .csv
    fold = -1

    m = re.search(r"_fold_(\d+)_preds$", stem)
    if m:
        fold = int(m.group(1))
        tag = stem[: m.start()]  # everything before "_fold_<k>_preds"
        tag = tag if tag else "None"
        return tag, fold

    # Fallback: any "fold<number>" pattern
    m2 = re.search(r"fold[_-]?(\d+)", stem)
    if m2:
        fold = int(m2.group(1))

    return stem, fold


def read_preds_csv(fp: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a single predictions CSV.
    Returns (y_true, y_pred) as float64 arrays. y_true may be NaN-filled if absent.
    """
    df = pd.read_csv(fp, sep=";", decimal=".")
    cols = [c.strip().lower() for c in df.columns]

    if "y_pred" in cols:
        y_pred = df[df.columns[cols.index("y_pred")]].to_numpy(dtype=np.float64).reshape(-1)
    else:
        # Fallback: first column is y_pred
        y_pred = df.iloc[:, 0].to_numpy(dtype=np.float64).reshape(-1)

    if "y_true" in cols:
        y_true = df[df.columns[cols.index("y_true")]].to_numpy(dtype=np.float64).reshape(-1)
    else:
        y_true = np.full(len(y_pred), np.nan, dtype=np.float64)

    return y_true, y_pred


def rows_to_table(
    dataset: str,
    split: str,
    fold: int,
    tag: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> "pa.Table":
    """Build a pyarrow Table with the standard schema."""
    require_pyarrow()

    n = len(y_pred)
    if len(y_true) != n:
        raise ValueError("y_true and y_pred must have the same length")

    df = pd.DataFrame(
        {
            "dataset": dataset,
            "split": split,
            "fold": np.int32(fold),
            "preproc_tag": tag,
            "row_id": np.arange(n, dtype=np.int64),
            "y_true": y_true.astype(np.float64),
            "y_pred": y_pred.astype(np.float64),
        }
    )
    return pa.Table.from_pandas(df, preserve_index=False)


# -----------------------------
# Main conversion
# -----------------------------

def convert_one_dataset(
    dataset_dir: Path,
    delete_csv: bool,
    verbose: bool,
) -> Optional[Path]:
    """
    Convert all CSV under <dataset_dir>/{valid,test} into <dataset_dir>/predictions.parquet.
    Returns the parquet path, or None if no CSV found.
    """
    require_pyarrow()

    dataset_name = dataset_dir.name
    parquet_path = dataset_dir / "predictions.parquet"

    csv_files = list(iter_csv_files(dataset_dir))
    if len(csv_files) == 0:
        return None

    # Define a stable schema for streaming writes
    schema = pa.schema(
        [
            ("dataset", pa.string()),
            ("split", pa.string()),
            ("fold", pa.int32()),
            ("preproc_tag", pa.string()),
            ("row_id", pa.int64()),
            ("y_true", pa.float64()),
            ("y_pred", pa.float64()),
        ]
    )

    # Stream-write to Parquet
    writer = pq.ParquetWriter(parquet_path, schema=schema, compression="zstd")

    try:
        for fp in csv_files:
            split = infer_split_from_path(fp)
            tag, fold = infer_tag_and_fold_from_filename(fp)

            y_true, y_pred = read_preds_csv(fp)
            table = rows_to_table(dataset_name, split, fold, tag, y_true, y_pred)

            # Ensure table matches schema exactly
            table = table.cast(schema)
            writer.write_table(table)

            if verbose:
                print(f"[ADD] {dataset_name} | {split} | fold={fold} | tag={tag} | n={len(y_pred)} | {fp.name}")

        writer.close()

    except Exception:
        # If something fails mid-way, close writer and keep partial file for debugging
        try:
            writer.close()
        except Exception:
            pass
        raise

    # Delete CSV files if requested (only after successful write)
    if delete_csv:
        for fp in csv_files:
            try:
                fp.unlink()
            except Exception:
                pass

        # Optionally remove empty valid/test folders
        for split in ("valid", "test"):
            d = dataset_dir / split
            if d.exists():
                try:
                    if not any(d.iterdir()):
                        d.rmdir()
                except Exception:
                    pass

    return parquet_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to results directory containing preds/. Example: Results/tabpfn_reg_raw",
    )
    parser.add_argument(
        "--delete_csv",
        action="store_true",
        help="Delete original CSV prediction files after successful conversion.",
    )
    parser.add_argument(
        "--only_dataset",
        type=str,
        default="",
        help="If set, only convert this dataset folder name under preds/.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each processed CSV.",
    )
    args = parser.parse_args()

    require_pyarrow()

    results_dir = Path(args.results_dir)
    preds_root = results_dir / "preds"
    if not preds_root.exists():
        raise FileNotFoundError(f"preds/ not found under: {results_dir}")

    dataset_dirs = iter_dataset_dirs(preds_root)
    if args.only_dataset.strip():
        target = args.only_dataset.strip()
        dataset_dirs = [d for d in dataset_dirs if d.name == target]

    if len(dataset_dirs) == 0:
        print(f"[WARN] No dataset directories to convert under: {preds_root}")
        return

    print(f"[INFO] Converting {len(dataset_dirs)} dataset(s) under: {preds_root}")
    print(f"[INFO] delete_csv={bool(args.delete_csv)}")

    n_ok = 0
    n_skip = 0
    for d in dataset_dirs:
        try:
            out = convert_one_dataset(d, delete_csv=bool(args.delete_csv), verbose=bool(args.verbose))
            if out is None:
                n_skip += 1
                print(f"[SKIP] {d.name}: no CSV files found")
            else:
                n_ok += 1
                print(f"[OK]   {d.name}: wrote {out}")
        except Exception as e:
            print(f"[ERR]  {d.name}: {repr(e)}")

    print(f"\nDONE. Converted={n_ok} Skipped={n_skip} Total={len(dataset_dirs)}")


if __name__ == "__main__":
    main()