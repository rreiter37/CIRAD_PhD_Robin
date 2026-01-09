#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize CSV separators for NIRS datasets.

This script:
- Iterates over all datasets in Data/Regression
- Checks Xcal.csv and Xval.csv
- Detects whether the separator is ',' or ';'
- Converts to ';' ONLY if needed
- Leaves already-correct files untouched

Safe, idempotent, and benchmark-ready.
"""

from pathlib import Path
import pandas as pd


BASE_DIR = Path("Data/Regression")
X_FILES = ["Xcal.csv", "Xval.csv"]


def detect_separator(csv_path, n_lines=5):
    """
    Detect the separator by inspecting the first non-empty line.
    Returns ',' or ';' or None if unclear.
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(n_lines):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if ";" in line and "," not in line:
                return ";"
            if "," in line and ";" not in line:
                return ","
            if ";" in line and "," in line:
                # Ambiguous but usually means ',' is decimal-free
                return ","
    return None


def normalize_csv(csv_path):
    """
    Ensure the CSV uses ';' as separator.
    """
    sep = detect_separator(csv_path)

    if sep is None:
        print(f"[?] Could not detect separator → skipping {csv_path}")
        return

    if sep == ";":
        print(f"[OK] {csv_path} already uses ';'")
        return

    print(f"[FIX] Converting separator ',' → ';' in {csv_path}")

    df = pd.read_csv(csv_path, sep=sep)
    df.to_csv(csv_path, sep=";", index=False)


def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

    datasets = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])

    print(f"Scanning {len(datasets)} datasets in {BASE_DIR}\n")

    for dataset_dir in datasets:
        print(f"=== Dataset: {dataset_dir.name} ===")

        for fname in X_FILES:
            csv_path = dataset_dir / fname
            if csv_path.exists():
                normalize_csv(csv_path)
            else:
                print(f"[--] {fname} not found")

        print()

    print("Separator normalization completed.")


if __name__ == "__main__":
    main()
