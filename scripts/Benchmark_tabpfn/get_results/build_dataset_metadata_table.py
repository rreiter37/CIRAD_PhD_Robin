#!/usr/bin/env python3
"""Build a metadata table for all datasets in the project.

This script reads the dataset description Excel file and inspects the on-disk
train/test CSV files to build a single metadata table that can be used by the
secondary analyses in the Results section.

Expected project layout
-----------------------
Data/
├── DatabaseDetail.xlsx
├── regression/
│   └── <database_name>/
│       └── <dataset_name>/
│           ├── Xtrain.csv
│           ├── Ytrain.csv
│           ├── Xtest.csv
│           └── Ytest.csv
└── classification/
    └── <database_name>/
        └── <dataset_name>/
            ├── Xtrain.csv
            ├── Ytrain.csv
            ├── Xtest.csv
            └── Ytest.csv

The Excel file is expected to contain, at minimum, the following columns:
- Database
- Type
- Dataset
- Sample
- Trait
- Split

The CSV convention follows the one already used in the project pipelines:
separator=';' and decimal='.'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def read_csv_strict(path: Path) -> pd.DataFrame:
    """Read a CSV file using the project convention.

    The existing training pipelines use ';' as separator and '.' as decimal mark.
    This helper reproduces the same convention so the metadata script stays fully
    aligned with the rest of the project.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, sep=";", decimal=".")


def load_y_series(path: Path) -> pd.Series:
    """Load a one-column target CSV as a pandas Series."""
    df = read_csv_strict(path)
    return df.iloc[:, 0]


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------

def normalize_excel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Excel column names for robust downstream access."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def task_to_folder(task_value: Optional[str]) -> str:
    """Convert the task label from the Excel sheet into a folder name."""
    task_str = str(task_value).strip().lower() if pd.notna(task_value) else ""
    if task_str.startswith("class"):
        return "classification"
    return "regression"


def safe_str(value: object) -> Optional[str]:
    """Convert a value to a clean string, returning None for missing values."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


# -----------------------------------------------------------------------------
# Classification helpers
# -----------------------------------------------------------------------------

def compute_classification_stats(y_train: pd.Series, y_test: Optional[pd.Series]) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """Compute the number of classes and a simple imbalance indicator.

    The imbalance degree is defined as the share of the majority class in the
    combined train+test target vector. This gives a simple, interpretable metric:
    - close to 1 / n_classes -> balanced dataset
    - close to 1.0 -> highly imbalanced dataset

    A human-readable imbalance label is also returned.
    """
    y_parts = [y_train]
    if y_test is not None:
        y_parts.append(y_test)

    y_all = pd.concat(y_parts, axis=0, ignore_index=True)
    y_all = y_all.dropna()
    if y_all.empty:
        return None, None, None

    counts = y_all.value_counts(dropna=False)
    n_classes = int(counts.shape[0])
    majority_ratio = float(counts.iloc[0] / counts.sum())

    if n_classes <= 1:
        imbalance_label = "degenerate"
    elif majority_ratio >= 0.90:
        imbalance_label = "severe"
    elif majority_ratio >= 0.75:
        imbalance_label = "high"
    elif majority_ratio >= 0.60:
        imbalance_label = "moderate"
    else:
        imbalance_label = "low"

    return n_classes, majority_ratio, imbalance_label


# -----------------------------------------------------------------------------
# Row builder
# -----------------------------------------------------------------------------

def build_row_from_excel_record(record: pd.Series, data_root: Path) -> Dict[str, object]:
    """Build one metadata row from one Excel record.

    The Excel file provides the semantic metadata (database name, trait, split,
    sample type, task), while the CSV files provide the structural statistics
    (number of samples, number of variables, class balance).
    """
    database_name = safe_str(record.get("Database"))
    dataset_name = safe_str(record.get("Dataset"))
    task_excel = safe_str(record.get("Type"))
    sample_type = safe_str(record.get("Sample"))
    trait = safe_str(record.get("Trait"))
    split_type = safe_str(record.get("Split"))

    task = task_to_folder(task_excel)

    row: Dict[str, object] = {
        "database_name": database_name,
        "dataset_name": dataset_name,
        "task": task,
        "sample_type": sample_type,
        "trait": trait,
        "split_type": split_type,
        "n_samples_total": np.nan,
        "n_samples_train": np.nan,
        "n_samples_test": np.nan,
        "n_features": np.nan,
        "p_over_n": np.nan,
        "n_classes": np.nan,
        "class_imbalance_ratio": np.nan,
        "class_imbalance_label": None,
        "dataset_dir": None,
        "status": "ok",
        "missing_files_json": "[]",
    }

    if database_name is None or dataset_name is None:
        row["status"] = "missing_excel_identifiers"
        return row

    dataset_dir = data_root / task / database_name / dataset_name
    row["dataset_dir"] = str(dataset_dir)

    xtrain_path = dataset_dir / "Xtrain.csv"
    xtest_path = dataset_dir / "Xtest.csv"
    ytrain_path = dataset_dir / "Ytrain.csv"
    ytest_path = dataset_dir / "Ytest.csv"

    expected_files = [xtrain_path, xtest_path, ytrain_path, ytest_path]
    missing_files = [str(p) for p in expected_files if not p.exists()]
    row["missing_files_json"] = json.dumps(missing_files, ensure_ascii=False)

    if missing_files:
        row["status"] = "missing_files"
        return row

    try:
        x_train = read_csv_strict(xtrain_path)
        x_test = read_csv_strict(xtest_path)
        y_train = load_y_series(ytrain_path)
        y_test = load_y_series(ytest_path)
    except Exception as exc:  # pragma: no cover - defensive branch
        row["status"] = f"read_error: {type(exc).__name__}"
        return row

    n_train = int(x_train.shape[0])
    n_test = int(x_test.shape[0])
    n_total = int(n_train + n_test)
    n_features = int(x_train.shape[1])
    p_over_n = float(n_features / n_total) if n_total > 0 else np.nan

    row.update(
        {
            "n_samples_total": n_total,
            "n_samples_train": n_train,
            "n_samples_test": n_test,
            "n_features": n_features,
            "p_over_n": p_over_n,
        }
    )

    if task == "classification":
        n_classes, imbalance_ratio, imbalance_label = compute_classification_stats(y_train, y_test)
        row["n_classes"] = n_classes if n_classes is not None else np.nan
        row["class_imbalance_ratio"] = imbalance_ratio if imbalance_ratio is not None else np.nan
        row["class_imbalance_label"] = imbalance_label

    return row


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------

def build_dataset_metadata_table(detail_xlsx: Path, data_root: Path) -> pd.DataFrame:
    """Build the full dataset metadata table."""
    detail_df = pd.read_excel(detail_xlsx)
    detail_df = normalize_excel_columns(detail_df)

    required_columns = ["Database", "Type", "Dataset", "Sample", "Trait", "Split"]
    missing_columns = [col for col in required_columns if col not in detail_df.columns]
    if missing_columns:
        raise ValueError(
            "The Excel file is missing required columns: " + ", ".join(missing_columns)
        )

    rows: List[Dict[str, object]] = []
    for _, record in detail_df.iterrows():
        rows.append(build_row_from_excel_record(record, data_root=data_root))

    metadata_df = pd.DataFrame(rows)

    # Drop exact duplicates if the Excel file contains repeated entries.
    metadata_df = metadata_df.drop_duplicates(
        subset=["database_name", "dataset_name", "task"], keep="first"
    ).reset_index(drop=True)

    # Stable sorting improves readability and reproducibility.
    metadata_df = metadata_df.sort_values(
        by=["task", "database_name", "dataset_name"], kind="stable"
    ).reset_index(drop=True)

    return metadata_df


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a metadata table for all datasets from the Excel catalog and on-disk CSV files."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("Data"),
        help="Root directory containing the regression/ and classification/ folders.",
    )
    parser.add_argument(
        "--detail_xlsx",
        type=Path,
        default=Path("Data") / "DatabaseDetail.xlsx",
        help="Path to the Excel file describing the datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where dataset_metadata.parquet and dataset_metadata.csv will be written.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    """Run the full metadata extraction pipeline."""
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = build_dataset_metadata_table(
        detail_xlsx=args.detail_xlsx,
        data_root=args.data_root,
    )

    parquet_path = args.output_dir / "dataset_metadata.parquet"
    csv_path = args.output_dir / "dataset_metadata.csv"
    errors_path = args.output_dir / "dataset_metadata_errors.csv"

    metadata_df.to_parquet(parquet_path, index=False)
    metadata_df.to_csv(csv_path, index=False)

    error_df = metadata_df[metadata_df["status"] != "ok"].copy()
    error_df.to_csv(errors_path, index=False)

    print(f"[OK] Metadata table written to: {parquet_path}")
    print(f"[OK] CSV export written to:    {csv_path}")
    print(f"[OK] Error rows written to:    {errors_path}")
    print(f"[INFO] Total datasets:         {len(metadata_df)}")
    print(f"[INFO] Status != ok rows:      {len(error_df)}")


if __name__ == "__main__":
    main()
