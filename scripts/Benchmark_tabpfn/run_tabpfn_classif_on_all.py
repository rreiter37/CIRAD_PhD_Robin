#!/usr/bin/env python3
"""
Run TabPFN raw classification on ALL datasets found under:

Data/classification/{dataset_type}/{dataset_name}/
  - Xtrain.csv, Ytrain.csv, Xtest.csv, Ytest.csv

Outputs:
- Results/tabpfn_classif_raw/
  - predictions__*.csv
  - metrics__*.json
  - report__*.txt
  - summary_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd

from pipeline_tabpfn_classif import (
    DatasetId,
    run_tabpfn_raw_classification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="Data/classification",
        help="Root directory for classification datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Results/tabpfn_classif_raw",
        help="Where to save predictions/metrics/reports and summary CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for TabPFN wrapper (deterministic runs).",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=16,
        help="Number of TabPFN estimators (ensemble size).",
    )
    parser.add_argument(
        "--only_type",
        type=str,
        default=None,
        help="If set, only run datasets under this dataset_type.",
    )
    parser.add_argument(
        "--only_dataset",
        type=str,
        default=None,
        help="If set, only run a specific dataset_name (within only_type if provided).",
    )
    return parser.parse_args()


def discover_datasets(data_root: Path) -> List[Dict[str, Path]]:
    """
    Discover datasets under Data/classification/{type}/{name}/ with required CSV files.
    """
    required = {"Xtrain.csv", "Ytrain.csv", "Xtest.csv", "Ytest.csv"}
    out = []

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    for dataset_type_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for dataset_dir in sorted([p for p in dataset_type_dir.iterdir() if p.is_dir()]):
            files = {p.name for p in dataset_dir.iterdir() if p.is_file()}
            if required.issubset(files):
                out.append(
                    {
                        "dataset_type": dataset_type_dir.name,
                        "dataset_name": dataset_dir.name,
                        "path": dataset_dir,
                    }
                )

    return out


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ds = discover_datasets(data_root)

    # Optional filters
    if args.only_type is not None:
        all_ds = [d for d in all_ds if d["dataset_type"] == args.only_type]
    if args.only_dataset is not None:
        all_ds = [d for d in all_ds if d["dataset_name"] == args.only_dataset]

    if not all_ds:
        raise RuntimeError(
            "No datasets found. Expected structure:\n"
            "Data/classification/{dataset_type}/{dataset_name}/"
            "{Xtrain.csv,Ytrain.csv,Xtest.csv,Ytest.csv}"
        )

    summary_rows = []

    print(f"Found {len(all_ds)} classification datasets.")
    print(f"Saving outputs to: {output_dir}")

    for i, d in enumerate(all_ds, start=1):
        ds_id = DatasetId(dataset_type=d["dataset_type"], dataset_name=d["dataset_name"])
        ds_path = d["path"]

        print("\n" + "=" * 100)
        print(f"[{i}/{len(all_ds)}] Dataset: {ds_id.tag}")
        print(f"Path: {ds_path}")
        print("=" * 100)

        payload = run_tabpfn_raw_classification(
            dataset_id=ds_id,
            dataset_dir=ds_path,
            output_dir=output_dir,
            seed=args.seed,
            n_estimators=args.n_estimators,
            ignore_pretraining_limits=True,
        )

        row = {
            "dataset_type": payload["dataset_type"],
            "dataset_name": payload["dataset_name"],
            "n_train": payload["n_train"],
            "n_test": payload["n_test"],
            "n_features": payload["n_features"],
            "n_classes": payload["n_classes"],
            "seed": payload["seed"],
            "n_estimators": payload["n_estimators"],
            "device": payload["device"],
        }
        # Flatten metrics
        for k, v in payload["metrics"].items():
            row[k] = v

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # Sort by balanced accuracy, then accuracy
    sort_cols = [c for c in ["balanced_accuracy", "accuracy"] if c in df_summary.columns]
    if sort_cols:
        df_summary = df_summary.sort_values(sort_cols, ascending=False)

    summary_path = output_dir / "summary_metrics.csv"
    df_summary.to_csv(summary_path, index=False)

    print("\n✅ Done.")
    print(f"Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
