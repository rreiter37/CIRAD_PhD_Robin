#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_pls_ridge_legacy_to_parquet.py

Convert legacy PLS/Ridge result folders to the newer storage layout used by the
updated pipeline, with particular emphasis on dataset-level Parquet files.

Expected legacy layout (typical case):
    <results_root>/pls/
        <dataset>__search_results.csv
        <dataset>__best_config.json
        <dataset>__final_predictions.csv
        <dataset>__cv_predictions.csv
    <results_root>/ridge/
        ...

New files created in-place:
    <results_root>/<model>/preds/<dataset>/predictions.parquet
    <results_root>/<model>/<dataset>__search_results.parquet

This script does not delete the legacy CSV/JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_csv_project(path: Path) -> pd.DataFrame:
    """Read a project CSV file with ';' separator and '.' decimal."""
    return pd.read_csv(path, sep=";", decimal=".")


def maybe_parse_json_string(value):
    """Parse a JSON-looking string when possible; otherwise return the raw value."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def convert_search_results_csv_to_parquet(model_dir: Path) -> List[Path]:
    """Convert all legacy search_results CSV files found in one model directory."""
    written: List[Path] = []
    for csv_path in sorted(model_dir.glob("*__search_results.csv")):
        df = read_csv_project(csv_path)
        out_path = csv_path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        written.append(out_path)
    return written


def legacy_cv_csv_to_predictions_parquet(model_dir: Path, dataset_name: str) -> Optional[Path]:
    """Convert one legacy <dataset>__cv_predictions.csv file into predictions.parquet."""
    cv_csv = model_dir / f"{dataset_name}__cv_predictions.csv"
    if not cv_csv.exists():
        return None

    df = read_csv_project(cv_csv)
    if df.empty:
        return None

    rename_map = {}
    if "config" in df.columns:
        rename_map["config"] = "preproc_tag"
    df = df.rename(columns=rename_map)

    if "preproc_tag" not in df.columns:
        df["preproc_tag"] = "None"
    if "model" not in df.columns:
        df["model"] = model_dir.name
    if "best_params" not in df.columns:
        df["best_params"] = "{}"
    if "dataset" not in df.columns:
        df["dataset"] = dataset_name
    if "split" not in df.columns:
        df["split"] = "valid"
    if "fold" not in df.columns:
        df["fold"] = -1
    if "y_true" not in df.columns:
        df["y_true"] = np.nan
    if "y_pred" not in df.columns:
        raise ValueError(f"Missing y_pred in {cv_csv}")

    # Rebuild a deterministic row_id within each logical prediction vector.
    group_cols = ["dataset", "split", "fold", "preproc_tag", "model", "best_params"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = "NA"
    df["row_id"] = df.groupby(group_cols).cumcount()

    out_dir = model_dir / "preds" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.parquet"

    out_cols = ["dataset", "split", "fold", "preproc_tag", "model", "best_params", "row_id", "y_true", "y_pred"]
    table = pa.Table.from_pandas(df[out_cols])
    pq.write_table(table, out_path, compression="zstd")
    return out_path


def detect_datasets_from_model_dir(model_dir: Path) -> List[str]:
    """Infer dataset names from legacy file prefixes in one model directory."""
    names = set()
    for path in model_dir.glob("*__search_results.csv"):
        names.add(path.name.replace("__search_results.csv", ""))
    for path in model_dir.glob("*__cv_predictions.csv"):
        names.add(path.name.replace("__cv_predictions.csv", ""))
    for path in model_dir.glob("*__final_predictions.csv"):
        names.add(path.name.replace("__final_predictions.csv", ""))
    return sorted(names)


def process_model_dir(model_dir: Path) -> Dict[str, List[str]]:
    """Convert every supported legacy artifact in one model directory."""
    summary = {
        "search_results_parquet": [],
        "predictions_parquet": [],
    }

    for path in convert_search_results_csv_to_parquet(model_dir):
        summary["search_results_parquet"].append(str(path))

    for dataset_name in detect_datasets_from_model_dir(model_dir):
        out_path = legacy_cv_csv_to_predictions_parquet(model_dir, dataset_name)
        if out_path is not None:
            summary["predictions_parquet"].append(str(out_path))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Root directory containing legacy 'pls' and/or 'ridge' result subfolders.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["pls", "ridge"],
        choices=["pls", "ridge"],
        help="Model subdirectories to convert.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(str(results_root))

    global_summary: Dict[str, Dict[str, List[str]]] = {}

    for model in args.models:
        model_dir = results_root / model
        if not model_dir.exists():
            print(f"[WARN] Missing model directory, skipped: {model_dir}")
            continue

        summary = process_model_dir(model_dir)
        global_summary[model] = summary

        print(f"\nModel: {model}")
        print(f"  search_results.parquet created: {len(summary['search_results_parquet'])}")
        print(f"  predictions.parquet created:    {len(summary['predictions_parquet'])}")

    summary_path = results_root / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    print(f"\nSaved conversion summary to: {summary_path}")


if __name__ == "__main__":
    main()
