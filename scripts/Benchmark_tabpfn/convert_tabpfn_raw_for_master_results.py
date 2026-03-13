#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_tabpfn_raw_for_master_results.py

Convert TabPFN raw outputs into a result structure compatible with
build_master_results_table.py.

Why this converter exists
-------------------------
The raw TabPFN pipeline saves:
- <dataset>__raw_predictions.csv
- <dataset>__raw_run.json
- <dataset>__raw_summary.csv
- preds/<dataset>/test/Raw_fold_0_preds.csv

The master-results builder expects dataset stems and artifacts shaped like:
- <dataset>__search_results.csv or .parquet
- <dataset>__best_config.json
- <dataset>__final_predictions.csv
- optionally all_predictions.parquet or preds/<dataset>/predictions.parquet

This script creates those expected artifacts while preserving the available
information from TabPFN raw. Since TabPFN raw has no CV search and no true
fold-wise external-test predictions, the converter fabricates a minimal,
transparent pseudo-search result based on the single raw run.

All comments are in English.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Basic helpers
# =============================================================================


def read_project_csv(path: Path) -> pd.DataFrame:
    """Read CSV using the project convention."""
    return pd.read_csv(path, sep=";", decimal=".")


def write_project_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV using the project convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=";", decimal=".", index=False)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def infer_task(raw_run_payload: Dict[str, Any], result_dir: Path) -> str:
    """
    Infer task type.

    Priority:
    1) raw_run.json content
    2) path heuristic aligned with build_master_results_table.py
    3) default to regression
    """
    task = raw_run_payload.get("task_type")
    if isinstance(task, str) and task.strip():
        return task.strip().lower()

    lower = str(result_dir).lower()
    if "classif" in lower:
        return "classification"
    return "regression"


def build_best_config_dict() -> Dict[str, str]:
    """Return the pseudo-preprocessing configuration for raw TabPFN."""
    return {
        "shape": "None",
        "scatter": "None",
        "pca": "None",
    }


def build_search_row(task_type: str, metric_value: Optional[float]) -> Dict[str, Any]:
    """
    Build one synthetic search-results row.

    Convention used here:
    - regression: mean_score = RMSE
    - classification: mean_score = -accuracy

    This keeps compatibility with the sorting logic of the master builder,
    which always sorts by ascending mean_score.
    """
    if metric_value is None:
        mean_score = np.nan
        fold_scores = []
    else:
        if task_type == "classification":
            mean_score = -float(metric_value)
            fold_scores = [-float(metric_value)]
        else:
            mean_score = float(metric_value)
            fold_scores = [float(metric_value)]

    return {
        "shape": "None",
        "scatter": "None",
        "pca": "None",
        "mean_score": mean_score,
        "fold_scores": json.dumps(fold_scores),
    }


def collect_datasets(input_dir: Path) -> List[str]:
    """Collect dataset names from TabPFN raw artifacts."""
    datasets = set()
    for path in input_dir.glob("*__raw_predictions.csv"):
        datasets.add(path.name.split("__raw_predictions.csv", 1)[0])
    for path in input_dir.glob("*__raw_run.json"):
        datasets.add(path.name.split("__raw_run.json", 1)[0])
    return sorted(datasets)


def load_raw_artifacts(input_dir: Path, dataset: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Load raw predictions and raw metadata for one dataset."""
    pred_path = input_dir / f"{dataset}__raw_predictions.csv"
    run_path = input_dir / f"{dataset}__raw_run.json"

    pred_df = read_project_csv(pred_path) if pred_path.exists() else None
    run_payload: Dict[str, Any] = {}
    if run_path.exists():
        with open(run_path, "r", encoding="utf-8") as f:
            run_payload = json.load(f)

    return pred_df, run_payload


def compute_metric_from_predictions(task_type: str, pred_df: Optional[pd.DataFrame]) -> Optional[float]:
    """Compute the proxy metric directly from raw predictions when possible."""
    if pred_df is None:
        return None
    if not {"y_true", "y_pred"}.issubset(pred_df.columns):
        return None

    clean = pred_df.dropna(subset=["y_true", "y_pred"]).copy()
    if clean.empty:
        return None

    if task_type == "classification":
        return accuracy(clean["y_true"].to_numpy(), clean["y_pred"].to_numpy())
    return rmse(clean["y_true"].to_numpy(), clean["y_pred"].to_numpy())


def build_best_config_payload(
    dataset: str,
    task_type: str,
    raw_run_payload: Dict[str, Any],
    metric_value: Optional[float],
) -> Dict[str, Any]:
    """Build a CatBoost-like best_config.json payload.
    """
    best_mean_score: Optional[float]
    best_fold_scores: List[float]

    if metric_value is None:
        best_mean_score = None
        best_fold_scores = []
    elif task_type == "classification":
        best_mean_score = -float(metric_value)
        best_fold_scores = [-float(metric_value)]
    else:
        best_mean_score = float(metric_value)
        best_fold_scores = [float(metric_value)]

    return {
        "dataset": dataset,
        "task_type": task_type,
        "seed": raw_run_payload.get("seed"),
        "n_splits": 1,
        "model": "TabPFN",
        "n_estimators_search": raw_run_payload.get("n_estimators"),
        "n_estimators_final": raw_run_payload.get("n_estimators"),
        "best_config": build_best_config_dict(),
        "best_mean_score": best_mean_score,
        "best_fold_scores": best_fold_scores,
        "best_model_params": {
            "n_estimators": raw_run_payload.get("n_estimators"),
            "device": raw_run_payload.get("tabpfn_device"),
            "model_path": raw_run_payload.get("model_path"),
            "source": "converted_from_tabpfn_raw",
        },
        "conversion_notes": (
            "Synthetic compatibility artifact created from TabPFN raw outputs. "
            "No CV search was performed; mean_score and fold scores come from the single raw test run when available."
        ),
    }


def build_all_predictions_rows(dataset: str, pred_df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Build rows for all_predictions.parquet compatible with build_master_results_table.py.

    We create a single pseudo-fold (fold=0) on split='test' with config='None'.
    """
    if pred_df is None:
        return []
    if not {"y_true", "y_pred"}.issubset(pred_df.columns):
        return []

    rows: List[Dict[str, Any]] = []
    clean = pred_df.dropna(subset=["y_true", "y_pred"]).copy()
    for _, row in clean.iterrows():
        rows.append(
            {
                "dataset": dataset,
                "split": "test",
                "fold": 0,
                "config": "None",
                "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]),
            }
        )
    return rows


def convert_one_dataset(
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    aggregate_all_predictions: List[Dict[str, Any]],
    overwrite: bool,
) -> Dict[str, Any]:
    """Convert one dataset."""
    pred_df, raw_run_payload = load_raw_artifacts(input_dir, dataset)
    task_type = infer_task(raw_run_payload, input_dir)
    metric_value = compute_metric_from_predictions(task_type, pred_df)

    dataset_outputs = {
        "dataset": dataset,
        "status": "ok",
        "details": None,
    }

    try:
        # 1) final_predictions.csv
        final_pred_path = output_dir / f"{dataset}__final_predictions.csv"
        if final_pred_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {final_pred_path}")
        if pred_df is None:
            raise FileNotFoundError(f"Missing raw predictions for dataset: {dataset}")
        final_df = pred_df.copy()
        write_project_csv(final_df, final_pred_path)

        # 2) search_results.csv
        search_results_path = output_dir / f"{dataset}__search_results.csv"
        if search_results_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {search_results_path}")
        search_df = pd.DataFrame([build_search_row(task_type=task_type, metric_value=metric_value)])
        write_project_csv(search_df, search_results_path)

        # 3) best_config.json
        best_config_path = output_dir / f"{dataset}__best_config.json"
        if best_config_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {best_config_path}")
        best_payload = build_best_config_payload(
            dataset=dataset,
            task_type=task_type,
            raw_run_payload=raw_run_payload,
            metric_value=metric_value,
        )
        best_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(best_config_path, "w", encoding="utf-8") as f:
            json.dump(best_payload, f, indent=2, ensure_ascii=False)

        # 4) Optional copy of legacy raw metadata for traceability
        raw_json_src = input_dir / f"{dataset}__raw_run.json"
        if raw_json_src.exists():
            raw_json_dst = output_dir / f"{dataset}__raw_run.json"
            shutil.copy2(raw_json_src, raw_json_dst)

        raw_summary_src = input_dir / f"{dataset}__raw_summary.csv"
        if raw_summary_src.exists():
            raw_summary_dst = output_dir / f"{dataset}__raw_summary.csv"
            shutil.copy2(raw_summary_src, raw_summary_dst)

        # 5) Build all_predictions rows for RMSE_MF compatibility
        aggregate_all_predictions.extend(build_all_predictions_rows(dataset=dataset, pred_df=pred_df))

    except Exception as exc:
        dataset_outputs["status"] = "error"
        dataset_outputs["details"] = repr(exc)

    return dataset_outputs


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TabPFN raw outputs into artifacts compatible with build_master_results_table.py."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing TabPFN raw results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where converted artifacts will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted artifacts.",
    )
    parser.add_argument(
        "--write_all_predictions_parquet",
        action="store_true",
        help="Write a global all_predictions.parquet compatible with the master-results builder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    datasets = collect_datasets(input_dir)
    if not datasets:
        raise RuntimeError(
            "No TabPFN raw artifacts were found. Expected files like '<dataset>__raw_predictions.csv' or '<dataset>__raw_run.json'."
        )

    aggregate_all_predictions: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        row = convert_one_dataset(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            aggregate_all_predictions=aggregate_all_predictions,
            overwrite=bool(args.overwrite),
        )
        report_rows.append(row)

    if args.write_all_predictions_parquet:
        parquet_path = output_dir / "all_predictions.parquet"
        if parquet_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {parquet_path}")
        pd.DataFrame(aggregate_all_predictions).to_parquet(parquet_path, index=False)

    report_path = output_dir / "conversion_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    ok_count = sum(1 for row in report_rows if row["status"] == "ok")
    err_count = len(report_rows) - ok_count

    print(f"Converted datasets: {ok_count}")
    print(f"Errors: {err_count}")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")
    if args.write_all_predictions_parquet:
        print(f"all_predictions.parquet written to: {output_dir / 'all_predictions.parquet'}")


if __name__ == "__main__":
    main()
