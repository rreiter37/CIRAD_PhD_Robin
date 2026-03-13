#!/usr/bin/env python3
"""
Compute dataset-wise model ranks from a master results table.

This script is intended to work with the output of build_master_results_table.py.
It computes, for each dataset, the rank of each model based on the main test
metric:
    - regression    -> RMSEP (lower is better)
    - classification -> ACCP / accuracy-like metric (higher is better)

It also exports mean ranks aggregated by task.

Expected inputs
---------------
A master results table in parquet or CSV format with at least:
    - dataset
    - task
    - model
and one of the following metric columns depending on the task:
    - regression: RMSEP
    - classification: ACCP / accuracy / acc_test / test_accuracy

Expected outputs
----------------
    - datasetwise_ranks.parquet
    - datasetwise_ranks.csv
    - mean_ranks.parquet
    - mean_ranks.csv
    - datasetwise_ranks_errors.csv

Notes
-----
- Ties are handled with pandas rank(method="average").
- Rows with missing metrics are excluded from ranking, but kept in the error log.
- The script is flexible regarding model count and missing model/dataset pairs.

Comments are intentionally written in English.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REGRESSION_METRIC_CANDIDATES = ["RMSEP", "rmsep", "test_rmsep"]
CLASSIFICATION_METRIC_CANDIDATES = [
    "ACCP",
    "accp",
    "accuracy",
    "test_accuracy",
    "acc_test",
    "balanced_accuracy",
]

REQUIRED_ID_COLUMNS = ["dataset", "task", "model"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dataset-wise model ranks from a master results table."
    )
    parser.add_argument(
        "--master_results",
        type=str,
        required=True,
        help="Path to master_results.parquet or master_results.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the rank tables will be written.",
    )
    parser.add_argument(
        "--rank_method",
        type=str,
        default="average",
        choices=["average", "min", "max", "dense", "first"],
        help="Pandas ranking method used to handle ties.",
    )
    parser.add_argument(
        "--keep_status_filter",
        type=str,
        default="ok,partial",
        help=(
            "Comma-separated list of statuses to keep from master_results. "
            "If the status column does not exist, all rows are kept."
        ),
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported input format. Use parquet or csv.")


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_task(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if "class" in text:
        return "classification"
    if "reg" in text:
        return "regression"
    return text


def select_metric_column(df_task: pd.DataFrame, task: str) -> Optional[str]:
    candidates = (
        REGRESSION_METRIC_CANDIDATES
        if task == "regression"
        else CLASSIFICATION_METRIC_CANDIDATES
    )
    for col in candidates:
        if col in df_task.columns:
            return col
    return None


def metric_orientation(task: str) -> str:
    if task == "regression":
        return "lower_is_better"
    if task == "classification":
        return "higher_is_better"
    return "unknown"


def compute_ranks_for_task(
    df_task: pd.DataFrame,
    task: str,
    rank_method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute ranks for a single task.

    Returns
    -------
    ranks_df:
        One row per dataset-model with metric value and rank.
    errors_df:
        Rows that could not be ranked.
    """
    metric_col = select_metric_column(df_task, task)
    if metric_col is None:
        errors = df_task.copy()
        errors["status"] = "missing_metric_column"
        errors["rank_error_info"] = json.dumps(
            {
                "task": task,
                "expected_metric_candidates": (
                    REGRESSION_METRIC_CANDIDATES
                    if task == "regression"
                    else CLASSIFICATION_METRIC_CANDIDATES
                ),
            }
        )
        return pd.DataFrame(), errors

    work = df_task.copy()
    work["metric_used_for_ranking"] = metric_col
    work["metric_value_for_ranking"] = pd.to_numeric(work[metric_col], errors="coerce")
    work["ranking_orientation"] = metric_orientation(task)

    errors_mask = work["metric_value_for_ranking"].isna()
    errors = work.loc[errors_mask].copy()
    if not errors.empty:
        errors["status"] = "missing_metric_value"
        errors["rank_error_info"] = errors.apply(
            lambda row: json.dumps(
                {
                    "task": task,
                    "metric_column": metric_col,
                    "dataset": row.get("dataset", None),
                    "model": row.get("model", None),
                }
            ),
            axis=1,
        )

    valid = work.loc[~errors_mask].copy()
    if valid.empty:
        return pd.DataFrame(), errors

    ascending = True if task == "regression" else False

    valid["rank"] = (
        valid.groupby("dataset")["metric_value_for_ranking"]
        .rank(method=rank_method, ascending=ascending)
        .astype(float)
    )

    valid["n_models_ranked_in_dataset"] = valid.groupby("dataset")["model"].transform("count")
    valid["best_metric_in_dataset"] = valid.groupby("dataset")["metric_value_for_ranking"].transform(
        "min" if task == "regression" else "max"
    )
    valid["worst_metric_in_dataset"] = valid.groupby("dataset")["metric_value_for_ranking"].transform(
        "max" if task == "regression" else "min"
    )
    valid["is_best_tied"] = valid.groupby("dataset")["rank"].transform(lambda s: (s == s.min()).sum() > 1)
    valid["is_dataset_winner"] = valid["rank"] == valid.groupby("dataset")["rank"].transform("min")

    return valid, errors


def build_mean_ranks(ranks_df: pd.DataFrame) -> pd.DataFrame:
    if ranks_df.empty:
        return pd.DataFrame(
            columns=[
                "task",
                "model",
                "mean_rank",
                "median_rank",
                "std_rank",
                "n_datasets",
                "n_wins",
                "win_rate",
                "mean_metric_value",
                "median_metric_value",
                "metric_used_for_ranking",
            ]
        )

    grouped = ranks_df.groupby(["task", "model"], dropna=False)
    out = grouped.agg(
        mean_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        std_rank=("rank", "std"),
        n_datasets=("dataset", "nunique"),
        n_wins=("is_dataset_winner", "sum"),
        mean_metric_value=("metric_value_for_ranking", "mean"),
        median_metric_value=("metric_value_for_ranking", "median"),
        metric_used_for_ranking=("metric_used_for_ranking", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
    ).reset_index()

    out["win_rate"] = np.where(out["n_datasets"] > 0, out["n_wins"] / out["n_datasets"], np.nan)
    out = out.sort_values(["task", "mean_rank", "model"], kind="stable").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()

    input_path = Path(args.master_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)
    ensure_required_columns(df, REQUIRED_ID_COLUMNS)

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)
    df["task"] = df["task"].apply(normalize_task)

    if "status" in df.columns:
        allowed_status = {s.strip() for s in args.keep_status_filter.split(",") if s.strip()}
        if allowed_status:
            df = df[df["status"].astype(str).isin(allowed_status)].copy()

    all_ranks: List[pd.DataFrame] = []
    all_errors: List[pd.DataFrame] = []

    for task, df_task in df.groupby("task", dropna=False):
        ranks_df, errors_df = compute_ranks_for_task(df_task, task, args.rank_method)
        if not ranks_df.empty:
            all_ranks.append(ranks_df)
        if not errors_df.empty:
            all_errors.append(errors_df)

    datasetwise_ranks = pd.concat(all_ranks, ignore_index=True) if all_ranks else pd.DataFrame()
    errors = pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame()
    mean_ranks = build_mean_ranks(datasetwise_ranks)

    datasetwise_ranks_parquet = output_dir / "datasetwise_ranks.parquet"
    datasetwise_ranks_csv = output_dir / "datasetwise_ranks.csv"
    mean_ranks_parquet = output_dir / "mean_ranks.parquet"
    mean_ranks_csv = output_dir / "mean_ranks.csv"
    errors_csv = output_dir / "datasetwise_ranks_errors.csv"

    if not datasetwise_ranks.empty:
        datasetwise_ranks.to_parquet(datasetwise_ranks_parquet, index=False)
    else:
        pd.DataFrame().to_parquet(datasetwise_ranks_parquet, index=False)
    datasetwise_ranks.to_csv(datasetwise_ranks_csv, index=False)

    if not mean_ranks.empty:
        mean_ranks.to_parquet(mean_ranks_parquet, index=False)
    else:
        pd.DataFrame().to_parquet(mean_ranks_parquet, index=False)
    mean_ranks.to_csv(mean_ranks_csv, index=False)

    errors.to_csv(errors_csv, index=False)

    print(f"[INFO] Saved dataset-wise ranks to: {datasetwise_ranks_parquet}")
    print(f"[INFO] Saved mean ranks to: {mean_ranks_parquet}")
    print(f"[INFO] Saved error log to: {errors_csv}")


if __name__ == "__main__":
    main()
