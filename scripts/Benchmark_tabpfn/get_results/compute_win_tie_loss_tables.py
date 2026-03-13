#!/usr/bin/env python3
"""
compute_win_tie_loss_tables.py

Build pairwise win/tie/loss tables from a master results file.

This script is intentionally flexible:
- it supports missing models on some datasets,
- it supports both regression and classification if the required metric exists,
- it can compare all models pairwise or focus on one reference model.

Expected inputs
---------------
A master results table in parquet or CSV format, typically produced by
build_master_results_table.py.

Expected outputs
----------------
- win_tie_loss_long.parquet / .csv
- win_tie_loss_matrix.parquet / .csv
- win_tie_loss_summary.parquet / .csv
- win_tie_loss_errors.csv

Comparison rules
----------------
- Regression: lower metric is better.
  Default metric priority: RMSEP, then rmsep.
- Classification: higher metric is better.
  Default metric priority: ACCP, then accuracy-compatible columns.

A tie is declared when the absolute difference between two model scores is
smaller than or equal to a tolerance:
- absolute tolerance only by default,
- or absolute + relative tolerance if requested.

All code comments are in English, as requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise win/tie/loss tables from a master results table."
    )
    parser.add_argument(
        "--master_results",
        type=str,
        required=True,
        help="Path to master_results.parquet or master_results.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where outputs will be written",
    )
    parser.add_argument(
        "--dataset_col",
        type=str,
        default="dataset",
        help="Dataset column name in the master table",
    )
    parser.add_argument(
        "--model_col",
        type=str,
        default="model",
        help="Model column name in the master table",
    )
    parser.add_argument(
        "--task_col",
        type=str,
        default="task",
        help="Task column name in the master table",
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="Optional reference model. If provided, only pairwise comparisons against this model are kept.",
    )
    parser.add_argument(
        "--regression_metric",
        type=str,
        default=None,
        help="Optional explicit regression metric column. If omitted, the script tries RMSEP then rmsep.",
    )
    parser.add_argument(
        "--classification_metric",
        type=str,
        default=None,
        help="Optional explicit classification metric column. If omitted, the script tries ACCP then common accuracy columns.",
    )
    parser.add_argument(
        "--tie_abs_tol",
        type=float,
        default=0.0,
        help="Absolute tolerance for declaring a tie",
    )
    parser.add_argument(
        "--tie_rel_tol",
        type=float,
        default=0.0,
        help="Relative tolerance for declaring a tie. Applied to max(|a|, |b|).",
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default="all",
        choices=["all", "regression", "classification"],
        help="Restrict analysis to one task if desired",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Metric selection helpers
# -----------------------------------------------------------------------------

def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_metric_columns(
    df: pd.DataFrame,
    regression_metric: Optional[str],
    classification_metric: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    # Regression metric priority: explicit argument, then common RMSEP spellings
    reg_col = regression_metric
    if reg_col is None:
        reg_col = first_existing_column(
            df,
            ["RMSEP", "rmsep", "RMSE_test", "rmse_test", "RMSE", "rmse"],
        )

    # Classification metric priority: explicit argument, then common accuracy spellings
    cls_col = classification_metric
    if cls_col is None:
        cls_col = first_existing_column(
            df,
            ["ACCP", "accp", "accuracy", "Accuracy", "balanced_accuracy", "acc_test"],
        )

    return reg_col, cls_col


def select_metric_for_row(row: pd.Series, task_col: str, reg_col: Optional[str], cls_col: Optional[str]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    task = str(row.get(task_col, "")).strip().lower()

    if task == "regression":
        if reg_col is None or reg_col not in row.index:
            return None, None, "missing_regression_metric"
        value = row.get(reg_col)
        return reg_col, value, None

    if task == "classification":
        if cls_col is None or cls_col not in row.index:
            return None, None, "missing_classification_metric"
        value = row.get(cls_col)
        return cls_col, value, None

    return None, None, f"unknown_task:{task}"


# -----------------------------------------------------------------------------
# Comparison helpers
# -----------------------------------------------------------------------------

def is_tie(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    threshold = max(abs_tol, rel_tol * max(abs(a), abs(b)))
    return abs(a - b) <= threshold


def compare_scores(task: str, score_a: float, score_b: float, abs_tol: float, rel_tol: float) -> str:
    """
    Return the result from the perspective of model A:
    - 'win'
    - 'tie'
    - 'loss'
    """
    if is_tie(score_a, score_b, abs_tol, rel_tol):
        return "tie"

    if task == "regression":
        return "win" if score_a < score_b else "loss"

    if task == "classification":
        return "win" if score_a > score_b else "loss"

    raise ValueError(f"Unsupported task for comparison: {task}")


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------

def compute_pairwise_long_table(
    df: pd.DataFrame,
    dataset_col: str,
    model_col: str,
    task_col: str,
    reg_col: Optional[str],
    cls_col: Optional[str],
    tie_abs_tol: float,
    tie_rel_tol: float,
    reference_model: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a long pairwise table with one row per:
    dataset x task x model_a x model_b

    Returns
    -------
    long_df, errors_df
    """
    rows: List[Dict] = []
    errors: List[Dict] = []

    required_cols = [dataset_col, model_col, task_col]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in master table: {missing_required}")

    # Keep only rows with a dataset and a model
    work_df = df.copy()
    work_df = work_df[work_df[dataset_col].notna() & work_df[model_col].notna()].copy()

    # Group by dataset and task to compare models only within the same dataset/task
    for (dataset_name, task_name), g in work_df.groupby([dataset_col, task_col], dropna=False):
        task_str = str(task_name).strip().lower()

        # Build one clean score record per model
        model_records: List[Dict] = []
        seen_models = set()

        for _, row in g.iterrows():
            model_name = str(row[model_col]).strip()

            # If duplicates exist for the same dataset/model/task, keep the first valid line
            if model_name in seen_models:
                errors.append(
                    {
                        "dataset": dataset_name,
                        "task": task_name,
                        "model": model_name,
                        "status": "duplicate_model_row_same_dataset_task",
                    }
                )
                continue

            metric_col, metric_value, error_status = select_metric_for_row(
                row=row,
                task_col=task_col,
                reg_col=reg_col,
                cls_col=cls_col,
            )

            if error_status is not None:
                errors.append(
                    {
                        "dataset": dataset_name,
                        "task": task_name,
                        "model": model_name,
                        "status": error_status,
                    }
                )
                continue

            if pd.isna(metric_value):
                errors.append(
                    {
                        "dataset": dataset_name,
                        "task": task_name,
                        "model": model_name,
                        "status": "metric_is_nan",
                        "metric_col": metric_col,
                    }
                )
                continue

            seen_models.add(model_name)
            model_records.append(
                {
                    "dataset": dataset_name,
                    "task": task_name,
                    "model": model_name,
                    "metric_name": metric_col,
                    "metric_value": float(metric_value),
                }
            )

        if len(model_records) < 2:
            # Not enough models to compare on this dataset
            if len(model_records) == 1:
                errors.append(
                    {
                        "dataset": dataset_name,
                        "task": task_name,
                        "model": model_records[0]["model"],
                        "status": "single_model_only_no_pairwise_comparison",
                    }
                )
            continue

        model_df = pd.DataFrame(model_records)

        # Restrict to the requested reference model if needed
        if reference_model is not None:
            ref_mask = model_df["model"] == reference_model
            if not ref_mask.any():
                errors.append(
                    {
                        "dataset": dataset_name,
                        "task": task_name,
                        "model": reference_model,
                        "status": "reference_model_missing_for_dataset",
                    }
                )
                continue

            ref_row = model_df.loc[ref_mask].iloc[0]
            others_df = model_df.loc[~ref_mask].copy()

            for _, other in others_df.iterrows():
                result_ref = compare_scores(
                    task=task_str,
                    score_a=float(ref_row["metric_value"]),
                    score_b=float(other["metric_value"]),
                    abs_tol=tie_abs_tol,
                    rel_tol=tie_rel_tol,
                )
                result_other = {"win": "loss", "loss": "win", "tie": "tie"}[result_ref]

                delta = float(ref_row["metric_value"]) - float(other["metric_value"])
                abs_delta = abs(delta)

                # Row from the perspective of the reference model
                rows.append(
                    {
                        "dataset": dataset_name,
                        "task": task_name,
                        "model_a": ref_row["model"],
                        "model_b": other["model"],
                        "metric_name": ref_row["metric_name"],
                        "score_a": float(ref_row["metric_value"]),
                        "score_b": float(other["metric_value"]),
                        "delta_a_minus_b": delta,
                        "abs_delta": abs_delta,
                        "result_for_a": result_ref,
                    }
                )

                # Symmetric row from the perspective of the other model
                rows.append(
                    {
                        "dataset": dataset_name,
                        "task": task_name,
                        "model_a": other["model"],
                        "model_b": ref_row["model"],
                        "metric_name": other["metric_name"],
                        "score_a": float(other["metric_value"]),
                        "score_b": float(ref_row["metric_value"]),
                        "delta_a_minus_b": -delta,
                        "abs_delta": abs_delta,
                        "result_for_a": result_other,
                    }
                )
        else:
            # Full pairwise comparison across all models
            recs = model_df.to_dict(orient="records")
            for i in range(len(recs)):
                for j in range(len(recs)):
                    if i == j:
                        continue

                    a = recs[i]
                    b = recs[j]

                    result_a = compare_scores(
                        task=task_str,
                        score_a=float(a["metric_value"]),
                        score_b=float(b["metric_value"]),
                        abs_tol=tie_abs_tol,
                        rel_tol=tie_rel_tol,
                    )

                    rows.append(
                        {
                            "dataset": dataset_name,
                            "task": task_name,
                            "model_a": a["model"],
                            "model_b": b["model"],
                            "metric_name": a["metric_name"],
                            "score_a": float(a["metric_value"]),
                            "score_b": float(b["metric_value"]),
                            "delta_a_minus_b": float(a["metric_value"]) - float(b["metric_value"]),
                            "abs_delta": abs(float(a["metric_value"]) - float(b["metric_value"])),
                            "result_for_a": result_a,
                        }
                    )

    long_df = pd.DataFrame(rows)
    errors_df = pd.DataFrame(errors)
    return long_df, errors_df


def build_summary_table(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "task", "model_a", "model_b", "n_common_datasets",
                "wins", "ties", "losses", "win_rate_excl_ties", "non_loss_rate"
            ]
        )

    summary = (
        long_df.assign(
            win=lambda x: (x["result_for_a"] == "win").astype(int),
            tie=lambda x: (x["result_for_a"] == "tie").astype(int),
            loss=lambda x: (x["result_for_a"] == "loss").astype(int),
        )
        .groupby(["task", "model_a", "model_b"], as_index=False)
        .agg(
            n_common_datasets=("dataset", "nunique"),
            wins=("win", "sum"),
            ties=("tie", "sum"),
            losses=("loss", "sum"),
        )
    )

    # Compute rates
    decisive = summary["wins"] + summary["losses"]
    summary["win_rate_excl_ties"] = np.where(
        decisive > 0,
        summary["wins"] / decisive,
        np.nan,
    )
    summary["non_loss_rate"] = (summary["wins"] + summary["ties"]) / summary["n_common_datasets"]

    return summary.sort_values(["task", "model_a", "model_b"]).reset_index(drop=True)


def build_matrix_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a matrix-like table per task with compact win/tie/loss strings.
    The returned table stays in flat format for easier export.
    """
    if summary_df.empty:
        return pd.DataFrame(columns=["task", "model_a", "model_b", "wtl_string"])

    matrix_df = summary_df.copy()
    matrix_df["wtl_string"] = (
        matrix_df["wins"].astype(int).astype(str)
        + "/"
        + matrix_df["ties"].astype(int).astype(str)
        + "/"
        + matrix_df["losses"].astype(int).astype(str)
    )
    return matrix_df[["task", "model_a", "model_b", "wtl_string"]].sort_values(
        ["task", "model_a", "model_b"]
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    master_path = Path(args.master_results)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    df = read_table(master_path)

    # Restrict to the requested task if needed
    if args.task_filter != "all":
        if args.task_col not in df.columns:
            raise ValueError(f"Column '{args.task_col}' not found in input table.")
        df = df[df[args.task_col].astype(str).str.lower() == args.task_filter].copy()

    reg_col, cls_col = infer_metric_columns(
        df=df,
        regression_metric=args.regression_metric,
        classification_metric=args.classification_metric,
    )

    long_df, errors_df = compute_pairwise_long_table(
        df=df,
        dataset_col=args.dataset_col,
        model_col=args.model_col,
        task_col=args.task_col,
        reg_col=reg_col,
        cls_col=cls_col,
        tie_abs_tol=args.tie_abs_tol,
        tie_rel_tol=args.tie_rel_tol,
        reference_model=args.reference_model,
    )

    summary_df = build_summary_table(long_df)
    matrix_df = build_matrix_table(summary_df)

    # Add metadata columns for traceability
    meta_payload = {
        "master_results": str(master_path),
        "reference_model": args.reference_model,
        "regression_metric": reg_col,
        "classification_metric": cls_col,
        "tie_abs_tol": args.tie_abs_tol,
        "tie_rel_tol": args.tie_rel_tol,
        "task_filter": args.task_filter,
    }

    if not long_df.empty:
        long_df["settings_json"] = json.dumps(meta_payload, ensure_ascii=False)
    if not summary_df.empty:
        summary_df["settings_json"] = json.dumps(meta_payload, ensure_ascii=False)
    if not matrix_df.empty:
        matrix_df["settings_json"] = json.dumps(meta_payload, ensure_ascii=False)
    if not errors_df.empty:
        errors_df["settings_json"] = json.dumps(meta_payload, ensure_ascii=False)

    # Write outputs
    long_parquet = output_dir / "win_tie_loss_long.parquet"
    long_csv = output_dir / "win_tie_loss_long.csv"
    summary_parquet = output_dir / "win_tie_loss_summary.parquet"
    summary_csv = output_dir / "win_tie_loss_summary.csv"
    matrix_parquet = output_dir / "win_tie_loss_matrix.parquet"
    matrix_csv = output_dir / "win_tie_loss_matrix.csv"
    errors_csv = output_dir / "win_tie_loss_errors.csv"

    long_df.to_parquet(long_parquet, index=False)
    long_df.to_csv(long_csv, index=False)

    summary_df.to_parquet(summary_parquet, index=False)
    summary_df.to_csv(summary_csv, index=False)

    matrix_df.to_parquet(matrix_parquet, index=False)
    matrix_df.to_csv(matrix_csv, index=False)

    if errors_df.empty:
        pd.DataFrame(columns=["dataset", "task", "model", "status", "settings_json"]).to_csv(errors_csv, index=False)
    else:
        errors_df.to_csv(errors_csv, index=False)

    print(f"Saved: {long_parquet}")
    print(f"Saved: {long_csv}")
    print(f"Saved: {summary_parquet}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {matrix_parquet}")
    print(f"Saved: {matrix_csv}")
    print(f"Saved: {errors_csv}")


if __name__ == "__main__":
    main()
