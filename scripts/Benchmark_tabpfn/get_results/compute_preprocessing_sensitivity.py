#!/usr/bin/env python3
"""
compute_preprocessing_sensitivity.py

Compute preprocessing sensitivity statistics from a search-level results table.

This script is intended to quantify how sensitive each model is to the choice
of preprocessing pipeline across datasets.

Expected input
--------------
A search-level results table with one row per tested configuration, typically:
    dataset x model x preprocessing_pipeline x score

The script is flexible and can work with parquet or CSV input.

Expected outputs
----------------
- preprocessing_sensitivity_dataset_level.parquet / .csv
- preprocessing_sensitivity_model_summary.parquet / .csv
- preprocessing_sensitivity_task_summary.parquet / .csv
- preprocessing_sensitivity_errors.csv

Main ideas
----------
For each dataset x model group, the script computes:
    - number of tested preprocessing configurations
    - best score
    - worst score
    - mean score
    - std of scores
    - sensitivity range
    - relative sensitivity range
    - raw-vs-best gain if a raw/no-preprocessing configuration exists

By default:
    - regression: lower score is better
    - classification: higher score is better

All code comments are in English, as requested.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def read_table(path):
    """Read a parquet or CSV table."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def ensure_dir(path):
    """Create output directory if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Column inference helpers
# -------------------------------------------------------------------------

def first_existing_column(df, candidates):
    """Return the first existing column among candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_score_columns(df, regression_score_col=None, classification_score_col=None):
    """
    Infer score columns for regression and classification.

    Priority:
    - explicit CLI value
    - common defaults
    """
    reg_col = regression_score_col
    if reg_col is None:
        reg_col = first_existing_column(
            df,
            [
                "mean_score",
                "RMSECV",
                "RMSE_MF",
                "RMSEP",
                "rmsep",
                "rmsecv",
                "score",
            ]
        )

    cls_col = classification_score_col
    if cls_col is None:
        cls_col = first_existing_column(
            df,
            [
                "mean_score",
                "ACCP",
                "accp",
                "accuracy",
                "score",
            ]
        )

    return reg_col, cls_col


def infer_pipeline_column(df, pipeline_col=None):
    """Infer preprocessing pipeline column."""
    if pipeline_col is not None:
        if pipeline_col not in df.columns:
            raise ValueError(f"Requested pipeline column '{pipeline_col}' not found.")
        return pipeline_col

    col = first_existing_column(
        df,
        [
            "preprocessing_pipeline",
            "pipeline",
            "preprocessing",
            "preproc_pipeline",
            "preproc",
            "config",
        ]
    )
    if col is None:
        raise ValueError(
            "Could not infer the preprocessing pipeline column. "
            "Use --pipeline_col explicitly."
        )
    return col


# -------------------------------------------------------------------------
# Pipeline helpers
# -------------------------------------------------------------------------

def normalize_text(text):
    """Normalize text representation."""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_pipeline(text):
    """Split pipeline string using common separators."""
    if "|" in text:
        return [p.strip() for p in text.split("|")]
    if "->" in text:
        return [p.strip() for p in text.split("->")]
    if ";" in text:
        return [p.strip() for p in text.split(";")]
    if "," in text:
        return [p.strip() for p in text.split(",")]
    return [text.strip()]


def is_raw_like_pipeline(pipeline_value, none_tokens=None):
    """
    Detect whether a preprocessing pipeline is equivalent to raw / no preprocessing.
    """
    if none_tokens is None:
        none_tokens = {"none", "raw", "identity", "null", "nan", ""}

    if pd.isna(pipeline_value):
        return True

    text = normalize_text(pipeline_value)
    if text.lower() in none_tokens:
        return True

    parts = split_pipeline(text)
    cleaned = []
    for p in parts:
        token = normalize_text(p)
        if token.lower() in none_tokens:
            continue
        cleaned.append(token)

    return len(cleaned) == 0


# -------------------------------------------------------------------------
# Score semantics
# -------------------------------------------------------------------------

def get_score_for_row(row, task_col, reg_col, cls_col):
    """
    Return the score value and whether lower is better for this row.
    """
    task = str(row[task_col]).strip().lower()

    if task == "regression":
        if reg_col is None or reg_col not in row.index:
            return None, None, "missing_regression_score_col"
        value = row[reg_col]
        return value, True, None

    if task == "classification":
        if cls_col is None or cls_col not in row.index:
            return None, None, "missing_classification_score_col"
        value = row[cls_col]
        return value, False, None

    return None, None, f"unknown_task:{task}"


def compute_best_worst(scores, lower_is_better):
    """Compute best and worst scores according to task direction."""
    scores = np.asarray(scores, dtype=float)
    if lower_is_better:
        best = np.nanmin(scores)
        worst = np.nanmax(scores)
    else:
        best = np.nanmax(scores)
        worst = np.nanmin(scores)
    return best, worst


def compute_relative_range(best_score, worst_score, lower_is_better):
    """
    Compute a relative sensitivity range.

    Regression:
        (worst - best) / |best|
    Classification:
        (best - worst) / |best|

    This yields positive values when spread exists.
    """
    denom = abs(best_score)
    if denom < 1e-12:
        return np.nan

    if lower_is_better:
        return (worst_score - best_score) / denom
    return (best_score - worst_score) / denom


def compute_raw_vs_best_gain(raw_score, best_score, lower_is_better):
    """
    Compute raw-vs-best relative gain.

    Regression:
        positive if best preprocessing reduces the error.
        100 * (raw - best) / raw

    Classification:
        positive if best preprocessing increases accuracy.
        100 * (best - raw) / |raw|
    """
    if raw_score is None or pd.isna(raw_score):
        return np.nan

    denom = abs(raw_score)
    if denom < 1e-12:
        return np.nan

    if lower_is_better:
        return 100.0 * (raw_score - best_score) / denom

    return 100.0 * (best_score - raw_score) / denom


# -------------------------------------------------------------------------
# Core computation
# -------------------------------------------------------------------------

def build_dataset_level_sensitivity(
    df,
    dataset_col,
    model_col,
    task_col,
    pipeline_col,
    reg_col,
    cls_col,
    min_n_configs=2,
):
    """
    Build one sensitivity row per dataset x model.

    This is the main output used later for figures and summaries.
    """
    rows = []
    errors = []

    required_cols = [dataset_col, model_col, task_col, pipeline_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()
    work_df = work_df[
        work_df[dataset_col].notna() &
        work_df[model_col].notna() &
        work_df[task_col].notna()
    ].copy()

    for (dataset_name, model_name, task_name), group in work_df.groupby(
        [dataset_col, model_col, task_col],
        dropna=False
    ):
        valid_rows = []
        group_errors = []

        for _, row in group.iterrows():
            score_value, lower_is_better, status = get_score_for_row(
                row=row,
                task_col=task_col,
                reg_col=reg_col,
                cls_col=cls_col,
            )

            if status is not None:
                group_errors.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "task": task_name,
                        "status": status,
                    }
                )
                continue

            if pd.isna(score_value):
                group_errors.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "task": task_name,
                        "status": "score_is_nan",
                    }
                )
                continue

            valid_rows.append(
                {
                    "score": float(score_value),
                    "pipeline_value": row[pipeline_col],
                    "is_raw_like": is_raw_like_pipeline(row[pipeline_col]),
                    "lower_is_better": bool(lower_is_better),
                }
            )

        if group_errors:
            errors.extend(group_errors)

        if len(valid_rows) == 0:
            continue

        if len(valid_rows) < min_n_configs:
            errors.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "task": task_name,
                    "status": "not_enough_configs_for_sensitivity",
                }
            )
            continue

        lower_is_better = valid_rows[0]["lower_is_better"]
        scores = np.array([r["score"] for r in valid_rows], dtype=float)

        best_score, worst_score = compute_best_worst(scores, lower_is_better)
        mean_score = float(np.nanmean(scores))
        median_score = float(np.nanmedian(scores))
        std_score = float(np.nanstd(scores, ddof=0))

        if lower_is_better:
            sensitivity_range = float(worst_score - best_score)
        else:
            sensitivity_range = float(best_score - worst_score)

        relative_sensitivity_range = compute_relative_range(
            best_score=best_score,
            worst_score=worst_score,
            lower_is_better=lower_is_better,
        )

        raw_candidates = [r["score"] for r in valid_rows if r["is_raw_like"]]
        raw_score = raw_candidates[0] if len(raw_candidates) > 0 else np.nan

        raw_vs_best_gain_pct = compute_raw_vs_best_gain(
            raw_score=raw_score,
            best_score=best_score,
            lower_is_better=lower_is_better,
        )

        best_rank_index = int(np.argmin(scores) if lower_is_better else np.argmax(scores))
        worst_rank_index = int(np.argmax(scores) if lower_is_better else np.argmin(scores))

        best_pipeline = valid_rows[best_rank_index]["pipeline_value"]
        worst_pipeline = valid_rows[worst_rank_index]["pipeline_value"]

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "task": task_name,
                "n_tested_configs": int(len(valid_rows)),
                "best_score": float(best_score),
                "worst_score": float(worst_score),
                "mean_score": mean_score,
                "median_score": median_score,
                "std_score": std_score,
                "sensitivity_range": sensitivity_range,
                "relative_sensitivity_range": relative_sensitivity_range,
                "raw_score": raw_score,
                "raw_vs_best_gain_pct": raw_vs_best_gain_pct,
                "best_pipeline": best_pipeline,
                "worst_pipeline": worst_pipeline,
                "has_raw_config": bool(len(raw_candidates) > 0),
                "score_direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "status": "ok",
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(errors)


def build_model_summary(dataset_level_df):
    """
    Aggregate sensitivity statistics at the model level.
    """
    if dataset_level_df.empty:
        return pd.DataFrame()

    summary = (
        dataset_level_df.groupby(["task", "model"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_n_tested_configs=("n_tested_configs", "mean"),
            mean_best_score=("best_score", "mean"),
            mean_worst_score=("worst_score", "mean"),
            mean_std_score=("std_score", "mean"),
            median_std_score=("std_score", "median"),
            mean_sensitivity_range=("sensitivity_range", "mean"),
            median_sensitivity_range=("sensitivity_range", "median"),
            mean_relative_sensitivity_range=("relative_sensitivity_range", "mean"),
            median_relative_sensitivity_range=("relative_sensitivity_range", "median"),
            mean_raw_vs_best_gain_pct=("raw_vs_best_gain_pct", "mean"),
            median_raw_vs_best_gain_pct=("raw_vs_best_gain_pct", "median"),
            raw_config_coverage=("has_raw_config", "mean"),
        )
    )

    return summary.sort_values(
        ["task", "mean_relative_sensitivity_range", "model"],
        ascending=[True, False, True]
    ).reset_index(drop=True)


def build_task_summary(dataset_level_df):
    """
    Aggregate sensitivity statistics at the task level.
    """
    if dataset_level_df.empty:
        return pd.DataFrame()

    summary = (
        dataset_level_df.groupby(["task"], as_index=False)
        .agg(
            n_rows=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            n_models=("model", "nunique"),
            mean_sensitivity_range=("sensitivity_range", "mean"),
            median_sensitivity_range=("sensitivity_range", "median"),
            mean_relative_sensitivity_range=("relative_sensitivity_range", "mean"),
            median_relative_sensitivity_range=("relative_sensitivity_range", "median"),
            mean_raw_vs_best_gain_pct=("raw_vs_best_gain_pct", "mean"),
            median_raw_vs_best_gain_pct=("raw_vs_best_gain_pct", "median"),
        )
    )

    return summary.sort_values("task").reset_index(drop=True)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute preprocessing sensitivity statistics from a search-level results table."
    )

    parser.add_argument("--search_results", required=True, help="Path to search-level results table (.parquet or .csv)")
    parser.add_argument("--output_dir", required=True, help="Directory where outputs will be written")

    parser.add_argument("--dataset_col", default="dataset", help="Dataset column name")
    parser.add_argument("--model_col", default="model", help="Model column name")
    parser.add_argument("--task_col", default="task", help="Task column name")
    parser.add_argument("--pipeline_col", default=None, help="Preprocessing pipeline column name")

    parser.add_argument("--regression_score_col", default=None, help="Regression score column")
    parser.add_argument("--classification_score_col", default=None, help="Classification score column")

    parser.add_argument(
        "--task_filter",
        default="all",
        choices=["all", "regression", "classification"],
        help="Restrict analysis to one task if needed"
    )

    parser.add_argument(
        "--min_n_configs",
        type=int,
        default=2,
        help="Minimum number of tested configurations required to compute sensitivity"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = read_table(args.search_results)

    if args.task_filter != "all":
        if args.task_col not in df.columns:
            raise ValueError(f"Column '{args.task_col}' not found in input table.")
        df = df[df[args.task_col].astype(str).str.lower() == args.task_filter].copy()

    pipeline_col = infer_pipeline_column(df, args.pipeline_col)
    reg_col, cls_col = infer_score_columns(
        df,
        regression_score_col=args.regression_score_col,
        classification_score_col=args.classification_score_col,
    )

    dataset_level_df, errors_df = build_dataset_level_sensitivity(
        df=df,
        dataset_col=args.dataset_col,
        model_col=args.model_col,
        task_col=args.task_col,
        pipeline_col=pipeline_col,
        reg_col=reg_col,
        cls_col=cls_col,
        min_n_configs=args.min_n_configs,
    )

    model_summary_df = build_model_summary(dataset_level_df)
    task_summary_df = build_task_summary(dataset_level_df)

    settings = {
        "search_results": str(args.search_results),
        "dataset_col": args.dataset_col,
        "model_col": args.model_col,
        "task_col": args.task_col,
        "pipeline_col": pipeline_col,
        "regression_score_col": reg_col,
        "classification_score_col": cls_col,
        "task_filter": args.task_filter,
        "min_n_configs": args.min_n_configs,
    }
    settings_json = json.dumps(settings, ensure_ascii=False)

    for out_df in [dataset_level_df, model_summary_df, task_summary_df]:
        if not out_df.empty:
            out_df["settings_json"] = settings_json

    if errors_df.empty:
        errors_df = pd.DataFrame(
            columns=["dataset", "model", "task", "status", "settings_json"]
        )
    else:
        errors_df["settings_json"] = settings_json

    dataset_level_df.to_parquet(output_dir / "preprocessing_sensitivity_dataset_level.parquet", index=False)
    dataset_level_df.to_csv(output_dir / "preprocessing_sensitivity_dataset_level.csv", index=False)

    model_summary_df.to_parquet(output_dir / "preprocessing_sensitivity_model_summary.parquet", index=False)
    model_summary_df.to_csv(output_dir / "preprocessing_sensitivity_model_summary.csv", index=False)

    task_summary_df.to_parquet(output_dir / "preprocessing_sensitivity_task_summary.parquet", index=False)
    task_summary_df.to_csv(output_dir / "preprocessing_sensitivity_task_summary.csv", index=False)

    errors_df.to_csv(output_dir / "preprocessing_sensitivity_errors.csv", index=False)

    print(f"Saved: {output_dir / 'preprocessing_sensitivity_dataset_level.parquet'}")
    print(f"Saved: {output_dir / 'preprocessing_sensitivity_dataset_level.csv'}")
    print(f"Saved: {output_dir / 'preprocessing_sensitivity_model_summary.parquet'}")
    print(f"Saved: {output_dir / 'preprocessing_sensitivity_model_summary.csv'}")
    print(f"Saved: {output_dir / 'preprocessing_sensitivity_task_summary.parquet'}")
    print(f"Saved: {output_dir / 'preprocessing_sensitivity_task_summary.csv'}")
    print(f"Saved: {output_dir / 'preprocessing_sensitivity_errors.csv'}")


if __name__ == "__main__":
    main()