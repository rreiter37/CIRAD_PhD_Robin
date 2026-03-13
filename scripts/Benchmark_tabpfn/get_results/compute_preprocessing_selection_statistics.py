#!/usr/bin/env python3
"""
compute_preprocessing_selection_statistics.py

Compute preprocessing selection statistics from the master_results table.

This script extracts statistics about:
    - how often each full preprocessing pipeline is selected
    - how often each preprocessing component appears
    - frequencies per model and per task

Input
-----
master_results.parquet or master_results.csv

Output
------
preprocessing_selection_stats.parquet
preprocessing_component_stats.parquet
preprocessing_model_task_stats.parquet
preprocessing_dataset_level.parquet
preprocessing_selection_errors.csv
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def read_table(path):
    """Read parquet or CSV file."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported file format")


def normalize_token(token):
    """Normalize a preprocessing token."""
    token = re.sub(r"\s+", " ", str(token)).strip()
    return token


def split_pipeline(pipeline):
    """
    Split pipeline string using common separators.
    """
    if "|" in pipeline:
        return [p.strip() for p in pipeline.split("|")]
    if "->" in pipeline:
        return [p.strip() for p in pipeline.split("->")]
    if ";" in pipeline:
        return [p.strip() for p in pipeline.split(";")]
    if "," in pipeline:
        return [p.strip() for p in pipeline.split(",")]
    return [pipeline]


def parse_pipeline(pipeline, none_tokens):
    """
    Convert pipeline string into normalized components.
    """
    if pd.isna(pipeline):
        return [], True

    pipeline = normalize_token(pipeline)

    if pipeline.lower() in none_tokens:
        return [], True

    parts = split_pipeline(pipeline)

    components = []
    for p in parts:
        token = normalize_token(p)
        if token.lower() in none_tokens:
            continue
        components.append(token)

    if len(components) == 0:
        return [], True

    return components, False


# -------------------------------------------------------------------------
# Core computation
# -------------------------------------------------------------------------

def build_dataset_level_table(df, dataset_col, model_col, task_col, pipeline_col):

    rows = []
    errors = []

    for _, row in df.iterrows():

        dataset = row[dataset_col]
        model = row[model_col]
        task = row[task_col]

        pipeline = row[pipeline_col]

        try:
            components, is_raw = parse_pipeline(
                pipeline,
                none_tokens={"none", "raw", "identity", "null", "nan", ""}
            )
        except Exception as e:

            errors.append({
                "dataset": dataset,
                "model": model,
                "task": task,
                "pipeline_value": pipeline,
                "error": str(e)
            })

            continue

        rows.append({
            "dataset": dataset,
            "model": model,
            "task": task,
            "original_pipeline": pipeline,
            "canonical_pipeline": " | ".join(components) if components else "NONE",
            "components": json.dumps(components),
            "n_components": len(components),
            "is_raw_like": is_raw
        })

    return pd.DataFrame(rows), pd.DataFrame(errors)


def build_pipeline_stats(df):

    stats = (
        df.groupby(["task", "canonical_pipeline"])
        .agg(
            count=("dataset", "size"),
            n_models=("model", "nunique"),
            n_datasets=("dataset", "nunique")
        )
        .reset_index()
    )

    stats["frequency"] = stats["count"] / len(df)

    return stats.sort_values(["task", "count"], ascending=[True, False])


def build_component_stats(df):

    rows = []

    for _, row in df.iterrows():

        comps = json.loads(row["components"])

        for c in comps:
            rows.append({
                "dataset": row["dataset"],
                "model": row["model"],
                "task": row["task"],
                "component": c
            })

    comp_df = pd.DataFrame(rows)

    if comp_df.empty:
        return comp_df

    stats = (
        comp_df.groupby(["task", "component"])
        .agg(
            count=("dataset", "size"),
            n_models=("model", "nunique"),
            n_datasets=("dataset", "nunique")
        )
        .reset_index()
    )

    stats["frequency"] = stats["count"] / len(comp_df)

    return stats.sort_values(["task", "count"], ascending=[True, False])


def build_model_task_stats(df):

    stats = (
        df.groupby(["task", "model", "canonical_pipeline"])
        .agg(
            count=("dataset", "size"),
            n_datasets=("dataset", "nunique")
        )
        .reset_index()
    )

    totals = (
        df.groupby(["task", "model"])
        .size()
        .rename("model_total")
        .reset_index()
    )

    stats = stats.merge(totals, on=["task", "model"])

    stats["frequency_within_model"] = stats["count"] / stats["model_total"]

    stats = stats.drop(columns=["model_total"])

    return stats.sort_values(["task", "model", "count"], ascending=[True, True, False])


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--master_results", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--dataset_col", default="dataset")
    parser.add_argument("--model_col", default="model")
    parser.add_argument("--task_col", default="task")
    parser.add_argument("--pipeline_col", default="preprocessing_pipeline")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.master_results)

    dataset_level, errors = build_dataset_level_table(
        df,
        args.dataset_col,
        args.model_col,
        args.task_col,
        args.pipeline_col
    )

    pipeline_stats = build_pipeline_stats(dataset_level)
    component_stats = build_component_stats(dataset_level)
    model_task_stats = build_model_task_stats(dataset_level)

    dataset_level.to_parquet(output_dir / "preprocessing_dataset_level.parquet", index=False)
    dataset_level.to_csv(output_dir / "preprocessing_dataset_level.csv", index=False)

    pipeline_stats.to_parquet(output_dir / "preprocessing_selection_stats.parquet", index=False)
    pipeline_stats.to_csv(output_dir / "preprocessing_selection_stats.csv", index=False)

    component_stats.to_parquet(output_dir / "preprocessing_component_stats.parquet", index=False)
    component_stats.to_csv(output_dir / "preprocessing_component_stats.csv", index=False)

    model_task_stats.to_parquet(output_dir / "preprocessing_model_task_stats.parquet", index=False)
    model_task_stats.to_csv(output_dir / "preprocessing_model_task_stats.csv", index=False)

    errors.to_csv(output_dir / "preprocessing_selection_errors.csv", index=False)

    print("Preprocessing statistics saved to:", output_dir)


if __name__ == "__main__":
    main()