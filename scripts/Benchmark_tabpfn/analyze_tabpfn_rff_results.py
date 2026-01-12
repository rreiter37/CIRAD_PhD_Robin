#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare TabPFN (raw) vs TabPFN + RFF results.

This script:
  - Parses test and validation prediction CSV files
  - Aggregates metrics per dataset / model / split
  - Supports validation metrics:
      * mean over folds
      * best fold
  - Saves comparison CSVs
  - Generates colored comparison tables
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ===================== CLI ===================== #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workspaces",
        nargs="+",
        required=True,
        help="List of workspaces to compare."
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Human-readable labels for each workspace."
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="RMSE",
        help="Metric used for comparison (RMSE, MSE, MAE, R2)."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Directory where CSVs and tables will be saved."
    )

    return parser.parse_args()


# ===================== Metrics ===================== #

def compute_metric(y_true, y_pred, metric: str) -> float:
    """
    Compute regression metric from y_true and y_pred.
    """
    metric = metric.lower()

    if metric == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == "mse":
        return mean_squared_error(y_true, y_pred)
    elif metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


# ===================== Data loading ===================== #

def load_predictions(workspace: Path, label: str, metric: str) -> pd.DataFrame:
    """
    Load test and validation predictions from a workspace.

    Returns a DataFrame with columns:
      dataset | model | split | metric | value
    """
    rows = []

    # ---------- Test predictions ----------
    for csv_path in workspace.rglob("Best_prediction_*.csv"):
        df = pd.read_csv(csv_path)

        if not {"y_true", "y_pred"}.issubset(df.columns):
            continue

        value = compute_metric(df["y_true"], df["y_pred"], metric)

        dataset = csv_path.parent.name.split("_", 1)[-1]

        rows.append({
            "dataset": dataset,
            "model": label,
            "split": "test",
            "metric": metric.upper(),
            "value": value,
        })

    # ---------- Validation folds ----------
    val_rows = []

    for csv_path in workspace.rglob("Validation_fold_*.csv"):
        df = pd.read_csv(csv_path)

        if not {"y_true", "y_pred"}.issubset(df.columns):
            continue

        value = compute_metric(df["y_true"], df["y_pred"], metric)

        dataset = csv_path.parent.name.split("_", 1)[-1]

        val_rows.append({
            "dataset": dataset,
            "model": label,
            "value": value,
        })

    if val_rows:
        df_val = pd.DataFrame(val_rows)

        # Mean over folds
        df_mean = (
            df_val
            .groupby(["dataset", "model"], as_index=False)
            .agg(value=("value", "mean"))
        )
        df_mean["split"] = "val_mean"
        df_mean["metric"] = metric.upper()

        # Best fold (min for error metrics, max for R2)
        if metric.lower() == "r2":
            idx = df_val.groupby(["dataset", "model"])["value"].idxmax()
        else:
            idx = df_val.groupby(["dataset", "model"])["value"].idxmin()

        df_best = df_val.loc[idx].copy()
        df_best["split"] = "val_best"
        df_best["metric"] = metric.upper()

        rows.extend(df_mean.to_dict("records"))
        rows.extend(df_best.to_dict("records"))

    return pd.DataFrame(rows)


# ===================== Tables ===================== #

def save_colored_table(df: pd.DataFrame, split: str, metric: str, outdir: Path):
    """
    Create a colored comparison table for a given split.
    """
    sub = df[df["split"] == split]
    table = sub.pivot(index="model", columns="dataset", values="value")

    def color_best_worst(col):
        min_val = col.min()
        max_val = col.max()
        styles = []
        for v in col:
            if v == min_val:
                styles.append("background-color: #b6e3b6")
            elif v == max_val:
                styles.append("background-color: #f4b6b6")
            else:
                styles.append("")
        return styles

    styled = (
        table
        .style
        .apply(color_best_worst, axis=0)
        .format("{:.4f}")
    )

    html_path = outdir / f"table_{split}_{metric}.html"
    csv_path = outdir / f"table_{split}_{metric}.csv"

    styled.to_html(html_path)
    table.to_csv(csv_path)

    print(f"Saved table → {html_path}")


# ===================== Main ===================== #

def main():
    args = parse_args()

    if len(args.workspaces) != len(args.labels):
        raise ValueError("Number of workspaces must match number of labels.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ws, label in zip(args.workspaces, args.labels):
        ws_path = Path(ws)
        if not ws_path.exists():
            raise FileNotFoundError(ws_path)

        df_ws = load_predictions(ws_path, label, args.metric)
        all_results.append(df_ws)

    df_all = pd.concat(all_results, ignore_index=True)

    # Aggregate duplicate entries
    df_all = (
        df_all
        .groupby(["dataset", "model", "split", "metric"], as_index=False)
        .agg(value=("value", "mean"))
    )

    # Save global CSV
    csv_path = outdir / f"tabpfn_rff_comparison_all_splits_{args.metric}.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV → {csv_path}")

    # Save tables per split
    for split in ["test", "val_mean", "val_best"]:
        if split in df_all["split"].unique():
            save_colored_table(df_all, split, args.metric, outdir)


if __name__ == "__main__":
    main()
