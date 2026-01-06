#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare TabPFN (raw) vs TabPFN + RFF results.

This script:
  - Parses best prediction CSV files from multiple workspaces
  - Aggregates performance metrics per dataset and per model
  - Saves a global comparison CSV
  - Generates:
      * Dataset-wise barplots
      * Inter-dataset heatmap (datasets x models)
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


# ===================== CLI ===================== #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workspaces",
        nargs="+",
        required=True,
        help="List of workspaces to compare (e.g. raw and rff)."
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Human-readable labels for each workspace (same order)."
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="RMSE",
        help="Metric used for comparison (e.g. RMSE, MSE, R2)."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results",
        help="Directory where CSVs and figures will be saved."
    )

    return parser.parse_args()


# ===================== Data loading ===================== #

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


def load_best_predictions(workspace: Path, label: str, metric: str) -> pd.DataFrame:
    """
    Load all Best_prediction CSV files from a workspace
    and compute metrics from y_true / y_pred.
    """
    rows = []

    for csv_path in workspace.rglob("Best_prediction_*.csv"):
        df = pd.read_csv(csv_path)

        # Basic sanity check
        if not {"y_true", "y_pred"}.issubset(df.columns):
            raise KeyError(
                f"{csv_path} does not contain required columns y_true / y_pred"
            )

        y_true = df["y_true"].values
        y_pred = df["y_pred"].values

        value = compute_metric(y_true, y_pred, metric)

        # Infer dataset name from folder
        dataset_name = csv_path.parent.name
        dataset_name = dataset_name.split("_", 1)[-1]  # robust to date prefix

        rows.append({
            "dataset": dataset_name,
            "model": label,
            "metric": metric.upper(),
            "value": value,
        })

    return pd.DataFrame(rows)


# ===================== Visualization ===================== #

def plot_dataset_bars(df: pd.DataFrame, metric: str, outdir: Path):
    """
    Generate one barplot per dataset comparing models.
    """
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]

        plt.figure(figsize=(6, 4))
        sns.barplot(data=sub, x="model", y="value")
        plt.title(f"{dataset} – {metric}")
        plt.ylabel(metric)
        plt.xlabel("Model")
        plt.tight_layout()

        outpath = outdir / f"{dataset}_barplot_{metric}.png"
        plt.savefig(outpath, dpi=300)
        plt.close()


def save_colored_table(df: pd.DataFrame, metric: str, outdir: Path):
    """
    Create a table (models x datasets) with:
      - Green cell for best model (lowest RMSE) per dataset
      - Red cell for worst model per dataset
    """
    # Pivot: rows = model, columns = dataset
    table = df.pivot(index="model", columns="dataset", values="value")

    def color_best_worst(col):
        """
        Color best (min) in green and worst (max) in red for one dataset.
        """
        min_val = col.min()
        max_val = col.max()

        styles = []
        for v in col:
            if v == min_val:
                styles.append("background-color: #b6e3b6")  # green
            elif v == max_val:
                styles.append("background-color: #f4b6b6")  # red
            else:
                styles.append("")
        return styles

    styled = (
        table
        .style
        .apply(color_best_worst, axis=0)
        .format("{:.4f}")
    )

    # Save as HTML (best for colors)
    html_path = outdir / f"table_comparison_{metric}.html"
    styled.to_html(html_path)

    # Also save raw numeric table as CSV
    csv_path = outdir / f"table_comparison_{metric}.csv"
    table.to_csv(csv_path)

    print(f"Saved colored table → {html_path}")
    print(f"Saved numeric table → {csv_path}")


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
            raise FileNotFoundError(f"Workspace not found: {ws_path}")

        df_ws = load_best_predictions(ws_path, label, args.metric)
        all_results.append(df_ws)

    df_all = pd.concat(all_results, ignore_index=True)

    # Aggregate duplicate (dataset, model) entries
    df_all = (
        df_all
        .groupby(["dataset", "model", "metric"], as_index=False)
        .agg(value=("value", "mean"))
    )

    # ===================== Save CSV ===================== #

    csv_path = outdir / f"tabpfn_rff_comparison_{args.metric}.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV → {csv_path}")

    # ===================== Visuals ===================== #

    plot_dataset_bars(df_all, args.metric, outdir)
    save_colored_table(df_all, args.metric, outdir)

    print(f"Figures saved in → {outdir}")


if __name__ == "__main__":
    main()
