#!/usr/bin/env python3
"""
Aggregate PCA-search CSV results and produce:
1) A boxplot of mean_accuracy per PCA technique (across datasets)
2) A "wins" histogram: how often each PCA technique is best per dataset

Expected inputs:
- Multiple CSV files matching: pca_search__*.csv
- Each CSV contains at least:
    - column "pca" (e.g., no_pca, pca_adapt_0.25n, pca_adapt_0.10n, pca_0.99)
    - column "mean_accuracy" (float)

Example:
python scripts/analyze_pca_search_results.py \
  --input_dir Results/tabpfn_classif_pca_search \
  --output_dir Results/tabpfn_classif_pca_search/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing pca_search__*.csv files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="pca_search__*.csv",
        help="Glob pattern used inside input_dir (default: pca_search__*.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where figures and aggregated CSV will be saved.",
    )
    parser.add_argument(
        "--ties",
        type=str,
        default="split",
        choices=["split", "all", "first"],
        help=(
            "How to handle ties for best mean_accuracy within a dataset:\n"
            "- split: split 1 win equally among tied techniques\n"
            "- all: give 1 full win to each tied technique\n"
            "- first: pick the first after sorting by technique name"
        ),
    )
    parser.add_argument(
        "--pca_order",
        type=str,
        default="no_pca,pca_adapt_0.25n,pca_adapt_0.10n,pca_0.99",
        help="Comma-separated order for PCA techniques in plots.",
    )
    return parser.parse_args()


def infer_dataset_name_from_filename(path: Path) -> str:
    """
    Extract dataset name from 'pca_search__DATASET.csv' as 'DATASET'.
    Falls back to the stem if the prefix is not present.
    """
    m = re.match(r"^pca_search__?(.*)$", path.stem)
    return m.group(1) if m else path.stem


def load_all_csvs(input_dir: Path, pattern: str) -> pd.DataFrame:
    """
    Load all matching CSV files and concatenate into a single DataFrame,
    adding a 'dataset' column derived from the filename.
    """
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} matching pattern '{pattern}'.")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if "pca" not in df.columns or "mean_accuracy" not in df.columns:
            raise ValueError(
                f"File {fp} must contain columns 'pca' and 'mean_accuracy'. "
                f"Found columns: {list(df.columns)}"
            )

        df = df.copy()
        df["dataset"] = infer_dataset_name_from_filename(fp)
        df["pca"] = df["pca"].astype(str)
        df["mean_accuracy"] = pd.to_numeric(df["mean_accuracy"], errors="coerce")
        df = df.dropna(subset=["mean_accuracy"])

        dfs.append(df[["dataset", "pca", "mean_accuracy"]])

    return pd.concat(dfs, ignore_index=True)


def compute_wins(df: pd.DataFrame, ties_mode: str) -> pd.Series:
    """
    Compute win counts per PCA technique based on best mean_accuracy per dataset.
    Returns a Series indexed by pca with win counts (float if ties are split).
    """
    wins = {}

    for dataset, g in df.groupby("dataset", sort=False):
        g = g.dropna(subset=["mean_accuracy"])
        if g.empty:
            continue

        best_val = g["mean_accuracy"].max()
        winners = g.loc[np.isclose(g["mean_accuracy"].values, best_val), "pca"].tolist()

        if not winners:
            continue

        if ties_mode == "first":
            winner = sorted(winners)[0]
            wins[winner] = wins.get(winner, 0.0) + 1.0
        elif ties_mode == "all":
            for w in set(winners):
                wins[w] = wins.get(w, 0.0) + 1.0
        else:  # "split"
            unique_winners = sorted(set(winners))
            share = 1.0 / float(len(unique_winners))
            for w in unique_winners:
                wins[w] = wins.get(w, 0.0) + share

    return pd.Series(wins, dtype=float).sort_index()


def ensure_order(index_like, desired_order):
    """
    Reindex a Series/DataFrame to a desired order, appending any unseen labels at the end.
    """
    desired = [x for x in desired_order if x in index_like]
    extras = [x for x in index_like if x not in desired_order]
    return desired + sorted(extras)


def plot_boxplot(df: pd.DataFrame, out_png: Path, pca_order: list[str]) -> None:
    """
    Boxplot of mean_accuracy per PCA technique.
    """
    groups = []
    labels = []
    for p in pca_order:
        vals = df.loc[df["pca"] == p, "mean_accuracy"].values
        if len(vals) == 0:
            continue
        groups.append(vals)
        labels.append(p)

    if not groups:
        raise RuntimeError("No data available to plot boxplot (check PCA labels / filters).")

    plt.figure()
    plt.boxplot(groups, labels=labels, showfliers=True)
    plt.ylabel("mean_accuracy")
    plt.title("Distribution of mean_accuracy by PCA technique")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_wins_bar(wins: pd.Series, out_png: Path, pca_order: list[str], ties_mode: str) -> None:
    """
    Bar chart of number of dataset wins per PCA technique.
    """
    if wins.empty:
        raise RuntimeError("No wins computed (check inputs).")

    ordered_idx = ensure_order(list(wins.index), pca_order)
    wins = wins.reindex(ordered_idx).fillna(0.0)

    plt.figure()
    plt.bar(wins.index.tolist(), wins.values.tolist())
    plt.ylabel("wins (count)")
    plt.title(f"Wins per PCA technique (ties: {ties_mode})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_order = [x.strip() for x in args.pca_order.split(",") if x.strip()]

    df = load_all_csvs(input_dir=input_dir, pattern=args.pattern)

    # Save an aggregated CSV for convenience
    aggregated_csv = output_dir / "pca_search__ALL.csv"
    df.to_csv(aggregated_csv, index=False)

    # Compute wins
    wins = compute_wins(df, ties_mode=args.ties)

    # Save wins table
    wins_csv = output_dir / "pca_wins.csv"
    wins.sort_values(ascending=False).to_csv(wins_csv, header=["wins"])

    # Plots
    boxplot_png = output_dir / "boxplot_mean_accuracy_by_pca.png"
    wins_png = output_dir / "wins_by_pca.png"

    # Use observed PCA labels order, but enforce your preferred order first
    observed_order = sorted(df["pca"].unique().tolist())
    final_order = ensure_order(observed_order, pca_order)

    plot_boxplot(df, boxplot_png, final_order)
    plot_wins_bar(wins, wins_png, final_order, ties_mode=args.ties)

    print(f"Loaded rows: {len(df)} from directory: {input_dir}")
    print(f"Saved aggregated CSV: {aggregated_csv}")
    print(f"Saved wins CSV:       {wins_csv}")
    print(f"Saved boxplot:        {boxplot_png}")
    print(f"Saved wins plot:      {wins_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
