#!/usr/bin/env python3
"""
Rank "simple_preproc" preprocessings using ONLY Stage 1 metrics (Option A),
and generate:
  - A boxplot of per-dataset best Stage-1 scores per preprocessing
  - A Critical Difference (Nemenyi) diagram based on average ranks across datasets

Inputs:
- A CSV produced by the TabPFN search pipeline (e.g., tabpfn_search_results.csv)

Option A:
1) For each (dataset, simple_preproc), keep the BEST Stage-1 score (min metric).
2) Compute ranks within each dataset and aggregate across datasets.

Outputs (saved next to the input CSV by default):
- ranking_simple_preproc_stage1.csv
- best_per_dataset_simple_preproc_stage1.csv
- matrix_best_metric_stage1.csv
- boxplot_best_metric_stage1.png
- cd_diagram_nemenyi_stage1.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy is used only for the studentized range distribution quantile (Nemenyi CD)
try:
    from scipy.stats import studentized_range
except Exception:
    studentized_range = None


REQUIRED_COLUMNS = ["dataset", "stage", "simple_preproc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank simple preprocessings using ONLY Stage 1 results (Option A) + figures."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to tabpfn_search_results.csv produced by the TabPFN search pipeline.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to the parent directory of csv_path.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_nrmse",
        choices=["val_nrmse", "val_rmse"],
        help="Which Stage-1 metric to use (lower is better). Default: val_nrmse.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the Nemenyi CD diagram. Default: 0.05.",
    )
    parser.add_argument(
        "--max_methods_plot",
        type=int,
        default=25,
        help="Max number of methods to show in plots (top by mean rank). Default: 25.",
    )
    parser.add_argument(
        "--fig_dpi",
        type=int,
        default=200,
        help="DPI for saved PNG figures. Default: 200.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, metric: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if metric not in df.columns:
        missing.append(metric)
    if missing:
        raise ValueError(
            f"Missing required columns in CSV: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def compute_best_stage1(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Keep Stage 1 only, then for each (dataset, simple_preproc) keep the best metric (min).
    Returns a DataFrame with columns: dataset, simple_preproc, best_metric, rank_in_dataset
    """
    df_s1 = df[df["stage"].astype(str).str.lower().eq("stage1")].copy()
    df_s1 = df_s1.dropna(subset=["dataset", "simple_preproc", metric])
    df_s1 = df_s1.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])

    if df_s1.empty:
        raise RuntimeError(
            "No Stage 1 rows found after filtering. "
            "Check that your CSV contains stage=='stage1' rows."
        )

    df_s1["simple_preproc"] = df_s1["simple_preproc"].astype(str)

    best = (
        df_s1.groupby(["dataset", "simple_preproc"], as_index=False)[metric]
        .min()
        .rename(columns={metric: f"best_{metric}"})
    )

    best["rank_in_dataset"] = (
        best.groupby("dataset")[f"best_{metric}"].rank(method="min", ascending=True)
    ).astype(float)

    return best


def aggregate_ranking(best: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Aggregate per-dataset best results into a ranking table.
    """
    mcol = f"best_{metric}"

    summary = (
        best.groupby("simple_preproc")
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_best=(mcol, "mean"),
            median_best=(mcol, "median"),
            mean_rank=("rank_in_dataset", "mean"),
            median_rank=("rank_in_dataset", "median"),
        )
        .reset_index()
        .sort_values(["mean_rank", "mean_best"], ascending=[True, True])
    )
    return summary


def build_matrix(best: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Pivot table simple_preproc x dataset with best metric.
    """
    mcol = f"best_{metric}"
    matrix = best.pivot_table(
        index="simple_preproc", columns="dataset", values=mcol, aggfunc="min"
    )
    return matrix


def save_boxplot(best: pd.DataFrame, summary: pd.DataFrame, metric: str, out_path: Path, top_n: int, dpi: int) -> None:
    """
    Boxplot of per-dataset best metric values by preprocessing (top_n by mean_rank).
    """
    mcol = f"best_{metric}"
    top_methods = summary["simple_preproc"].head(top_n).tolist()

    plot_df = best[best["simple_preproc"].isin(top_methods)].copy()
    # Ensure consistent order
    plot_df["simple_preproc"] = pd.Categorical(plot_df["simple_preproc"], categories=top_methods, ordered=True)
    plot_df = plot_df.sort_values("simple_preproc")

    data = [plot_df.loc[plot_df["simple_preproc"] == m, mcol].values for m in top_methods]

    plt.figure(figsize=(max(10, 0.5 * len(top_methods)), 6))
    plt.boxplot(
        data,
        labels=top_methods,
        showfliers=False,
        vert=True,
    )
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(mcol)
    plt.title(f"Stage 1 — per-dataset BEST {mcol} by simple_preproc (top {len(top_methods)} by mean rank)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def nemenyi_cd(k: int, n_datasets: int, alpha: float) -> float:
    """
    Compute Nemenyi critical difference:
      CD = q_alpha * sqrt( k(k+1) / (6N) )
    where q_alpha is the critical value from the studentized range distribution
    with k groups and infinite df (common approximation).
    """
    if studentized_range is None:
        raise RuntimeError(
            "scipy.stats.studentized_range is not available in your environment. "
            "Install/upgrade SciPy to use the Nemenyi CD diagram."
        )

    if k < 2 or n_datasets < 2:
        return np.nan

    # q_alpha for Nemenyi: studentized range quantile at 1-alpha, k groups, df -> inf
    q_alpha = studentized_range.ppf(1.0 - alpha, k, np.inf)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_datasets))
    return float(cd)


def compute_non_sig_segments(sorted_ranks: List[float], cd: float) -> List[Tuple[int, int]]:
    """
    Build a set of maximal contiguous segments (i,j) in sorted order such that
    rank[j] - rank[i] <= cd. Then remove segments contained in others.
    """
    k = len(sorted_ranks)
    segments: List[Tuple[int, int]] = []

    for i in range(k):
        j = i
        while j + 1 < k and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd + 1e-12:
            j += 1
        if j > i:
            segments.append((i, j))

    # Remove contained segments (keep only maximal)
    maximal: List[Tuple[int, int]] = []
    for seg in segments:
        contained = False
        for other in segments:
            if other == seg:
                continue
            if other[0] <= seg[0] and other[1] >= seg[1]:
                contained = True
                break
        if not contained:
            maximal.append(seg)

    # Deduplicate
    maximal = sorted(set(maximal), key=lambda x: (x[0], x[1]))
    return maximal


def save_cd_diagram(best: pd.DataFrame,
                    summary: pd.DataFrame,
                    metric: str,
                    out_path: Path,
                    alpha: float,
                    top_n: int,
                    dpi: int) -> None:
    """
    Readable Critical Difference diagram (Nemenyi) with side labels (Demšar-style).
    - Axis of average ranks in the middle
    - Method names listed on the left/right columns (no overlap)
    - Lines connecting each method to its rank
    - Non-significant cliques drawn as thick segments on top
    """
    # Keep top methods for readability
    top_methods = summary["simple_preproc"].head(top_n).tolist()
    df = best[best["simple_preproc"].isin(top_methods)].copy()

    # Average ranks (smaller is better)
    avg_rank = (
        df.groupby("simple_preproc")["rank_in_dataset"]
        .mean()
        .reindex(top_methods)
        .sort_values()
    )

    methods_sorted = avg_rank.index.tolist()
    ranks_sorted = avg_rank.values.astype(float).tolist()

    n_datasets = int(df["dataset"].nunique())
    k = len(methods_sorted)
    cd = nemenyi_cd(k=k, n_datasets=n_datasets, alpha=alpha)

    # --- Figure sizing ---
    # Height grows with number of methods to keep labels readable
    fig_h = max(4.0, 0.35 * k)
    fig_w = 12.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    # Add horizontal margins for side labels
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1.0, k + 2.8)

    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title(
        f"Nemenyi Critical Difference Diagram (Stage 1) — alpha={alpha}, "
        f"CD={cd:.3f}, N={n_datasets}"
    )

    # Middle axis line
    y_axis = k + 0.8
    ax.hlines(y_axis, 1, k, linewidth=1.2)
    ax.set_xticks(range(1, k + 1))
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)

    # No y ticks
    ax.set_yticks([])
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    # Split methods into left and right columns (roughly half/half)
    mid = k // 2
    left_methods = list(reversed(methods_sorted[:mid]))   # best ranks on top-left
    right_methods = methods_sorted[mid:]                  # worse ranks on top-right

    # Corresponding ranks
    rank_map = dict(zip(methods_sorted, ranks_sorted))
    left_ranks = [rank_map[m] for m in left_methods]
    right_ranks = [rank_map[m] for m in right_methods]

    # Vertical positions for labels (evenly spaced)
    # Put some top padding for clique bars
    y_start = k + 0.2
    y_step = 1.0

    left_ys = [y_start - i * y_step for i in range(len(left_methods))]
    right_ys = [y_start - i * y_step for i in range(len(right_methods))]

    # X positions for label columns
    x_left_text = 0.55
    x_right_text = k + 0.45

    # Helper: draw one side (labels + connectors)
    def draw_side(methods, ranks, ys, side: str) -> None:
        for m, r, y in zip(methods, ranks, ys):
            # Connector: from rank on axis down/up to label line
            ax.plot([r, r], [y_axis, y], linewidth=1.0)

            # Connector: from rank position to label column
            if side == "left":
                ax.plot([r, x_left_text + 0.05], [y, y], linewidth=1.0)
                ax.text(
                    x_left_text,
                    y,
                    f"{m}  ({r:.2f})",
                    ha="left",
                    va="center",
                    fontsize=10,
                )
            else:
                ax.plot([r, x_right_text - 0.05], [y, y], linewidth=1.0)
                ax.text(
                    x_right_text,
                    y,
                    f"({r:.2f})  {m}",
                    ha="right",
                    va="center",
                    fontsize=10,
                )

    draw_side(left_methods, left_ranks, left_ys, side="left")
    draw_side(right_methods, right_ranks, right_ys, side="right")

    # --- Non-significant cliques as bars near the top ---
    if np.isfinite(cd) and cd > 0:
        segments = compute_non_sig_segments(sorted_ranks=ranks_sorted, cd=cd)

        # Stack clique bars above the axis
        y0 = y_axis + 1.0
        dy = 0.22
        for s_i, s_j in segments[:20]:  # cap to avoid clutter
            x1 = ranks_sorted[s_i]
            x2 = ranks_sorted[s_j]
            ax.hlines(y0, x1, x2, linewidth=4.0)
            y0 += dy

    # --- CD scale bar (bottom-left) ---
    if np.isfinite(cd) and cd > 0:
        y_cd = -0.3
        x_cd1 = 1.0
        x_cd2 = min(k, 1.0 + cd)
        ax.hlines(y_cd, x_cd1, x_cd2, linewidth=4.0)
        ax.vlines([x_cd1, x_cd2], y_cd - 0.10, y_cd + 0.10, linewidth=2.0)
        ax.text((x_cd1 + x_cd2) / 2.0, y_cd - 0.25, f"CD = {cd:.3f}",
                ha="center", va="top", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric
    alpha = float(args.alpha)
    top_n = int(args.max_methods_plot)
    dpi = int(args.fig_dpi)

    df = pd.read_csv(csv_path)
    validate_columns(df, metric)

    best = compute_best_stage1(df, metric=metric)
    summary = aggregate_ranking(best, metric=metric)
    matrix = build_matrix(best, metric=metric)

    # Save tables
    ranking_path = out_dir / "ranking_simple_preproc_stage1.csv"
    best_path = out_dir / "best_per_dataset_simple_preproc_stage1.csv"
    matrix_path = out_dir / "matrix_best_metric_stage1.csv"

    summary.to_csv(ranking_path, index=False)
    best.to_csv(best_path, index=False)
    matrix.to_csv(matrix_path)

    # Figures
    boxplot_path = out_dir / "boxplot_best_metric_stage1.png"
    cd_path = out_dir / "cd_diagram_nemenyi_stage1.png"

    save_boxplot(best, summary, metric=metric, out_path=boxplot_path, top_n=top_n, dpi=dpi)
    save_cd_diagram(best, summary, metric=metric, out_path=cd_path, alpha=alpha, top_n=top_n, dpi=dpi)

    # Console report
    n_datasets = best["dataset"].nunique()
    n_preprocs = best["simple_preproc"].nunique()

    print("\n" + "=" * 110)
    print("Stage-1 preprocessing ranking (Option A) + Figures")
    print("=" * 110)
    print(f"Input CSV:        {csv_path}")
    print(f"Output directory: {out_dir}")
    print(f"Metric:           {metric} (lower is better)")
    print(f"Datasets seen:    {n_datasets}")
    print(f"Preprocs seen:    {n_preprocs}")
    print(f"Saved ranking →   {ranking_path}")
    print(f"Saved best map →  {best_path}")
    print(f"Saved matrix →    {matrix_path}")
    print(f"Saved boxplot →   {boxplot_path}")
    print(f"Saved CD diag →   {cd_path}")

    print("\nTop 15 preprocessings:")
    cols = ["simple_preproc", "n_datasets", "mean_rank", "mean_best", "median_best"]
    with pd.option_context("display.max_columns", 200, "display.width", 150):
        print(summary[cols].head(15).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
