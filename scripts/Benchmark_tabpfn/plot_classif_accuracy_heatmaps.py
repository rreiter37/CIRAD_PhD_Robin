#!/usr/bin/env python3
"""
Plot classification accuracy heatmaps with the SAME visual style
as TabPFN comparison heatmaps.

- Matplotlib only (no seaborn)
- Viridis colormap
- Global normalization
- Annotated values
- Red rectangle highlighting the best model per dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--metric", type=str, default="accuracy")
    p.add_argument("--keep_common_only", action="store_true")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_summary(path: Path, metric: str) -> pd.DataFrame:
    csv_path = path / "summary_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    required = {"dataset_type", "dataset_name", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df["dataset_id"] = df["dataset_type"] + "__" + df["dataset_name"]
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if len(args.results_dirs) != len(args.labels):
        raise ValueError("results_dirs and labels must have the same length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric = args.metric

    best_per_ws = []
    dataset_sets = []

    # -----------------------------------------------------------------
    # Load & reduce each workspace (best model per dataset)
    # -----------------------------------------------------------------
    for ws_path, label in zip(args.results_dirs, args.labels):
        df = load_summary(Path(ws_path), metric)

        df_best = (
            df.sort_values(metric, ascending=False)
              .groupby("dataset_id", as_index=False)
              .first()
        )
        df_best["workspace"] = label

        best_per_ws.append(df_best)
        dataset_sets.append(set(df_best["dataset_id"]))

    # -----------------------------------------------------------------
    # Dataset intersection / union
    # -----------------------------------------------------------------
    if args.keep_common_only:
        datasets = set.intersection(*dataset_sets)
    else:
        datasets = set.union(*dataset_sets)

    if not datasets:
        raise RuntimeError("No datasets left after filtering")

    # -----------------------------------------------------------------
    # Build pivot table
    # -----------------------------------------------------------------
    df_all = pd.concat(best_per_ws, axis=0)
    df_all = df_all[df_all["dataset_id"].isin(datasets)]

    pivot = df_all.pivot(
        index="dataset_id",
        columns="workspace",
        values=metric,
    )

    # Sort datasets by mean performance (TabPFN style)
    pivot["__mean__"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__mean__", ascending=False)
    pivot = pivot.drop(columns="__mean__")

    values = pivot.values
    row_labels = pivot.index.tolist()
    col_labels = pivot.columns.tolist()

    # -----------------------------------------------------------------
    # Figure sizing (TabPFN-like)
    # -----------------------------------------------------------------
    fig_w = max(8, 0.7 * len(col_labels))
    fig_h = max(6, 0.30 * len(row_labels))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # -----------------------------------------------------------------
    # Color normalization (GLOBAL, like TabPFN)
    # -----------------------------------------------------------------
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(values, aspect="auto", cmap="viridis", norm=norm)

    # -----------------------------------------------------------------
    # Axes & labels
    # -----------------------------------------------------------------
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))

    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title(f"Classification heatmap ({metric})", pad=12)

    # -----------------------------------------------------------------
    # Annotate cells (TabPFN style)
    # -----------------------------------------------------------------
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if norm(val) < 0.7 else "white",
                )

    # -----------------------------------------------------------------
    # Highlight best model per dataset (RED RECTANGLE)
    # -----------------------------------------------------------------
    for i in range(values.shape[0]):
        row = values[i]
        if np.all(np.isnan(row)):
            continue
        j_best = int(np.nanargmax(row))
        ax.add_patch(
            Rectangle(
                (j_best - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=2.5,
            )
        )

    # -----------------------------------------------------------------
    # Colorbar (thin, clean)
    # -----------------------------------------------------------------
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(metric)

    fig.tight_layout()

    out_png = output_dir / f"heatmap_{metric}_tabpfn_style.png"
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)

    print(f"✅ Saved heatmap → {out_png}")


if __name__ == "__main__":
    main()
