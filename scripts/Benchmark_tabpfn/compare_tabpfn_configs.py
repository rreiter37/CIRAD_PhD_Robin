#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic comparison script for multiple TabPFN workspaces.

This script generalizes the logic of analyze_tabpfn_rff_results.py to:
  - Compare >= 2 workspaces coming from slightly different configurations
  - Ensure comparison is performed ONLY on the intersection of datasets
  - Reuse the same metrics logic (val_mean, val_best, test)
  - Produce identical outputs (CSV + figures) regardless of configuration

All comments are in English by design.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ===================== CLI ===================== #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--workspaces", nargs="+", required=True,
                        help="List of workspace directories to compare")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Human-readable labels (same length as workspaces)")
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--output_dir", type=str, default="comparison_multi_ws")

    return parser.parse_args()


# ===================== Utilities ===================== #

def extract_metric(scores_json, partition, metric):
    scores = json.loads(scores_json)
    return scores[partition][metric]

def is_higher_better(metric: str) -> bool:
    """
    Returns True if higher metric values indicate better performance.
    """
    return metric.lower() in ["r2", "accuracy", "balanced_accuracy", "f1", "auc"]

def load_workspace(workspace: Path, label: str, metric: str) -> pd.DataFrame:
    """
    Load one workspace and extract val_mean / val_best / test metrics
    from *.meta.parquet files.
    """
    rows = []

    for meta_path in workspace.rglob("*.meta.parquet"):
        df = pd.read_parquet(meta_path)
        dataset = df["dataset_name"].iloc[0]

        # Only CV folds
        fold_df = df[df["fold_id"].isin(["0", "1", "2"])].copy()

        for part in ["train", "val", "test"]:
            fold_df[f"{part}_{metric}"] = fold_df["scores"].apply(
                lambda s: extract_metric(s, part, metric)
            )

        # ---- Validation metrics ----
        val_df = fold_df[fold_df["partition"] == "val"]
        if not val_df.empty:
            mean_vals = val_df.groupby("model_name")[f"val_{metric}"].mean()
            best_vals = (
                val_df.groupby("model_name")[f"val_{metric}"].max()
                if metric == "r2" else
                val_df.groupby("model_name")[f"val_{metric}"].min()
            )

            for m in mean_vals.index:
                rows.append(dict(dataset=dataset, model=label,
                                 split="val_mean", metric=metric,
                                 value=mean_vals[m]))
                rows.append(dict(dataset=dataset, model=label,
                                 split="val_best", metric=metric,
                                 value=best_vals[m]))

        # ---- Test metrics ----
        test_df = df[(df["partition"] == "test") &
                     (df["fold_id"].isin(["avg", "w_avg"]))].copy()

        if test_df.empty:
            test_df = fold_df[fold_df["partition"] == "test"].copy()

        test_df.loc[:, f"test_{metric}"] = test_df["scores"].apply(
            lambda s: extract_metric(s, "test", metric)
        )

        test_vals = test_df.groupby("model_name")[f"test_{metric}"].mean()

        for m in test_vals.index:
            rows.append(dict(dataset=dataset, model=label,
                             split="test", metric=metric,
                             value=test_vals[m]))

    return pd.DataFrame(rows)


# ===================== Figures ===================== #

def save_global_bar(df, split, outdir):
    plt.figure(figsize=(10, 4))
    sub = df[df["split"] == split]
    means = sub.groupby("model")["value"].mean()
    means.plot(kind="bar")
    plt.title(f"Global mean {split}")
    plt.ylabel(sub["metric"].iloc[0])
    plt.tight_layout()
    plt.savefig(outdir / f"global_bar_{split}.png")
    plt.close()


def save_boxplot(df, split, outdir):
    plt.figure(figsize=(10, 4))
    sub = df[df["split"] == split]
    data = [sub[sub["model"] == m]["value"] for m in sub["model"].unique()]
    plt.boxplot(data, tick_labels=sub["model"].unique())
    plt.title(f"Distribution of {split}")
    plt.tight_layout()
    plt.savefig(outdir / f"boxplot_{split}.png")
    plt.close()


def save_heatmap(df, split, outdir):
    """
    Heatmap with numeric values inside each cell.
    Best value per dataset is highlighted with a red rectangle.
    """
    pivot = df[df["split"] == split].pivot_table(
        index="model", columns="dataset", values="value"
    )

    n_models, n_datasets = pivot.shape
    fig_width = max(12, 0.8 * n_datasets)
    fig_height = max(4, 0.6 * n_models)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    ax.set_xticks(np.arange(n_datasets))
    ax.set_yticks(np.arange(n_models))

    ax.set_xticklabels(pivot.columns, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=10)

    ax.set_title(f"Heatmap {split}")

    # ---- Write values inside cells ----
    for i in range(n_models):
        for j in range(n_datasets):
            val = pivot.iloc[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j, i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    # ---- Highlight best per dataset ----
    higher_is_better = is_higher_better(df["metric"].iloc[0])

    for j in range(n_datasets):
        col = pivot.iloc[:, j]
        if col.isna().all():
            continue

        best_i = np.nanargmax(col.values) if higher_is_better else np.nanargmin(col.values)

        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, best_i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=2.5
            )
        )

    ax.set_xlim(-0.5, n_datasets - 0.5)
    ax.set_ylim(n_models - 0.5, -0.5)

    plt.subplots_adjust(bottom=0.35)
    plt.savefig(outdir / f"heatmap_{split}.png", dpi=300)
    plt.close()


def save_binary_heatmap(df, split, outdir):
    """
    Heatmap-style visualization with:
      - Green = best model per dataset
      - Red = other models
      - Numeric metric values written in each cell
      - Dataset labels displayed like standard heatmaps (rotated, readable)
    """
    pivot = df[df["split"] == split].pivot_table(
        index="model", columns="dataset", values="value"
    )

    n_models, n_datasets = pivot.shape

    higher_is_better = is_higher_better(df["metric"].iloc[0])

    # ---- Build binary mask: 1 = best, 0 = others ----
    binary = np.zeros_like(pivot.values, dtype=int)

    for j in range(n_datasets):
        col = pivot.iloc[:, j]
        if col.isna().all():
            continue
        best_i = np.nanargmax(col.values) if higher_is_better else np.nanargmin(col.values)
        binary[best_i, j] = 1

    # ---- Figure size ----
    fig_width = max(12, 0.8 * n_datasets)
    fig_height = max(4, 0.6 * n_models)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # ---- Two-color heatmap ----
    cmap = plt.matplotlib.colors.ListedColormap(["#f5b7b1", "#b6f2c2"])
    ax.imshow(binary, aspect="auto", interpolation="nearest", cmap=cmap)

    # ---- Axis ticks & labels (same as heatmaps) ----
    ax.set_xticks(np.arange(n_datasets))
    ax.set_yticks(np.arange(n_models))

    ax.set_xticklabels(
        pivot.columns,
        rotation=70,
        ha="right",
        fontsize=8
    )
    ax.set_yticklabels(pivot.index, fontsize=10)

    ax.set_title(f"Best model per dataset – {split}")

    # ---- Write metric values inside cells ----
    for i in range(n_models):
        for j in range(n_datasets):
            val = pivot.iloc[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j, i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    # ---- Force bounds (important for alignment) ----
    ax.set_xlim(-0.5, n_datasets - 0.5)
    ax.set_ylim(n_models - 0.5, -0.5)

    plt.subplots_adjust(bottom=0.35)
    plt.savefig(outdir / f"binary_heatmap_{split}.png", dpi=300)
    plt.close()


def save_per_dataset_figures(df, split, outdir):
    base = outdir / "per_dataset"
    base.mkdir(exist_ok=True)

    for dataset in df["dataset"].unique():
        ddir = base / dataset
        ddir.mkdir(exist_ok=True)

        sub = df[(df["dataset"] == dataset) & (df["split"] == split)]
        if sub.empty:
            continue

        plt.figure(figsize=(6, 4))
        means = sub.groupby("model")["value"].mean()
        means.plot(kind="bar")
        plt.title(f"{dataset} – {split}")
        plt.ylabel(sub["metric"].iloc[0])
        plt.tight_layout()
        plt.savefig(ddir / f"bar_{split}.png")
        plt.close()


# ===================== Main ===================== #

def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load all workspaces ----
    dfs = []
    for ws, label in zip(args.workspaces, args.labels):
        dfs.append(load_workspace(Path(ws), label, args.metric))

    df_all = pd.concat(dfs, ignore_index=True)

    # ---- Keep only dataset intersection ----
    datasets_by_model = df_all.groupby("model")["dataset"].unique()
    common_datasets = set(datasets_by_model.iloc[0])
    for dsets in datasets_by_model.iloc[1:]:
        common_datasets &= set(dsets)

    df_all = df_all[df_all["dataset"].isin(common_datasets)]

    # ---- Aggregate if needed ----
    df_all = df_all.groupby(
        ["dataset", "model", "split", "metric"],
        as_index=False
    ).agg(value=("value", "mean"))

    # ---- Save CSV ----
    csv_path = outdir / f"comparison_{args.metric}.csv"
    df_all.to_csv(csv_path, index=False)

    # ---- Figures ----
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    for split in ["test", "val_mean", "val_best"]:
        if split not in df_all["split"].unique():
            continue
        save_global_bar(df_all, split, figdir)
        save_boxplot(df_all, split, figdir)
        save_heatmap(df_all, split, figdir)
        save_binary_heatmap(df_all, split, figdir)
        save_per_dataset_figures(df_all, split, figdir)


    # ---- Simple global stats ----
    stats = []
    for split in df_all["split"].unique():
        sub = df_all[df_all["split"] == split]
        models = sub["model"].unique()
        if len(models) == 2:
            m1, m2 = models
            v1 = sub[sub["model"] == m1].sort_values("dataset")["value"].values
            v2 = sub[sub["model"] == m2].sort_values("dataset")["value"].values
            stat, p = wilcoxon(v1, v2)
            stats.append(dict(split=split, model_1=m1, model_2=m2,
                              wilcoxon_p=p,
                              median_diff=np.median(v1 - v2)))

    if stats:
        pd.DataFrame(stats).to_csv(outdir / "stats.csv", index=False)

    print(f"Comparison finished. Results saved in {outdir}")


if __name__ == "__main__":
    main()
