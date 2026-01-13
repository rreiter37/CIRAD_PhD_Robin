#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full Parquet-based analysis of TabPFN Raw vs RFF results.

This script:
  - Reads *.meta.parquet files only
  - Reconstructs metrics from JSON scores
  - Computes validation mean / validation best / test scores
  - Saves a global comparison CSV
  - Generates multiple figures (global + per-dataset)
  - Performs paired statistical comparisons (Wilcoxon, wins, ranks)

All figures are saved next to the CSV outputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata


# ===================== CLI ===================== #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--workspaces", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--output_dir", type=str, default="comparison_parquet")

    return parser.parse_args()


# ===================== Core utilities ===================== #

def extract_metric(scores_json, partition, metric):
    scores = json.loads(scores_json)
    return scores[partition][metric]


def load_workspace(workspace: Path, label: str, metric: str) -> pd.DataFrame:
    rows = []

    for meta_path in workspace.rglob("*.meta.parquet"):
        df = pd.read_parquet(meta_path)
        dataset = df["dataset_name"].iloc[0]

        fold_df = df[df["fold_id"].isin(["0", "1", "2"])].copy()

        for part in ["train", "val", "test"]:
            fold_df[f"{part}_{metric}"] = fold_df["scores"].apply(
                lambda s: extract_metric(s, part, metric)
            )

        # -------- Validation --------
        val_df = fold_df[fold_df["partition"] == "val"]
        if not val_df.empty:
            mean_vals = val_df.groupby("model_name")[f"val_{metric}"].mean()
            if metric == "r2":
                best_vals = val_df.groupby("model_name")[f"val_{metric}"].max()
            else:
                best_vals = val_df.groupby("model_name")[f"val_{metric}"].min()

            for m in mean_vals.index:
                rows.append(dict(dataset=dataset, model=label,
                                 base_model=m, split="val_mean",
                                 metric=metric, value=mean_vals[m]))
                rows.append(dict(dataset=dataset, model=label,
                                 base_model=m, split="val_best",
                                 metric=metric, value=best_vals[m]))

        # -------- Test --------
        test_df = df[(df["partition"] == "test") &
                     (df["fold_id"].isin(["avg", "w_avg"]))]

        if test_df.empty:
            test_df = fold_df[fold_df["partition"] == "test"]

        test_df[f"test_{metric}"] = test_df["scores"].apply(
            lambda s: extract_metric(s, "test", metric)
        )

        test_vals = test_df.groupby("model_name")[f"test_{metric}"].mean()

        for m in test_vals.index:
            rows.append(dict(dataset=dataset, model=label,
                             base_model=m, split="test",
                             metric=metric, value=test_vals[m]))

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
    plt.boxplot(data, labels=sub["model"].unique())
    plt.title(f"Distribution of {split}")
    plt.tight_layout()
    plt.savefig(outdir / f"boxplot_{split}.png")
    plt.close()


def save_heatmap(df, split, outdir):
    pivot = df[df["split"] == split].pivot_table(
        index="model", columns="dataset", values="value"
    )
    plt.figure(figsize=(1.2 * pivot.shape[1], 3))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar()
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
    plt.title(f"Heatmap {split}")
    plt.tight_layout()
    plt.savefig(outdir / f"heatmap_{split}.png")
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


# ===================== Statistics ===================== #

def statistical_comparison(df, split):
    results = []

    sub = df[df["split"] == split]
    models = sub["model"].unique()
    datasets = sub["dataset"].unique()

    if len(models) != 2:
        return pd.DataFrame()

    m1, m2 = models

    vals1, vals2 = [], []

    for d in datasets:
        v1 = sub[(sub["dataset"] == d) & (sub["model"] == m1)]["value"]
        v2 = sub[(sub["dataset"] == d) & (sub["model"] == m2)]["value"]
        if not v1.empty and not v2.empty:
            vals1.append(v1.values[0])
            vals2.append(v2.values[0])

    vals1, vals2 = np.array(vals1), np.array(vals2)

    stat, p = wilcoxon(vals1, vals2)
    wins_1 = np.sum(vals1 < vals2)
    wins_2 = np.sum(vals2 < vals1)

    ranks = np.vstack([vals1, vals2])
    mean_ranks = rankdata(ranks, axis=0).mean(axis=1)

    results.append(dict(
        split=split,
        model_1=m1,
        model_2=m2,
        wilcoxon_p=p,
        wins_model_1=wins_1,
        wins_model_2=wins_2,
        median_diff=np.median(vals1 - vals2),
        mean_rank_model_1=mean_ranks[0],
        mean_rank_model_2=mean_ranks[1],
    ))

    return pd.DataFrame(results)


# ===================== Main ===================== #

def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for ws, label in zip(args.workspaces, args.labels):
        dfs.append(load_workspace(Path(ws), label, args.metric))

    df_all = pd.concat(dfs, ignore_index=True)

    df_all = df_all.groupby(
        ["dataset", "model", "base_model", "split", "metric"],
        as_index=False
    ).agg(value=("value", "mean"))

    # Save CSV
    csv_path = outdir / f"tabpfn_rff_comparison_{args.metric}.csv"
    df_all.to_csv(csv_path, index=False)

    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    stats = []

    for split in ["test", "val_mean", "val_best"]:
        if split not in df_all["split"].unique():
            continue

        save_global_bar(df_all, split, figdir)
        save_boxplot(df_all, split, figdir)
        save_heatmap(df_all, split, figdir)
        save_per_dataset_figures(df_all, split, figdir)

        stat_df = statistical_comparison(df_all, split)
        if not stat_df.empty:
            stats.append(stat_df)

    if stats:
        stat_df = pd.concat(stats, ignore_index=True)
        stat_df.to_csv(outdir / f"stats_comparison_{args.metric}.csv", index=False)


if __name__ == "__main__":
    main()
