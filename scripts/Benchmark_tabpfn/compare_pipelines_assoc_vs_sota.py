#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare results from:
  1) association_pp_model pipeline (Results/assoc_pp_model/per_dataset/...)
  2) predictions pipeline (workspace/*.meta.parquet)

Outputs:
  - Per-dataset heatmaps comparing RRMSE for models:
        CNN, Ridge, PLS, LGBM, Transformer, Autogluon
    between the two pipelines.

  - A global heatmap of average relative performance (RRMSE normalized
    per dataset with respect to the best model across both pipelines)
    to "remove" dataset difficulty effect.

  - Candlestick-style plots showing the distribution of RRMSE across
    datasets for each (pipeline, model) pair.

Assumptions:
  - assoc_pp_model "results*dynamic*.csv" store a normalized RMSE
    consistent with _compute_regression_metric in evaluation.py:
      rmse / (max(y) - min(y))  (i.e., RRMSE). 
  - predictions meta parquet files are compatible with nirs4all.data.Predictions,
    and their underlying DataFrame (polars) has at least:
      - columns: "partition", "model_name", "model_classname"
      - metric column "nrmse" (preferred) or "rmse" as fallback. :contentReference[oaicite:1]{index=1}
  - Workspace files are named like "<dataset_name>.meta.parquet"
  - assoc_pp_model per-dataset results are in folders:
      Results/assoc_pp_model/per_dataset/<data_source>/
    with filenames containing "results" and "dynamic". :contentReference[oaicite:2]{index=2}
  - Difficulty ranking JSON is structured like run_baseline_by_difficulty.py:
      {"rankings": {"best_rrmse": [...], "mean_rrmse": [...], ...}} :contentReference[oaicite:3]{index=3}
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from nirs4all.data.predictions import Predictions  # used to load .meta.parquet


# =====================================================================
# CONFIG / CONSTANTS
# =====================================================================

DEFAULT_ASSOC_ROOT = Path("Results/assoc_pp_model/per_dataset")
DEFAULT_WORKSPACE = Path("workspace")
DEFAULT_OUTPUT = Path("Figures/compare_pipelines")

# Same default JSON as run_baseline_by_difficulty.py :contentReference[oaicite:4]{index=4}
DEFAULT_DIFFICULTY_JSON = (
    Path("/home/robinr/Desktop/VSCode/CIRAD_PhD_Robin/")
    / "Results/assoc_pp_model/All_datasets/Rank_datasets_difficulty/dataset_difficulty_ranking.json"
)

DEFAULT_MODELS = ["CNN", "Ridge", "PLS", "LGBM", "Transformer", "Autogluon"]


# =====================================================================
# UTILS: NAME NORMALIZATION / MODEL FAMILY
# =====================================================================

def normalize_dataset_name(name: str) -> str:
    """Normalize dataset name to match assoc_pp_model and predictions naming.

    Strategy:
      - lowercase
      - remove all non-alphanumeric characters
    """
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def map_assoc_model_to_family(model_name: str) -> str | None:
    """Map assoc_pp_model model names to generic families.

    Typical names (Regression): Ridge_reg, PLS_reg, LGBM_reg, CNN_reg. :contentReference[oaicite:5]{index=5}
    """
    name_lower = model_name.lower()
    if "ridge" in name_lower:
        return "Ridge"
    if "pls" in name_lower:
        return "PLS"
    if "lgbm" in name_lower:
        return "LGBM"
    if "cnn" in name_lower or "nicon" in name_lower:
        return "CNN"
    # No Transformer / Autogluon in assoc_pp_model → return None for others
    return None


def map_pred_model_to_family(model_name: str | None, model_class: str | None) -> str | None:
    """Map predictions model_classname/model_name to generic families.

    It looks at both model_name and model_classname strings and assigns
    them to one of the target families.
    """
    base = (model_name or "") + " " + (model_class or "")
    s = base.lower()

    if "autogluon" in s:
        return "Autogluon"
    if "tabpfn" in s or "transformer" in s:
        return "Transformer"
    if "ridge" in s:
        return "Ridge"
    if "pls" in s:
        return "PLS"
    if "lgbm" in s or "lightgbm" in s:
        return "LGBM"
    if "cnn" in s or "nicon" in s:
        return "CNN"

    return None


# =====================================================================
# LOAD DIFFICULTY
# =====================================================================

def load_difficulty_order(json_path: Path, ranking_key: str) -> dict[str, int]:
    """Load dataset difficulty order from JSON.

    Returns:
        dict: normalized_dataset_name -> rank_index (0 = easiest or hardest,
              depending on how it was stored, but we only use it for sorting).
    """
    if not json_path.exists():
        print(f"[WARNING] Difficulty JSON not found: {json_path}")
        return {}

    with open(json_path, "r") as f:
        data = json.load(f)

    rankings = data.get("rankings", {})
    if ranking_key not in rankings:
        print(f"[WARNING] Ranking key '{ranking_key}' not found in JSON.")
        return {}

    ordered_list = rankings[ranking_key]

    # normalize names for robust matching
    order = {}
    for idx, name in enumerate(ordered_list):
        order[normalize_dataset_name(name)] = idx

    return order


# =====================================================================
# LOAD ASSOC_PPMODEL RRMSE (PER DATASET / MODEL)
# =====================================================================

def load_assoc_results(assoc_root: Path) -> dict[str, dict[str, float]]:
    """Load assoc_pp_model per-dataset best RRMSE per model family.

    For each dataset folder:
      - find CSV files whose name contains "results" AND "dynamic"
      - load the pivot matrix (models x preprocessings)
      - best RRMSE per model = min across preprocessings

    Returns:
        dict:
          normalized_dataset_name -> {model_family -> rrmse}
    """
    results: dict[str, dict[str, float]] = {}

    if not assoc_root.exists():
        print(f"[WARNING] assoc_pp_model root not found: {assoc_root}")
        return results

    for ds_dir in sorted(assoc_root.iterdir()):
        if not ds_dir.is_dir():
            continue

        # Find a CSV file containing "results" and "dynamic"
        candidates = list(ds_dir.glob("*results*dynamic*.csv"))
        if not candidates:
            continue

        csv_path = candidates[0]  # take first match
        try:
            pivot = pd.read_csv(csv_path, index_col=0)  # rows=Model, cols=Preprocessing
        except Exception as e:
            print(f"[WARNING] Failed to read CSV {csv_path}: {e}")
            continue

        # Best RRMSE per model_name = min score across preprocessings
        best_per_model = pivot.min(axis=1)  # Series index = model_name

        fam_dict: dict[str, float] = {}
        for mdl_name, rrmse in best_per_model.items():
            family = map_assoc_model_to_family(mdl_name)
            if family is None:
                continue
            # If multiple model variants map to same family, keep best
            if family not in fam_dict or rrmse < fam_dict[family]:
                fam_dict[family] = float(rrmse)

        if not fam_dict:
            continue

        norm_name = normalize_dataset_name(ds_dir.name)
        results[norm_name] = fam_dict

        print(f"[INFO] assoc_pp_model: {ds_dir.name} → families={list(fam_dict.keys())}")

    return results


# =====================================================================
# LOAD PREDICTIONS RRMSE (PER DATASET / MODEL)
# =====================================================================

def load_predictions_results(workspace: Path) -> dict[str, dict[str, float]]:
    """Load predictions meta parquet and compute best RRMSE per model family.

    Strategy:
      - For each <dataset>.meta.parquet file:
          - load via Predictions.load()
          - access underlying polars DataFrame (. _storage._df)
          - filter partition == 'test'
          - prefer metric column 'nrmse', else try to normalize 'rmse', else fallback to 'rmse'
          - group rows by model family and take min(metric) as best RRMSE.

    Returns:
        dict:
          normalized_dataset_name -> {model_family -> rrmse}
    """
    results: dict[str, dict[str, float]] = {}

    if not workspace.exists():
        print(f"[WARNING] workspace not found: {workspace}")
        return results

    for meta_path in sorted(workspace.glob("*.meta.parquet")):
        ds_raw_name = meta_path.stem.replace(".meta", "")

        try:
            preds = Predictions.load(str(meta_path))
        except Exception as e:
            print(f"[WARNING] Failed to load Predictions from {meta_path}: {e}")
            continue

        try:
            df: pl.DataFrame = preds._storage._df  # polars DataFrame (as used in predictions.py) :contentReference[oaicite:6]{index=6}
        except Exception as e:
            print(f"[WARNING] Could not access internal DataFrame for {meta_path}: {e}")
            continue

        # Restrict to test partition
        if "partition" in df.columns:
            df = df.filter(pl.col("partition") == "test")

        # Determine metric column
        metric_col = None
        if "nrmse" in df.columns:
            metric_col = "nrmse"
        elif "rrmse" in df.columns:
            metric_col = "rrmse"
        elif "rmse" in df.columns:
            metric_col = "rmse"
        else:
            print(f"[WARNING] No rmse/nrmse/rrmse column in {meta_path}; skipping.")
            continue

        # Convert to pandas for convenience
        pdf = df.to_pandas()

        fam_metrics: dict[str, list[float]] = defaultdict(list)

        for _, row in pdf.iterrows():
            family = map_pred_model_to_family(
                row.get("model_name"),
                row.get("model_classname"),
            )
            if family is None:
                continue

            val = row.get(metric_col)
            if pd.isna(val):
                continue

            fam_metrics[family].append(float(val))

        if not fam_metrics:
            continue

        fam_best: dict[str, float] = {
            fam: float(np.min(vals)) for fam, vals in fam_metrics.items()
        }

        norm_name = normalize_dataset_name(ds_raw_name)
        results[norm_name] = fam_best

        print(f"[INFO] predictions: {ds_raw_name} → families={list(fam_best.keys())}")

    return results


# =====================================================================
# BUILD COMMON PANEL
# =====================================================================

def build_common_panel(
    assoc_res: dict[str, dict[str, float]],
    pred_res: dict[str, dict[str, float]],
    models_of_interest: list[str],
) -> pd.DataFrame:
    """Build a long-format DataFrame with RRMSE for datasets common to both pipelines.

    Returns DataFrame with columns:
        dataset, pipeline, model_family, rrmse
    where:
        pipeline ∈ {"assoc", "pred"}
        model_family ∈ models_of_interest
    """
    common_datasets = sorted(set(assoc_res.keys()) & set(pred_res.keys()))
    print(f"[INFO] Number of common datasets: {len(common_datasets)}")

    rows = []
    for ds in common_datasets:
        for fam in models_of_interest:
            assoc_val = assoc_res[ds].get(fam, np.nan)
            pred_val = pred_res[ds].get(fam, np.nan)

            if not np.isnan(assoc_val):
                rows.append(
                    {
                        "dataset": ds,
                        "pipeline": "assoc",
                        "model": fam,
                        "rrmse": assoc_val,
                    }
                )
            if not np.isnan(pred_val):
                rows.append(
                    {
                        "dataset": ds,
                        "pipeline": "pred",
                        "model": fam,
                        "rrmse": pred_val,
                    }
                )

    panel = pd.DataFrame(rows)
    return panel


# =====================================================================
# HEATMAPS: PER DATASET
# =====================================================================

def plot_per_dataset_heatmaps(
    panel: pd.DataFrame,
    output_dir: Path,
    models_of_interest: list[str],
    difficulty_order: dict[str, int],
):
    """Generate per-dataset heatmaps (2 x models) for RRMSE.

    One PNG per dataset:
        y-axis: pipeline (assoc / pred)
        x-axis: models (ordered)
        values: RRMSE
    """
    out_dir = output_dir / "per_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine dataset order (using difficulty if available)
    datasets = sorted(panel["dataset"].unique())
    if difficulty_order:
        datasets.sort(key=lambda d: difficulty_order.get(d, 999999))

    sns.set_theme(style="white", font_scale=1.0)

    for ds in datasets:
        sub = panel[panel["dataset"] == ds]

        # Build matrix pipeline x model_family
        pivot = (
            sub.pivot(index="pipeline", columns="model", values="rrmse")
            .reindex(index=["assoc", "pred"])
            .reindex(columns=models_of_interest)
        )

        fig, ax = plt.subplots(figsize=(1.5 * len(models_of_interest), 3))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis_r",  # lower is better, so reverse
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"RRMSE per model – dataset: {ds}")
        ax.set_xlabel("Model family")
        ax.set_ylabel("Pipeline (assoc_pp_model vs predictions)")

        fig.tight_layout()
        fname = out_dir / f"heatmap_{ds}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)

        print(f"[INFO] Saved per-dataset heatmap → {fname}")


# =====================================================================
# GLOBAL HEATMAP (RELATIVE TO BEST PER DATASET)
# =====================================================================

def plot_global_relative_heatmap(
    panel: pd.DataFrame,
    output_dir: Path,
    models_of_interest: list[str],
):
    """Global heatmap of average relative performance (affranchi de la difficulté).

    For each dataset:
      - find best RRMSE across all (pipeline, model)
      - define relative score = rrmse / best_rrmse  (1.0 = best on that dataset)
    Then:
      - compute mean relative score over datasets
      - build heatmap with:
          y-axis: model family
          x-axis: pipeline (assoc, pred)
    """
    if panel.empty:
        print("[WARNING] Empty panel; cannot plot global heatmap.")
        return

    # Compute best RRMSE per dataset
    best_by_ds = (
        panel.groupby("dataset")["rrmse"]
        .min()
        .rename("best_rrmse")
        .to_dict()
    )

    panel_rel = panel.copy()
    panel_rel["relative"] = panel_rel.apply(
        lambda row: row["rrmse"] / best_by_ds[row["dataset"]]
        if best_by_ds[row["dataset"]] > 0
        else np.nan,
        axis=1,
    )

    # Average relative performance per (pipeline, model)
    mean_rel = (
        panel_rel.groupby(["model", "pipeline"])["relative"]
        .mean()
        .unstack("pipeline")
        .reindex(index=models_of_interest)
    )

    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="white", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(4, 0.7 * len(models_of_interest) + 2))
    sns.heatmap(
        mean_rel,
        annot=True,
        fmt=".3f",
        cmap="mako_r",
        cbar=True,
        ax=ax,
    )

    ax.set_title("Mean relative RRMSE (1.0 = best per dataset)")
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Model family")

    fig.tight_layout()
    fname = out_dir / "global_relative_heatmap.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved global relative heatmap → {fname}")


# =====================================================================
# CANDLESTICK PLOT
# =====================================================================

def plot_candlestick(
    panel: pd.DataFrame,
    output_dir: Path,
    models_of_interest: list[str],
):
    """Candlestick-style plot for RRMSE distribution across datasets.

    For each (pipeline, model) pair:
      - collect list of RRMSE over datasets
      - draw a "candlestick" (min, Q1, median, Q3, max).

    Implementation uses simple vertical lines and rectangles without
    external finance libraries.
    """
    if panel.empty:
        print("[WARNING] Empty panel; cannot plot candlesticks.")
        return

    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pipelines = ["assoc", "pred"]
    width = 0.35  # half-width of boxes

    # Prepare structure: dict[(pipeline, model)] -> list of values
    values = defaultdict(list)
    for _, row in panel.iterrows():
        key = (row["pipeline"], row["model"])
        values[key].append(row["rrmse"])

    fig, ax = plt.subplots(figsize=(1.7 * len(models_of_interest), 5))

    x_positions = []
    x_labels = []

    idx = 0
    for i, model in enumerate(models_of_interest):
        for j, pip in enumerate(pipelines):
            key = (pip, model)
            vals = values.get(key, [])
            if not vals:
                idx += 1
                continue

            vals = np.array(vals)
            vmin, vmax = np.min(vals), np.max(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            med = np.median(vals)

            x = idx
            x_positions.append(x)
            x_labels.append(f"{model}\n{pip}")

            # Whisker (min-max)
            ax.vlines(x, vmin, vmax, linewidth=1.0)

            # Box (Q1-Q3)
            rect = plt.Rectangle(
                (x - width / 2, q1),
                width,
                q3 - q1,
                fill=False,
                linewidth=1.5,
            )
            ax.add_patch(rect)

            # Median line
            ax.hlines(med, x - width / 2, x + width / 2, linewidth=1.5)

            idx += 1

        # Add small gap between model groups
        idx += 1

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("RRMSE")
    ax.set_title("RRMSE distribution across datasets (candlestick-style)")

    fig.tight_layout()
    fname = out_dir / "candlestick_rrmse.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print(f"[INFO] Saved candlestick plot → {fname}")


# =====================================================================
# MAIN
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare assoc_pp_model vs predictions pipelines."
    )

    parser.add_argument(
        "--assoc_root",
        type=str,
        default=str(DEFAULT_ASSOC_ROOT),
        help="Root folder for assoc_pp_model per-dataset results.",
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=str(DEFAULT_WORKSPACE),
        help="Workspace folder containing *.meta.parquet (predictions pipeline).",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output folder for comparison figures.",
    )

    parser.add_argument(
        "--difficulty_json",
        type=str,
        default=str(DEFAULT_DIFFICULTY_JSON),
        help="JSON file with dataset difficulty rankings.",
    )

    parser.add_argument(
        "--difficulty_ranking",
        type=str,
        default="best_rrmse",
        choices=["best_rrmse", "mean_rrmse", "mean_best_model"],
        help="Which ranking key to use in the difficulty JSON.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model families to keep (subset of: CNN, Ridge, PLS, LGBM, Transformer, Autogluon).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    assoc_root = Path(args.assoc_root)
    workspace = Path(args.workspace)
    output_dir = Path(args.output_dir)
    difficulty_json = Path(args.difficulty_json)
    models_of_interest = list(args.models)

    print("[INFO] Loading assoc_pp_model results...")
    assoc_res = load_assoc_results(assoc_root)

    print("[INFO] Loading predictions results...")
    pred_res = load_predictions_results(workspace)

    # Difficulty order for optional sorting
    print("[INFO] Loading difficulty ranking...")
    difficulty_order = load_difficulty_order(difficulty_json, args.difficulty_ranking)

    print("[INFO] Building common panel...")
    panel = build_common_panel(assoc_res, pred_res, models_of_interest)

    if panel.empty:
        print("[WARNING] No common data between the two pipelines with current settings.")
        return

    # Per-dataset heatmaps
    print("[INFO] Generating per-dataset heatmaps...")
    plot_per_dataset_heatmaps(panel, output_dir, models_of_interest, difficulty_order)

    # Global heatmap (relative to best per dataset)
    print("[INFO] Generating global relative heatmap...")
    plot_global_relative_heatmap(panel, output_dir, models_of_interest)

    # Candlestick plots
    print("[INFO] Generating candlestick plot...")
    plot_candlestick(panel, output_dir, models_of_interest)

    print("[INFO] Comparison script completed successfully.")


if __name__ == "__main__":
    main()
