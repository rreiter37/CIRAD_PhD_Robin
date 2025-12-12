#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction analysis script — FINAL CLEAN VERSION (Notebook-equivalent)

Key points:
- Uses Arrow/Parquet loader (manual_load_predictions) identical to notebook logic
- Does NOT rely on Predictions.load() (incompatible with .arrays.parquet)
- Works for all datasets in workspace
- Produces same figures as notebook
"""

import os
import re
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq

from nirs4all.data import Predictions
from nirs4all.data.predictions import PredictionStorage
from nirs4all.visualization.predictions import PredictionAnalyzer


# =============================================================================
# CONFIG
# =============================================================================

WORKSPACE_PATH = "/home/robinr/Desktop/VSCode/CIRAD_PhD_Robin/workspace"
CHARTS_OUTPUT_DIR = "/home/robinr/Desktop/VSCode/CIRAD_PhD_Robin/charts"

FILENAMES = sorted([
    f.stem.replace(".meta", "")
    for f in Path(WORKSPACE_PATH).glob("*.meta.parquet")
])

EXCLUDE_MODELS = ["KernelPLS", "Sequential", "NICON"]
MODEL_RENAME_MAP = {
    "tabpfn": "transformer",
    "tabpfn-real": "transformer-2",
}

AGGREGATION_KEY = "ID"


# =============================================================================
# 1) NOTEBOOK-COMPATIBLE PREDICTION LOADER
# =============================================================================
def manual_load_predictions(meta_path: str):
    """
    Load predictions EXACTLY like the notebook:
    Using nirs4all's official split-parquet loader.
    """
    meta_path = Path(meta_path)
    dataset_name = meta_path.stem.replace(".meta", "")
    arrays_path = meta_path.parent / f"{dataset_name}.arrays.parquet"

    if not arrays_path.exists():
        raise FileNotFoundError(
            f"Arrays parquet missing:\n  {arrays_path}\n"
            f"Existing arrays:\n" +
            "\n".join(str(p) for p in meta_path.parent.glob("*.arrays.parquet"))
        )

    # THIS is the correct loader for your version of nirs4all
    predictions = Predictions.load_from_parquet(meta_path, arrays_path)

    return predictions

# =============================================================================
# 2) MODEL FILTERS (unchanged from your notebook)
# =============================================================================

def apply_model_filters(predictions, exclude_models, rename_map, strip_suffixes=None):

    if strip_suffixes is None:
        strip_suffixes = ["classifier", "regressor"]

    df = predictions._storage._df
    original_count = len(df)

    # Exclude models
    if exclude_models:
        pattern = "(?i)(" + "|".join(re.escape(x) for x in exclude_models) + ")"
        df = df.filter(~pl.col("model_classname").str.contains(pattern))
        df = df.filter(~pl.col("model_name").str.contains(pattern))

    # Strip suffixes
    suffix_pattern = "(?i)(" + "|".join(strip_suffixes) + ")$"
    df = df.with_columns([
        pl.col("model_classname").str.replace(suffix_pattern, "").alias("model_classname"),
        pl.col("model_name").str.replace(suffix_pattern, "").alias("model_name"),
    ])

    # Rename patterns
    for old, new in rename_map.items():
        pattern = f"(?i){re.escape(old)}"
        df = df.with_columns([
            pl.when(pl.col("model_classname").str.contains(pattern)).then(new).otherwise(pl.col("model_classname")).alias("model_classname"),
            pl.when(pl.col("model_name").str.contains(pattern)).then(new).otherwise(pl.col("model_name")).alias("model_name"),
        ])

    predictions._storage._df = df
    return predictions


# =============================================================================
# 3) SAVE FIGURES
# =============================================================================

def save_figure(fig, dataset_name: str, name: str):
    outdir = Path(CHARTS_OUTPUT_DIR) / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)

    path = outdir / f"{dataset_name}_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  📁 Saved: {path}")


# =============================================================================
# 4) MAIN PROCESSING OF A DATASET
# =============================================================================

def process_dataset(dataset_name: str):

    meta_path = Path(WORKSPACE_PATH) / f"{dataset_name}.meta.parquet"

    print(f"\n=== DATASET: {dataset_name} ===")
    print(f"Loading {meta_path}")

    # --- Load predictions (Arrow)
    predictions = manual_load_predictions(meta_path)

    print(f"   Loaded {len(predictions)} predictions")
    if len(predictions) == 0:
        print("   ⚠️ No predictions found, skipping.")
        return

    # --- Filter models
    predictions = apply_model_filters(predictions, EXCLUDE_MODELS, MODEL_RENAME_MAP)
    print(f"   → {len(predictions)} predictions after filtering")

    # --- Determine task type
    analyzer = PredictionAnalyzer(predictions, output_dir=None)

    task_types = predictions.get_unique_values("task_type")
    is_classification = any("classification" in str(t).lower() for t in task_types)
    rank_metric = "balanced_accuracy" if is_classification else "rmse"

    # --- Top models
    print("\n📊 Top 5 models:")
    for m in predictions.top(5, rank_metric=rank_metric, rank_partition="val", aggregate="ID"):
        print("   ", m.get("model_name"), m.get("rank_score"))

    print("\n📊 Generating charts…")

    # === Confusion matrices (classification only)
    if is_classification:
        def cm(part, agg=None):
            fig = analyzer.plot_confusion_matrix(
                rank_metric=rank_metric,
                display_metric=rank_metric,
                display_partition="test",
                rank_partition=part,
                aggregate=agg
            )
            name = f"confusion_{part}" + (f"_agg_{AGGREGATION_KEY}" if agg else "")
            save_figure(fig, dataset_name, name)

        cm("val"); cm("test")
        cm("val", AGGREGATION_KEY)
        cm("test", AGGREGATION_KEY)

    # === Top-K (regression)
    if not is_classification:
        def tk(part, agg=None):
            fig = analyzer.plot_top_k(
                k=3,
                rank_metric=rank_metric,
                rank_partition=part,
                display_partition="test",
                aggregate=agg
            )
            name = f"topk_{part}" + (f"_agg_{AGGREGATION_KEY}" if agg else "")
            save_figure(fig, dataset_name, name)

        tk("val"); tk("test")
        tk("val", AGGREGATION_KEY); tk("test", AGGREGATION_KEY)

    # === Heatmaps
    def heat(part, agg=None, top_k=20):
        fig = analyzer.plot_heatmap(
            x_var="partition",
            y_var="model_name",
            rank_metric=rank_metric,
            display_metric=rank_metric,
            rank_partition=part,
            aggregate=agg,
            top_k=top_k,
            column_scale=True,
            sort_by="value",
            config={"annotation_fontsize": 18}
        )
        name = f"heatmap_{rank_metric}_{part}" + (f"_agg_{AGGREGATION_KEY}" if agg else "")
        save_figure(fig, dataset_name, name)

    heat("val", top_k=20)
    heat("test", top_k=8)
    heat("val", AGGREGATION_KEY, 20)
    heat("test", AGGREGATION_KEY, 8)

    # === Candlestick
    fig = analyzer.plot_candlestick(
        variable="model_name",
        display_metric=rank_metric,
        display_partition="test",
    )
    save_figure(fig, dataset_name, f"candlestick_{rank_metric}")

    # === Histogram
    fig = analyzer.plot_histogram(
        display_metric=rank_metric,
        display_partition="test",
    )
    save_figure(fig, dataset_name, f"histogram_{rank_metric}")

    print(f"✔ Finished dataset: {dataset_name}")


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    print(f"🔧 Workspace = {WORKSPACE_PATH}")
    print(f"🔎 Found {len(FILENAMES)} datasets")

    for ds in FILENAMES:
        process_dataset(ds)

    print("\n🎉 All datasets processed!")


if __name__ == "__main__":
    main()
