#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ranking NIRS datasets by difficulty based on association results.

Updates:
  - Optional argument --only_model that allows selecting one or more model
    families among: PLS, LGBM, CNN, Ridge.
  - Difficulty computation is then restricted to the selected models only.
  - Output files (.json, .png) include suffixes listing the selected models.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "Results/assoc_pp_model/per_dataset"
DATASET_BASE_DIR = "Data/Regression"
FORBIDDEN = ["LGBM", "PLS", "Ridge", "CNN", "NICON"]

OUTPUT_BASE_JSON = (
    "Results/assoc_pp_model/All_datasets/Rank_datasets_difficulty/"
)

FIGURE_DIR = "Figures/assoc_pp_model/All_datasets/Ranking_datasets_difficulty"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only_model",
        nargs="+",
        type=str,
        default=None,
        choices=["PLS", "LGBM", "CNN", "Ridge"],
        help="Restrict difficulty comparison to selected model families."
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Dataset scanning
# ----------------------------------------------------------------------
def find_valid_result_files():
    """Return a dict: dataset_name -> result_csv_path."""
    dataset_files = {}

    for dataset in os.listdir(BASE_DIR):
        ds_dir = os.path.join(BASE_DIR, dataset)
        if not os.path.isdir(ds_dir):
            continue

        for f in os.listdir(ds_dir):
            if (
                f.endswith(".csv")
                and f.startswith("results_")
                and "dynamic_batch_size" in f
                and not any(x in f for x in FORBIDDEN)
            ):
                dataset_files[dataset] = os.path.join(ds_dir, f)

    return dataset_files


# ----------------------------------------------------------------------
# Model filtering helper
# ----------------------------------------------------------------------
def filter_models_in_pivot(pivot, selected_models):
    """
    Keep only rows whose model names contain one of the selected model keys.
    Example: selected_models = ["PLS", "CNN"]
        -> keep rows where index contains "PLS" or "CNN"
    """
    if selected_models is None:
        return pivot  # no filtering

    mask = pivot.index.str.contains("|".join(selected_models))
    filtered = pivot.loc[mask]

    if filtered.empty:
        return None  # This dataset has no relevant models

    return filtered


# ----------------------------------------------------------------------
# Difficulty metrics
# ----------------------------------------------------------------------
def compute_difficulty_scores(pivot):
    """Compute three difficulty metrics from a score pivot matrix."""
    pivot = pivot.replace([np.inf, -np.inf], np.nan)

    best_rrmse = pivot.min().min()
    mean_rrmse = pivot.mean().mean()
    mean_best_model = pivot.min(axis=1).mean()

    return {
        "best_rrmse": float(best_rrmse),
        "mean_rrmse": float(mean_rrmse),
        "mean_best_model": float(mean_best_model),
    }


def build_rankings(scores_dict):
    """Build ranking lists for each difficulty metric and dataset size."""
    rankings = {}

    # Difficulty-based rankings (higher = harder)
    for metric in ["best_rrmse", "mean_rrmse", "mean_best_model"]:
        sorted_items = sorted(
            scores_dict.items(),
            key=lambda x: x[1][metric],
            reverse=True
        )
        rankings[metric] = [ds for ds, _ in sorted_items]

    # Dataset size ranking (smaller first to control computation time)
    size_sorted = sorted(
        scores_dict.items(),
        key=lambda x: x[1]["dataset_size"]
    )
    rankings["dataset_size_ascending"] = [ds for ds, _ in size_sorted]

    return rankings



# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_one_ranking(metric_name, rankings, scores_dict, suffix):
    """Create a separate barplot for a given ranking metric."""
    ordered = rankings[metric_name]
    values = [scores_dict[ds][metric_name] for ds in ordered]

    plt.figure(figsize=(12, max(6, len(ordered) * 0.4)))
    plt.barh(ordered, values)
    plt.xlabel(f"{metric_name} (higher = harder)")
    plt.title(f"Dataset Difficulty Ranking — {metric_name} ({suffix})")
    plt.gca().invert_yaxis()

    ensure_dir(FIGURE_DIR)
    out_path = os.path.join(FIGURE_DIR, f"ranking_{metric_name}__{suffix}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[INFO] Saved figure → {out_path}")


def plot_dataset_size_ranking(rankings, scores_dict, suffix):
    """Create a barplot ranking datasets by size (ascending)."""
    ordered = rankings["dataset_size_ascending"]
    values = [scores_dict[ds]["dataset_size"] for ds in ordered]

    plt.figure(figsize=(12, max(6, len(ordered) * 0.4)))
    plt.barh(ordered, values)
    plt.xlabel("Dataset size (number of evaluated models)")
    plt.title(f"Dataset Size Ranking (ascending) ({suffix})")
    plt.gca().invert_yaxis()

    ensure_dir(FIGURE_DIR)
    out_path = os.path.join(
        FIGURE_DIR, f"ranking_dataset_size_ascending__{suffix}.png"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    print(f"[INFO] Saved figure → {out_path}")

def get_training_dataset_size(dataset_name):
    """
    Return the number of training samples based on Xcal.csv.

    Expected path:
        Data/Regression/{dataset_name}/Xcal.csv
    """
    xcal_path = os.path.join(
        DATASET_BASE_DIR,
        dataset_name,
        "Xcal.csv"
    )

    if not os.path.exists(xcal_path):
        raise FileNotFoundError(
            f"Training file not found: {xcal_path}"
        )

    # Read only the number of rows (fast and memory-safe)
    df = pd.read_csv(xcal_path)
    return df.shape[0]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    selected_models = args.only_model

    # Build suffix for file outputs
    suffix = "ALL" if not selected_models else "_".join(selected_models)

    dataset_files = find_valid_result_files()
    if not dataset_files:
        print("[ERROR] No valid result files found.")
        return

    scores_dict = {}

    for dataset, path in dataset_files.items():
        print(f"[INFO] Loading {dataset} → {path}")
        df = pd.read_csv(path, index_col=0)

        # Filter by model family if requested
        filtered = filter_models_in_pivot(df, selected_models)
        if filtered is None:
            print(f"[WARNING] Dataset {dataset} skipped (no selected models present).")
            continue

        # Dataset size based on training samples (Xcal.csv)
        try:
            dataset_size = get_training_dataset_size(dataset)
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
            continue

        difficulty_scores = compute_difficulty_scores(filtered)
        difficulty_scores["dataset_size"] = int(dataset_size)

        scores_dict[dataset] = difficulty_scores


    if not scores_dict:
        print("[ERROR] No datasets remain after model filtering.")
        return

    rankings = build_rankings(scores_dict)

    # Save JSON with suffix
    output_json = os.path.join(
        OUTPUT_BASE_JSON, f"dataset_difficulty_ranking__{suffix}.json"
    )
    ensure_dir(os.path.dirname(output_json))

    with open(output_json, "w") as f:
        json.dump({"scores": scores_dict, "rankings": rankings}, f, indent=4)

    print(f"[INFO] Saved difficulty ranking JSON → {output_json}")

    # Generate figures per ranking method
    for metric in ["best_rrmse", "mean_rrmse", "mean_best_model"]:
        plot_one_ranking(metric, rankings, scores_dict, suffix)
    
    plot_dataset_size_ranking(rankings, scores_dict, suffix)

if __name__ == "__main__":
    main()
