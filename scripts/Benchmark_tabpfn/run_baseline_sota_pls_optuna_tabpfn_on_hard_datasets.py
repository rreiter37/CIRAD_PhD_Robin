#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automatically run baseline_sota_pls_optuna_select_then_tabpfn.py
on the hardest datasets according to a difficulty ranking.

This script:
  - Loads a dataset difficulty ranking from a JSON file
  - Lets the user choose the ranking metric
  - Selects the top-N hardest datasets (or all)
  - Optionally skips datasets before a given one
  - Builds dataset paths dynamically (same logic as baseline_sota.py)
  - Calls baseline_sota_pls_optuna_select_then_tabpfn.py with --datasets
  - Optionally triggers figure generation afterwards
"""

import json
import argparse
import subprocess
from pathlib import Path


# =============================================================================
# Project paths (adjust if needed)
# =============================================================================

PROJECT_PATH = "/home/robinr/Desktop/VSCode/CIRAD_PhD_Robin/"
JSON_PATH = (
    PROJECT_PATH
    + "Results/assoc_pp_model/All_datasets/Rank_datasets_difficulty/"
    + "dataset_difficulty_ranking.json"
)

# Script to run (NEW pipeline)
BASELINE_SCRIPT_PATH = (
    "scripts/Benchmark_tabpfn/"
    "baseline_sota_pls_optuna_select_then_tabpfn.py"
)

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ranking_json",
        type=str,
        default=JSON_PATH,
        help="Path to the JSON difficulty ranking file."
    )

    parser.add_argument(
        "--difficulty_ranking",
        type=str,
        default="best_rrmse",
        choices=["best_rrmse", "mean_rrmse", "mean_best_model"],
        help="Metric used to sort datasets by difficulty."
    )

    parser.add_argument(
        "--top_n",
        type=str,
        default="all",
        help='How many datasets to run: integer or "all".'
    )

    parser.add_argument(
        "--after_dataset",
        type=str,
        default=None,
        help="If provided, skip all datasets that appear before this one."
    )

    parser.add_argument(
        "--baseline_script",
        type=str,
        default=BASELINE_SCRIPT_PATH,
        help="Path to baseline_sota_pls_optuna_select_then_tabpfn.py."
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace",
        help="Workspace folder where predictions will be written."
    )

    parser.add_argument(
        "--generate_figures",
        action="store_true",
        help="If set, launches the figure-generation notebook or script."
    )

    return parser.parse_args()


# =============================================================================
# Ranking utilities
# =============================================================================

def load_ranking(json_path):
    """Load the difficulty ranking JSON file."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Ranking file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if "rankings" not in data:
        raise ValueError("JSON ranking file is missing the 'rankings' section.")

    return data["rankings"]


def pick_top_datasets(ranking_list, top_n):
    """Return the top-N datasets. If top_n == 'all', return the full list."""
    if top_n == "all":
        return ranking_list

    top_n = int(top_n)
    return ranking_list[:top_n]


def filter_after_dataset(dataset_list, after_dataset):
    """
    If after_dataset is provided, return the sublist starting from that dataset.
    Example:
        dataset_list = [A, B, C, D]
        after_dataset = "C"
        → returns [C, D]
    """
    if after_dataset is None:
        return dataset_list

    if after_dataset not in dataset_list:
        raise ValueError(
            f"--after_dataset '{after_dataset}' not found in ranking list. "
            "Check spelling and difficulty_ranking."
        )

    start_index = dataset_list.index(after_dataset)
    return dataset_list[start_index:]


# =============================================================================
# Dataset path handling (SAME convention as baseline_sota.py)
# =============================================================================

def dataset_to_path(dataset_name):
    """
    Convert a dataset name into its actual folder path.

    This matches the baseline_sota.py convention:
        Data_nirs4all/Regression/<DATASET_NAME>
    """
    base = Path("Data_nirs4all/Regression")
    return str(base / dataset_name)


# =============================================================================
# Execution helpers
# =============================================================================

def run_baseline(baseline_script, dataset_paths, workspace):
    """
    Call baseline_sota_pls_optuna_select_then_tabpfn.py
    with the overridden dataset list.
    """
    cmd = (
        ["python", baseline_script, "--datasets"]
        + dataset_paths
        + ["--workspace", workspace]
    )

    print("\n=== Running Optuna-PLS → TabPFN baseline ===")
    print("Command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print("Completed.\n")


def run_figures():
    """
    Trigger the figure-generation process.
    Adjust this function to call the correct notebook or script.
    """
    print("Launching figure generation notebook...")
    subprocess.run(
        ["jupyter", "nbconvert", "--execute", "predictions 1.ipynb"],
        check=False,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # 1) Load difficulty ranking
    rankings = load_ranking(args.ranking_json)

    if args.difficulty_ranking not in rankings:
        raise KeyError(
            f"Ranking method '{args.difficulty_ranking}' not found in JSON."
        )

    ranking_list = rankings[args.difficulty_ranking]

    # 2) Optionally skip datasets before --after_dataset
    ranking_list = filter_after_dataset(ranking_list, args.after_dataset)

    # 3) Apply top-N selection
    selected_datasets = pick_top_datasets(ranking_list, args.top_n)

    print("\nSelected datasets (ordered by difficulty):")
    for ds in selected_datasets:
        print("  •", ds)

    # 4) Convert dataset names → folder paths
    dataset_paths = [dataset_to_path(ds) for ds in selected_datasets]

    # 5) Run the Optuna-PLS → TabPFN pipeline
    run_baseline(
        baseline_script=args.baseline_script,
        dataset_paths=dataset_paths,
        workspace=args.workspace,
    )

    # 6) Optionally generate figures
    if args.generate_figures:
        run_figures()


if __name__ == "__main__":
    main()