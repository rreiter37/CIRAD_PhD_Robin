#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run baseline_sota.py automatically on the top-N hardest datasets.

This script:
  - Loads dataset difficulty ranking (.json)
  - Lets the user select the ranking method (best_rrmse, mean_rrmse, mean_best_model)
  - Selects the top-N hardest datasets (or all)
  - Builds DATA_PATH dynamically
  - Calls baseline_sota.py with overridden dataset list
  - Optionally triggers figure generation afterwards
"""

import json
import argparse
import subprocess
from pathlib import Path


JSON_PATH = "Results/assoc_pp_model/All_datasets/Rank_datasets_difficulty/dataset_difficulty_ranking.json"
BASELINE_PATH = "scripts/Benchmark_tabpfn/baseline_sota.py"


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
        help="If provided, skip all datasets that appear before this one in the difficulty ranking."
    )

    parser.add_argument(
        "--baseline_script",
        type=str,
        default=BASELINE_PATH,
        help="Path to baseline_sota.py."
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace",
        help="Workspace folder where baseline_sota.py will write predictions."
    )

    parser.add_argument(
        "--generate_figures",
        action="store_true",
        help="If set, launches the figure-generation notebook or script."
    )

    return parser.parse_args()


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
    """Return the top-N datasets. If top_n = 'all', return full list."""
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

    Raises an error if after_dataset is not found.
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

def dataset_to_path(ds_name):
    """
    Convert dataset name to actual folder path.

    Adjust this function to match your filesystem layout.
    """
    base = Path("Data/Regression")
    return str(base / ds_name)


def run_baseline(baseline_script, dataset_paths, workspace):
    """
    Call baseline_sota.py with the overridden dataset list.
    """
    cmd = [
        "python",
        baseline_script,
        "--datasets",
    ] + dataset_paths + ["--workspace", workspace]

    print("\n=== Running baseline_sota.py ===")
    subprocess.run(cmd, check=True)
    print("Completed.\n")


def run_figures():
    """
    Trigger the figure-generation process.
    Modify this to call the correct .ipynb or python script.
    """
    print("Launching figure generation notebook...")
    subprocess.run(["jupyter", "nbconvert", "--execute", "predictions 1.ipynb"], check=False)


def main():
    args = parse_args()

    # 1) Load ranking
    rankings = load_ranking(args.ranking_json)
    if args.difficulty_ranking not in rankings:
        raise KeyError(f"Ranking method {args.difficulty_ranking} not found.")

    ranking_list = rankings[args.difficulty_ranking]

    # 2) Optionally skip datasets before --after_dataset
    ranking_list = filter_after_dataset(ranking_list, args.after_dataset)

    # 3) Apply top_n selection AFTER filtering
    selected_datasets = pick_top_datasets(ranking_list, args.top_n)

    print("\nSelected datasets:")
    for ds in selected_datasets:
        print("  •", ds)

    # 3) Convert dataset names → actual folder paths
    dataset_paths = [dataset_to_path(ds) for ds in selected_datasets]

    # 4) Run baseline_sota.py
    run_baseline(
        baseline_script=args.baseline_script,
        dataset_paths=dataset_paths,
        workspace=args.workspace,
    )

    # 5) Optionally run figure generation
    if args.generate_figures:
        run_figures()


if __name__ == "__main__":
    main()
