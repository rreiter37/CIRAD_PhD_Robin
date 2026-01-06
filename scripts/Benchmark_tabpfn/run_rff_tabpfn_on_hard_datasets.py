#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run rff_tabpfn.py automatically on the top-N hardest datasets.

This script:
  - Loads dataset difficulty ranking (.json)
  - Lets the user select the ranking method
  - Selects the top-N hardest datasets (or all)
  - Optionally skips datasets before a given one
  - Builds DATA_PATH dynamically
  - Calls rff_tabpfn.py with overridden dataset list
"""

import json
import argparse
import subprocess
from pathlib import Path


# ===================== Paths ===================== #

PROJECT_PATH = Path("/home/robinr/Desktop/VSCode/CIRAD_PhD_Robin/")
JSON_PATH = PROJECT_PATH / "Results/assoc_pp_model/All_datasets/Rank_datasets_difficulty/dataset_difficulty_ranking.json"
RFF_SCRIPT_PATH = PROJECT_PATH / "scripts/Benchmark_tabpfn/rff_tabpfn.py"


# ===================== Argument parsing ===================== #

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ranking_json",
        type=str,
        default=str(JSON_PATH),
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
        "--workspace",
        type=str,
        default="workspace_rff_tabpfn",
        help="Workspace folder for rff_tabpfn outputs."
    )

    # ===================== RFF parameters ===================== #

    parser.add_argument("--rff_components", type=int, default=256)
    parser.add_argument("--rff_sigma", type=float, default=1.0)
    parser.add_argument(
        "--rff_append_raw",
        action="store_true",
        help="Concatenate raw features with RFF features."
    )
    parser.add_argument("--rff_seed", type=int, default=42)

    return parser.parse_args()


# ===================== Utilities ===================== #

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
    return ranking_list[: int(top_n)]


def filter_after_dataset(dataset_list, after_dataset):
    """
    Skip all datasets that appear before `after_dataset`.
    """
    if after_dataset is None:
        return dataset_list

    if after_dataset not in dataset_list:
        raise ValueError(
            f"--after_dataset '{after_dataset}' not found in ranking list."
        )

    idx = dataset_list.index(after_dataset)
    return dataset_list[idx:]


def dataset_to_path(ds_name):
    """
    Convert dataset name to actual folder path.
    Adjust if your folder structure changes.
    """
    base = PROJECT_PATH / "Data_nirs4all/Regression"
    path = base / ds_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {path}")
    return str(path)


# ===================== Runner ===================== #

def run_rff_tabpfn(args, dataset_paths):
    """
    Call rff_tabpfn.py with the overridden dataset list and RFF parameters.
    """
    cmd = [
        "python",
        str(RFF_SCRIPT_PATH),
        "--datasets",
        *dataset_paths,
        "--workspace", args.workspace,
        "--rff_components", str(args.rff_components),
        "--rff_sigma", str(args.rff_sigma),
        "--rff_seed", str(args.rff_seed),
    ]

    if args.rff_append_raw:
        cmd.append("--rff_append_raw")

    print("\n=== Running rff_tabpfn.py ===")
    print("Command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    print("Completed RFF TabPFN benchmark.\n")


# ===================== Main ===================== #

def main():
    args = parse_args()

    # 1) Load ranking
    rankings = load_ranking(args.ranking_json)
    ranking_list = rankings[args.difficulty_ranking]

    # 2) Filter + top-N
    ranking_list = filter_after_dataset(ranking_list, args.after_dataset)
    selected_datasets = pick_top_datasets(ranking_list, args.top_n)

    print("\nSelected datasets:")
    for ds in selected_datasets:
        print("  •", ds)

    # 3) Dataset names → paths
    dataset_paths = [dataset_to_path(ds) for ds in selected_datasets]

    # 4) Run RFF TabPFN
    run_rff_tabpfn(args, dataset_paths)


if __name__ == "__main__":
    main()
