#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run rff_tabpfn.py automatically on the top-N hardest datasets.
"""

import json
import argparse
import subprocess
from pathlib import Path


JSON_PATH = "Results/assoc_pp_model/All_datasets/Rank_datasets_difficulty/dataset_difficulty_ranking.json"
RFF_SCRIPT = "scripts/Benchmark_tabpfn/rff_tabpfn.py"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ranking_json", type=str, default=JSON_PATH)
    parser.add_argument("--difficulty_ranking", type=str, default="best_rrmse")
    parser.add_argument("--top_n", type=str, default="all")
    parser.add_argument("--after_dataset", type=str, default=None)

    parser.add_argument("--workspace", type=str, default="workspace_rff_tabpfn")

    parser.add_argument("--rff_components", type=int, default=256)
    parser.add_argument("--rff_sigma", type=float, default=1.0)
    parser.add_argument("--rff_append_raw", action="store_true")
    parser.add_argument("--rff_seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.ranking_json, "r") as f:
        rankings = json.load(f)["rankings"][args.difficulty_ranking]

    if args.after_dataset is not None:
        rankings = rankings[rankings.index(args.after_dataset):]

    datasets = rankings if args.top_n == "all" else rankings[: int(args.top_n)]

    dataset_paths = [f"Data/Regression/{ds}" for ds in datasets]

    cmd = [
        "python", RFF_SCRIPT,
        "--datasets", *dataset_paths,
        "--workspace", args.workspace,
        "--rff_components", str(args.rff_components),
        "--rff_sigma", str(args.rff_sigma),
        "--rff_seed", str(args.rff_seed),
    ]

    if args.rff_append_raw:
        cmd.append("--rff_append_raw")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
