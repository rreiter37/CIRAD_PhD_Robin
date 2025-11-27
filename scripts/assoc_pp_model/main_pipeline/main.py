"""
Main orchestration script for the preprocessing/model association pipeline.
This file handles:
- argument parsing
- dataset loading
- building preprocessing pipelines
- building the model registry
- evaluating all (preprocessing, model) combinations
- saving results and plotting heatmaps

All heavy logic is delegated to specialized modules for clarity.
"""

import os
import numpy as np
import torch
from tqdm import tqdm

from .config import get_config, setup_environment
from .preprocessing import build_preprocessing_list
from .model_registry import get_models, filter_models
from .evaluation import evaluate_combination
from .results_handler import (
    save_score_matrix,
    save_timings,
    save_best_trials,
    save_batch_sizes,
)
from .heatmap import generate_heatmaps
from .parallel import run_parallel_evaluation
from .utils_data import load_dataset_safely


def main():
    # -----------------------------------------------------------
    # 1. Parse arguments and initialize environment
    # -----------------------------------------------------------
    cfg = get_config()          # configuration dict
    setup_environment(cfg)      # seeds, GPU settings, TF, logging

    print(f"[INFO] Dataset selected: {cfg['data_source']}")
    print(f"[INFO] Mode: {cfg['mode']}")
    print(f"[INFO] Optimization strategy: "
          f"{'Progressive' if cfg['progressive_optim'] else 'Uniform'}")
    print(f"[INFO] Parallel execution: {cfg['use_parallelism']}")

    # -----------------------------------------------------------
    # 2. Load dataset (train/val split)
    # -----------------------------------------------------------
    Xcal, Ycal, Xval, Yval = load_dataset_safely(
        mode=cfg["mode"],
        data_source=cfg["data_source"]
    )

    # Some metrics require the number of classes
    num_classes = len(np.unique(Ycal))
    if cfg["mode"] == "Classification":
        print(f"[INFO] Number of classes detected: {num_classes}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Torch device: {device}")

    # -----------------------------------------------------------
    # 3. Generate all preprocessing pipelines
    # -----------------------------------------------------------
    preprocessings = build_preprocessing_list(cfg["random_seed"])

    # -----------------------------------------------------------
    # 4. Build available models and filter based on CLI arguments
    # -----------------------------------------------------------
    dict_models = get_models(
        mode=cfg["mode"],
        rd_seed=cfg["random_seed"],
        num_classes=num_classes,
        device=device,
        adaptive_batch_size=cfg["adaptive_batch_size"]
    )

    models = filter_models(
        cfg["model_names"],
        cfg["mode"],
        dict_models
    )

    # -----------------------------------------------------------
    # 5. Build all combinations (preprocessing, model)
    # -----------------------------------------------------------
    combinations = [
        (pp_name, pp_method, mdl_name, mdl)
        for (pp_name, pp_method) in preprocessings
        for (mdl_name, mdl) in models
    ]

    print(f"[INFO] Total combinations to evaluate: {len(combinations)}")

    # -----------------------------------------------------------
    # 6. Run evaluations (parallel or sequential)
    # -----------------------------------------------------------
    if cfg["use_parallelism"]:
        print("[INFO] Running in parallel mode...")
        raw_results = run_parallel_evaluation(
            combinations, cfg, Xcal, Ycal, Xval, Yval
        )
    else:
        print("[INFO] Running in sequential mode...")
        raw_results = []
        for (pp_name, pp_method, mdl_name, mdl) in tqdm(
            combinations, desc="Evaluations"
        ):
            out = evaluate_combination(
                pp_name, pp_method, mdl_name, mdl,
                cfg, Xcal, Ycal, Xval, Yval
            )
            raw_results.append(out)

    # -----------------------------------------------------------
    # 7. Convert raw results into score matrix
    # -----------------------------------------------------------
    df_scores, pivot_score = save_score_matrix(
        raw_results, cfg
    )

    # -----------------------------------------------------------
    # 8. Generate heatmaps
    # -----------------------------------------------------------
    generate_heatmaps(
        pivot_score=pivot_score,
        results=raw_results,
        cfg=cfg
    )

    # -----------------------------------------------------------
    # 9. Save timing information
    # -----------------------------------------------------------
    save_timings(
        raw_results=raw_results,
        cfg=cfg
    )

    # -----------------------------------------------------------
    # 10. Save best_trials information (CNN & LGBM)
    # -----------------------------------------------------------
    save_best_trials(raw_results, cfg)

    # -----------------------------------------------------------
    # 11. If adaptive batch size: save batch information
    # -----------------------------------------------------------
    save_batch_sizes(raw_results, cfg)

    print("[INFO] Pipeline completed successfully.")


if __name__ == "__main__":
    main()
