"""
Results handling module for the preprocessing/model association pipeline.

Responsibilities:
- Convert raw evaluation results to score matrices (DataFrames)
- Save CSV results and timing files
- Save JSON metadata (best_trials and CNN batch sizes)
- Standardize output directory hierarchy

This module is a clean replacement for the long saving section
in the original association_pp_model.py script.
"""

import os
import json
import pandas as pd
import numpy as np

from scripts.utils.build_filename import build_filename
from scripts.utils.make_serializable import make_json_serializable


# =============================================================
# OUTPUT DIRECTORY HELPERS
# =============================================================

def _ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _get_output_directory(cfg):
    """Directory where results for a dataset should be stored."""
    return os.path.join(
        "Results",
        "assoc_pp_model",
        "per_dataset",
        cfg["data_source"]
    )


# =============================================================
# SCORE MATRIX SAVING
# =============================================================

def save_score_matrix(raw_results, cfg):
    """
    Convert raw results to DataFrame in pivot format and save CSV.

    Args:
        raw_results (list): outputs from evaluate_combination()
        cfg (dict): global configuration

    Returns:
        df_scores (pd.DataFrame): long format
        pivot (pd.DataFrame): pivot form matrix of shape (models x preprocessings)
    """

    mode = cfg["mode"]
    out_dir = _get_output_directory(cfg)
    _ensure_dir(out_dir)

    # ---------------------------------------------------------
    # Extract tuple format depending on mode
    # ---------------------------------------------------------
    rows = []
    if mode == "Regression":
        # Expected tuple: (pp_name, mdl_name, metric, best_trials, time, batch_info)
        for pp, mdl, score, _, _, _ in raw_results:
            rows.append((pp, mdl, score))
    else:
        # Expected tuple: (pp_name, mdl_name, acc, f1, fpr, best_trials, time, batch_info)
        for pp, mdl, acc, f1, fpr, _, _, _ in raw_results:
            rows.append((pp, mdl, acc))

    df_scores = pd.DataFrame(rows, columns=["Preprocessing", "Model", "Score"])

    # Pivot: models as rows, preprocessings as columns
    pivot = df_scores.pivot(
        index="Model",
        columns="Preprocessing",
        values="Score"
    )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------
    optim_type = "progressive" if cfg["progressive_optim"] else "uniform"

    file_name = build_filename(
        prefix="results",
        data_source=cfg["data_source"],
        top_n=cfg["top_n_preprocs"],
        optim_type=optim_type,
        model_names=cfg["model_names"],
        adaptive_batch_size=cfg["adaptive_batch_size"],
        extension="csv"
    )

    pivot.to_csv(os.path.join(out_dir, file_name))
    print(f"[INFO] Saved main results matrix → {file_name}")

    return df_scores, pivot


# =============================================================
# TIMING SAVING (GLOBAL + PER MODEL)
# =============================================================

def save_timings(raw_results, cfg):
    """
    Save timing information for:
    - total pipeline execution time
    - per-model average evaluation time

    Args:
        raw_results (list): results of each combination
        cfg (dict): pipeline configuration
    """

    out_dir = _get_output_directory(cfg)
    _ensure_dir(out_dir)

    optim_type = "progressive" if cfg["progressive_optim"] else "uniform"

    # ---------------------------------------------------------
    # Total execution time (measured outside this function)
    # NOTE: This module will not measure time here; the caller
    #       should include "elapsed_time" in cfg if needed.
    # ---------------------------------------------------------

    if "elapsed_time" in cfg:
        timing_data = {"time": cfg["elapsed_time"]}
    else:
        timing_data = {"time": np.nan}

    timing_data.update({
        "data_source": cfg["data_source"],
        "optimization_type": optim_type
    })

    # Save global timing file
    file_name = build_filename(
        prefix="timing_results",
        data_source=cfg["data_source"],
        top_n=cfg["top_n_preprocs"],
        optim_type=optim_type,
        model_names=cfg["model_names"],
        adaptive_batch_size=cfg["adaptive_batch_size"],
        extension="csv"
    )

    timing_csv_path = os.path.join(out_dir, file_name)

    if os.path.exists(timing_csv_path):
        df_prev = pd.read_csv(timing_csv_path)
        df_prev = pd.concat([df_prev, pd.DataFrame([timing_data])], ignore_index=True)
        df_prev.to_csv(timing_csv_path, index=False)
    else:
        pd.DataFrame([timing_data]).to_csv(timing_csv_path, index=False)

    print(f"[INFO] Saved global timing → {file_name}")

    # ---------------------------------------------------------
    # Per-model average evaluation time
    # ---------------------------------------------------------
    # raw_results contain per-combination times

    rows = []
    mode = cfg["mode"]

    if mode == "Regression":
        # Format: (pp, mdl, metric, best, time, batch)
        for _, mdl, _, _, combo_time, _ in raw_results:
            rows.append((mdl, combo_time))
    else:
        # Format: (pp, mdl, acc, f1, fpr, best, time, batch)
        for _, mdl, _, _, _, _, combo_time, _ in raw_results:
            rows.append((mdl, combo_time))

    df_models = pd.DataFrame(rows, columns=["Model", "Time_seconds"])
    df_avg = df_models.groupby("Model", as_index=False)["Time_seconds"].mean()

    file_name_m = file_name.replace("results", "per_model")
    timing_models_path = os.path.join(out_dir, file_name_m)

    if os.path.exists(timing_models_path):
        df_prev = pd.read_csv(timing_models_path)
        df_avg = pd.concat([df_prev, df_avg], ignore_index=True)

    df_avg.to_csv(timing_models_path, index=False)
    print(f"[INFO] Saved per-model timing → {file_name_m}")


# =============================================================
# BEST_TRIALS SAVING (CNN / LGBM)
# =============================================================

def save_best_trials(raw_results, cfg):
    """
    Save best_trials information (only when progressive optimization is used).

    Args:
        raw_results (list): pipeline evaluation results
        cfg (dict): configuration
    """

    if not cfg["progressive_optim"]:
        return

    out_dir = os.path.join(
        "Results",
        "assoc_pp_model",
        "per_dataset",
        cfg["data_source"]
    )
    _ensure_dir(out_dir)

    adaptive_suffix = ""
    if cfg["adaptive_batch_size"] == "static":
        adaptive_suffix = "_static_batch_size"
    elif cfg["adaptive_batch_size"] == "dynamic":
        adaptive_suffix = "_dynamic_batch_size"

    # Collect best_trials for CNN and LGBM
    best_trials_cnn = None
    best_trials_lgbm = None

    mode = cfg["mode"]

    if mode == "Regression":
        for pp, mdl, _, best_trials, _, _ in raw_results:
            if "CNN" in mdl and best_trials is not None:
                best_trials_cnn = best_trials
            elif "LGBM" in mdl and best_trials is not None:
                best_trials_lgbm = best_trials
    else:
        for pp, mdl, _, _, _, best_trials, _, _ in raw_results:
            if "CNN" in mdl and best_trials is not None:
                best_trials_cnn = best_trials
            elif "LGBM" in mdl and best_trials is not None:
                best_trials_lgbm = best_trials

    # Save CNN best_trials
    if best_trials_cnn is not None:
        path = os.path.join(
            out_dir,
            f"best_trials_CNN_{cfg['data_source']}{adaptive_suffix}.json"
        )
        json.dump(make_json_serializable(best_trials_cnn), open(path, "w"), indent=2)
        print(f"[INFO] Saved CNN best_trials → {path}")

    # Save LGBM best_trials
    if best_trials_lgbm is not None:
        path = os.path.join(
            out_dir,
            f"best_trials_LGBM_{cfg['data_source']}{adaptive_suffix}.json"
        )
        json.dump(make_json_serializable(best_trials_lgbm), open(path, "w"), indent=2)
        print(f"[INFO] Saved LGBM best_trials → {path}")


# =============================================================
# CNN BATCH SIZE SAVING
# =============================================================

def save_batch_sizes(raw_results, cfg):
    """
    Save batch size information used by CNN models, for each preprocessing.

    Args:
        raw_results (list): evaluation results
        cfg (dict): pipeline configuration
    """

    out_dir = _get_output_directory(cfg)
    _ensure_dir(out_dir)

    batch_dict = {}

    mode = cfg["mode"]

    if mode == "Regression":
        for pp, mdl, _, _, _, batch in raw_results:
            if "CNN" in mdl and batch is not None:
                batch_dict[pp] = batch
    else:
        for pp, mdl, _, _, _, _, _, batch in raw_results:
            if "CNN" in mdl and batch is not None:
                batch_dict[pp] = batch

    if not batch_dict:
        return

    adaptive_suffix = ""
    if cfg["adaptive_batch_size"] == "static":
        adaptive_suffix = "_static_batch_size"
    elif cfg["adaptive_batch_size"] == "dynamic":
        adaptive_suffix = "_dynamic_batch_size"

    path = os.path.join(
        out_dir,
        f"batch_sizes_CNN_{cfg['data_source']}{adaptive_suffix}.json"
    )

    json.dump(make_json_serializable(batch_dict), open(path, "w"), indent=2)
    print(f"[INFO] Saved CNN batch size mapping → {path}")
