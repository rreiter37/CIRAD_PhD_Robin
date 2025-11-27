"""
Heatmap generation module.

This module handles:
- Building the main heatmap (RMSE for Regression, Accuracy for Classification)
- Optional heatmaps for F1-score and FPR (Classification only)
- Pretty styling and clean axis formatting
- Saving figures in the correct dataset subdirectory

Fully compatible with the pivot tables produced by results_handler.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils.build_filename import build_filename


# ===============================================================
# Internal helpers
# ===============================================================

def _ensure_dir(path):
    """Create directory if needed."""
    os.makedirs(path, exist_ok=True)


def _get_output_dir(cfg):
    """Return the output directory for this dataset."""
    return os.path.join(
        "Figures",
        "assoc_pp_model",
        "per_dataset",
        cfg["data_source"]
    )


def _apply_top_n_filter(pivot, cfg):
    """
    Keep only the top N preprocessings if requested.
    Sorting is based on the mean score of each column.
    """
    if cfg["top_n_preprocs"] is None:
        return pivot

    N = cfg["top_n_preprocs"]

    # Compute mean score per preprocessing
    means = pivot.mean(axis=0)
    top_cols = means.sort_values(ascending=False).head(N).index

    return pivot[top_cols]


def _heatmap_style():
    """Configure general Seaborn / Matplotlib styles."""
    sns.set_theme(style="white", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.titleweight"] = "bold"


def _save_heatmap(fig, cfg, ext, title_suffix=""):
    """
    Save heatmap figure with clean naming.
    ext: "main", "f1", "fpr"
    """
    out_dir = _get_output_dir(cfg)
    _ensure_dir(out_dir)

    optim_type = "progressive" if cfg["progressive_optim"] else "uniform"

    fname = build_filename(
        prefix=f"heatmap_{ext}",
        data_source=cfg["data_source"],
        top_n=cfg["top_n_preprocs"],
        optim_type=optim_type,
        model_names=cfg["model_names"],
        adaptive_batch_size=cfg["adaptive_batch_size"],
        extension="png"
    )

    fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
    print(f"[INFO] Saved heatmap → {fname}")


# ===============================================================
# Main public interface
# ===============================================================

def generate_heatmaps(pivot_score, results, cfg):
    """
    Main public function that generates:
    - The main heatmap (RMSE in regression, ACC in classification)
    - Extra classification heatmaps: F1 and FPR

    Args:
        pivot_score (pd.DataFrame): pivot table of model x preprocessing (main metric)
        results (list): raw results from evaluate_combination()
        cfg (dict): configuration
    """

    mode = cfg["mode"]
    only_colors = cfg["only_colors"]

    # Apply Seaborn style
    _heatmap_style()

    # Optionally restrict to top-N preprocessings
    pivot_filtered = _apply_top_n_filter(pivot_score, cfg)

    # -----------------------------------------------------------
    # MAIN HEATMAP
    # -----------------------------------------------------------
    fig = plt.figure(figsize=(max(10, pivot_filtered.shape[1] * 0.6), 6))
    ax = sns.heatmap(
        pivot_filtered,
        annot=not only_colors,
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        fmt=".3f",
        cbar=True
    )

    title_metric = "Normalized RMSE" if mode == "Regression" else "Accuracy"
    ax.set_title(f"Model vs Preprocessing — {title_metric}")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel("Model")

    _save_heatmap(fig, cfg, ext="main")
    plt.close(fig)

    # -----------------------------------------------------------
    # CLASSIFICATION: F1-SCORE & FPR HEATMAPS
    # -----------------------------------------------------------
    if mode == "Classification":
        _generate_secondary_heatmaps(results, cfg, only_colors)


# ===============================================================
# Classification secondary heatmaps
# ===============================================================

def _generate_secondary_heatmaps(results, cfg, only_colors):
    """
    Build and save the F1-score and FPR heatmaps for classification tasks.
    """

    # Extract tuples from results
    # Format: (pp, mdl, acc, f1, fpr, best, time, batch)
    rows_f1 = []
    rows_fpr = []

    for pp, mdl, acc, f1, fpr, *_ in results:
        rows_f1.append((pp, mdl, f1))
        rows_fpr.append((pp, mdl, fpr))

    df_f1 = pd.DataFrame(rows_f1, columns=["Preprocessing", "Model", "F1"])
    df_fpr = pd.DataFrame(rows_fpr, columns=["Preprocessing", "Model", "FPR"])

    pivot_f1 = df_f1.pivot(index="Model", columns="Preprocessing", values="F1")
    pivot_fpr = df_fpr.pivot(index="Model", columns="Preprocessing", values="FPR")

    # Apply top-N filter
    pivot_f1 = _apply_top_n_filter(pivot_f1, cfg)
    pivot_fpr = _apply_top_n_filter(pivot_fpr, cfg)

    # Create and save the F1 heatmap
    fig = plt.figure(figsize=(max(10, pivot_f1.shape[1] * 0.6), 6))
    ax = sns.heatmap(
        pivot_f1,
        annot=not only_colors,
        cmap="magma",
        linewidths=0.5,
        linecolor="white",
        fmt=".3f",
        cbar=True
    )
    ax.set_title("Model vs Preprocessing — F1-score")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel("Model")

    _save_heatmap(fig, cfg, ext="f1")
    plt.close(fig)

    # Create and save the FPR heatmap
    fig = plt.figure(figsize=(max(10, pivot_fpr.shape[1] * 0.6), 6))
    ax = sns.heatmap(
        pivot_fpr,
        annot=not only_colors,
        cmap="inferno",
        linewidths=0.5,
        linecolor="white",
        fmt=".3f",
        cbar=True
    )
    ax.set_title("Model vs Preprocessing — FPR")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel("Model")

    _save_heatmap(fig, cfg, ext="fpr")
    plt.close(fig)
