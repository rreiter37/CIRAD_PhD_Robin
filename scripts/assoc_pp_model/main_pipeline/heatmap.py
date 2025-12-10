"""
Heatmap generation module (modified to highlight best model per preprocessing).

This module handles:
- Building the main heatmap (RMSE for Regression, Accuracy for Classification)
- Optional heatmaps for F1-score and FPR (Classification only)
- Pretty styling and clean axis formatting
- Saving figures in the correct dataset subdirectory
- NEW: Red rectangle around the best model for each preprocessing
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle   # <-- Added for red rectangles

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

    means = pivot.mean(axis=0)
    top_cols = means.sort_values(ascending=False).head(N).index

    return pivot[top_cols]


def _heatmap_style():
    """Configure general Seaborn / Matplotlib styles."""
    sns.set_theme(style="white", font_scale=1.2)
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.titleweight"] = "bold"


def _add_best_boxes(ax, pivot, mode):
    """
    Draw a red rectangle around the best score for each preprocessing column.

    For regression → best = min value
    For classification → best = max value
    """

    # Loop through columns (= preprocessings)
    for j, col in enumerate(pivot.columns):

        values = pivot[col]

        # Skip entirely NaN columns
        if values.isnull().all():
            continue

        # Select optimal value depending on task type
        if mode == "Regression":
            best_val = values.min()
        else:  # Classification
            best_val = values.max()

        # List of rows (models) matching the best score
        best_indices = values[values == best_val].index

        # Draw a red rectangle around each best cell
        for idx in best_indices:
            i = list(pivot.index).index(idx)
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1, 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2
                )
            )


def _remove_outliers_from_pivot(pivot, mode):
    """
    Remove outliers from the pivoted score table, following the same logic
    as the evaluation script:
    - Regression: very large values are removed
    - Classification: very low accuracies are removed
    """

    cleaned = pivot.copy()

    for col in cleaned.columns:
        col_values = cleaned[col]

        # Ignore empty columns
        if col_values.isnull().all():
            continue

        mean_val = np.nanmean(col_values.values)

        # Regression → remove abnormally large RMSE
        if mode == "Regression":
            mask = np.abs(col_values) > 10 * mean_val
            cleaned.loc[mask, col] = np.nan

        # Classification → remove abnormally low accuracy
        else:
            mask = col_values < 0.4 * mean_val
            cleaned.loc[mask, col] = np.nan

    return cleaned


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
    Generates:
    - Main heatmap (RMSE or Accuracy)
    - Extra heatmaps (F1, FPR) for Classification
    """

    mode = cfg["mode"]
    only_colors = cfg["only_colors"]

    _heatmap_style()

    # Optional top-N filtering
    pivot_filtered = _apply_top_n_filter(pivot_score, cfg)

    # Remove outliers as in the main evaluation script
    pivot_filtered = _remove_outliers_from_pivot(pivot_filtered, mode)

    # -----------------------------------------------------------
    # MAIN HEATMAP
    # -----------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(max(10, pivot_filtered.shape[1] * 0.6), 6)
    )

    heatmap = sns.heatmap(
        pivot_filtered,
        annot=not only_colors,
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        fmt=".3f",
        cbar=True,
        ax=ax
    )

    title_metric = "Normalized RMSE" if mode == "Regression" else "Accuracy"
    ax.set_title(f"Model vs Preprocessing — {title_metric}")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel("Model")

    # === NEW: Add red boxes around best model per preprocessing ===
    _add_best_boxes(ax, pivot_filtered, mode)

    _save_heatmap(fig, cfg, ext="main")
    plt.close(fig)

    # -----------------------------------------------------------
    # Secondary heatmaps (Classification only)
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

    rows_f1 = []
    rows_fpr = []

    # Extract metrics from raw results
    for pp, mdl, acc, f1, fpr, *_ in results:
        rows_f1.append((pp, mdl, f1))
        rows_fpr.append((pp, mdl, fpr))

    df_f1 = pd.DataFrame(rows_f1, columns=["Preprocessing", "Model", "F1"])
    df_fpr = pd.DataFrame(rows_fpr, columns=["Preprocessing", "Model", "FPR"])

    pivot_f1 = df_f1.pivot(index="Model", columns="Preprocessing", values="F1")
    pivot_fpr = df_fpr.pivot(index="Model", columns="Preprocessing", values="FPR")

    # Top-N filtering
    pivot_f1 = _apply_top_n_filter(pivot_f1, cfg)
    pivot_fpr = _apply_top_n_filter(pivot_fpr, cfg)

    pivot_f1 = _remove_outliers_from_pivot(pivot_f1, mode="Classification")
    pivot_fpr = _remove_outliers_from_pivot(pivot_fpr, mode="Regression")  # min is best

    # ---- F1 Heatmap ----
    fig, ax = plt.subplots(
        figsize=(max(10, pivot_f1.shape[1] * 0.6), 6)
    )
    sns.heatmap(
        pivot_f1,
        annot=not only_colors,
        cmap="magma",
        linewidths=0.5,
        linecolor="white",
        fmt=".3f",
        cbar=True,
        ax=ax
    )
    ax.set_title("Model vs Preprocessing — F1-score")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel("Model")

    # Red rectangles around best F1 values (max)
    _add_best_boxes(ax, pivot_f1, mode="Classification")

    _save_heatmap(fig, cfg, ext="f1")
    plt.close(fig)

    # ---- FPR Heatmap ----
    fig, ax = plt.subplots(
        figsize=(max(10, pivot_fpr.shape[1] * 0.6), 6)
    )
    sns.heatmap(
        pivot_fpr,
        annot=not only_colors,
        cmap="inferno",
        linewidths=0.5,
        linecolor="white",
        fmt=".3f",
        cbar=True,
        ax=ax
    )
    ax.set_title("Model vs Preprocessing — FPR")
    ax.set_xlabel("Preprocessing")
    ax.set_ylabel("Model")

    # Red rectangles around best FPR values (min)
    _add_best_boxes(ax, pivot_fpr, mode="Regression")  # Regression = min is best

    _save_heatmap(fig, cfg, ext="fpr")
    plt.close(fig)