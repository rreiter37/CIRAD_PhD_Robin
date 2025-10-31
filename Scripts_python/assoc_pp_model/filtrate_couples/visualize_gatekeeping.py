#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global visualizations for the iterative gatekeeping (adaptive correlation) results.

Reads *_reg_report.csv and *_classif_report.csv and produces:
  1. Decision heatmap (models × preprocessings)
  2. P-value histograms (Gate 1 & Gate 2)
  3. Scatter Gate 1 vs Gate 2 (Performance vs Correlation)
  4. Global decision pie chart

All figures are aggregated per task (Regression / Classification).
No dataset- or model-specific plots.

Author: ChatGPT (comments in English)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR_DEFAULT = "Results/asso_pp_model/All_datasets"
FIG_DIR_DEFAULT = "Figures/assoc_pp_model/All_datasets"
PREFIX_DEFAULT = "gatekeeping_iter_adaptive"

sns.set(style="whitegrid", context="talk")

# -------------------------------------------------------------------------
# Utility: Load data
# -------------------------------------------------------------------------

def load_reports(results_dir: str, prefix: str):
    """Load regression and classification reports if they exist."""
    reg_path = os.path.join(results_dir, f"{prefix}_reg_report.csv")
    clf_path = os.path.join(results_dir, f"{prefix}_classif_report.csv")
    df_reg = pd.read_csv(reg_path) if os.path.exists(reg_path) else pd.DataFrame()
    df_clf = pd.read_csv(clf_path) if os.path.exists(clf_path) else pd.DataFrame()
    return df_reg, df_clf

# -------------------------------------------------------------------------
# Visualization functions
# -------------------------------------------------------------------------

def decision_heatmap(df: pd.DataFrame, task_label: str, out_dir: str):
    """Heatmap of final decisions (KEEP/EXCLUDE) across all (model, prep)."""
    if df.empty:
        print(f"[WARN] No data for {task_label}, skipping heatmap.")
        return
    pivot = df.pivot_table(index="candidate_model", columns="prep",
                           values="decision", aggfunc=lambda x: x.iloc[0])
    mat = pivot.applymap(lambda v: 1 if str(v).upper()=="KEEP" else 0 if str(v).upper()=="EXCLUDE" else np.nan)
    cmap = sns.color_palette(["red", "green"])
    plt.figure(figsize=(max(8, 0.6*len(mat.columns)), max(6, 0.6*len(mat.index))))
    sns.heatmap(mat, cmap=cmap, cbar=False, linewidths=0.5, linecolor="gray",
                xticklabels=True, yticklabels=True, annot=True, fmt=".0f")
    plt.title(f"Global Decision Heatmap ({task_label})", fontsize=14)
    plt.xlabel("Preprocessing")
    plt.ylabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{task_label}_decision_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved {task_label} heatmap -> {out_path}")

def pvalue_distributions(df: pd.DataFrame, task_label: str, out_dir: str):
    """Histogram of p-values for Gate 1 and Gate 2."""
    if df.empty:
        return
    plt.figure(figsize=(8,6))
    sns.histplot(df["gate1_perf_sign_p"].dropna(), bins=20, color="steelblue", alpha=0.6, label="Gate 1 (Performance)")
    sns.histplot(df["gate2_corr_p"].dropna(), bins=20, color="orange", alpha=0.6, label="Gate 2 (Correlation)")
    plt.axvline(0.05, color="black", linestyle="--", label="α = 0.05")
    plt.xlabel("p-value")
    plt.ylabel("Count")
    plt.title(f"P-value distributions ({task_label})")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{task_label}_pvalue_distributions.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved p-value histograms -> {out_path}")

def scatter_perf_corr(df: pd.DataFrame, task_label: str, out_dir: str):
    """Scatter plot: Gate 1 vs Gate 2 significance."""
    if df.empty:
        return
    subset = df.dropna(subset=["gate1_perf_sign_p","gate2_corr_q"])
    if subset.empty:
        print(f"[WARN] No valid p/q values for {task_label}, skipping scatter.")
        return
    color_map = {"KEEP":"green","EXCLUDE":"red"}
    plt.figure(figsize=(7,6))
    plt.scatter(subset["gate1_perf_sign_p"], subset["gate2_corr_q"],
                c=subset["decision"].map(color_map), s=150/(subset["median_delta"]+1e-3),
                alpha=0.7, edgecolor="k", linewidth=0.3)
    plt.axvline(0.05, color="gray", linestyle="--")
    plt.axhline(0.05, color="gray", linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Gate 1 p-value (Performance)")
    plt.ylabel("Gate 2 q-value (Correlation, FDR)")
    plt.title(f"Performance vs Correlation ({task_label})")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{task_label}_scatter_perf_corr.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved scatter plot -> {out_path}")

def global_pie(df: pd.DataFrame, task_label: str, out_dir: str):
    """Pie chart: KEEP vs EXCLUDE proportions."""
    if df.empty:
        return
    counts = df["decision"].value_counts()
    plt.figure(figsize=(5,5))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%",
            colors=["green" if lab=="KEEP" else "red" for lab in counts.index])
    plt.title(f"Overall decision breakdown ({task_label})")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{task_label}_decision_pie.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved pie chart -> {out_path}")

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

results_dir = RESULTS_DIR_DEFAULT
fig_dir = FIG_DIR_DEFAULT
prefix = PREFIX_DEFAULT
os.makedirs(fig_dir, exist_ok=True)

df_reg, df_clf = load_reports(results_dir, prefix)

for task, df in zip(["regression", "classification"], [df_reg, df_clf]):
    if df.empty:
        print(f"[INFO] No data for {task}, skipping.")
        continue
    decision_heatmap(df, task, fig_dir)
    pvalue_distributions(df, task, fig_dir)
    scatter_perf_corr(df, task, fig_dir)
    global_pie(df, task, fig_dir)
