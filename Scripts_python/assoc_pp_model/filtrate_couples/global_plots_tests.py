#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect (model, preprocessing) pairs that are statistically uninformative:
  - Significantly less performant than the best model for their preprocessing
  - Significantly correlated (redundant) with that best model across datasets,
    as determined by a one-sided Spearman correlation test.

Also generates scatterplots (rank-rank) for redundant pairs, showing the monotonic relationship
that led to their classification as redundant.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------- Parameters ----------------

DEFAULT_INPUT = "Results/assoc_pp_model/All_datasets/noninferiority_reg_diffs.csv"
DEFAULT_OUTPUT = "Results/assoc_pp_model/All_datasets/uninformative_pairs_hypotest_visual.csv"

DEFAULT_ALPHA_PERF = 0.05
DEFAULT_ALPHA_DIVERSITY = 0.05
DEFAULT_MIN_DATASETS = 3
DEFAULT_NORMALIZE = True


# ---------------- Core Functions ----------------

def build_profiles(df: pd.DataFrame, normalize: bool = True, min_datasets: int = 3):
    """Build normalized performance profiles per (model, preprocessing)."""
    df = df.copy()
    if normalize:
        # Normalize RRMSE by best per dataset
        df["rrmse_norm"] = df.groupby("dataset")["delta_value"].transform(lambda x: x / x.min())
    else:
        df["rrmse_norm"] = df["delta_value"]

    pivot = df.pivot_table(index=["candidate_model", "prep"], columns="dataset",
                           values="rrmse_norm", aggfunc="mean")
    pivot = pivot.dropna(thresh=min_datasets, axis=0)
    return pivot


def wilcoxon_test_per_prep(df: pd.DataFrame):
    """For each preprocessing, test whether each model is significantly worse than the best one."""
    results = []
    for prep, sub in df.groupby("prep"):
        pivot = sub.pivot(index="candidate_model", columns="dataset", values="delta_value")
        if pivot.shape[1] < 3:
            continue
        best_model = pivot.mean(axis=1).idxmin()
        for model in pivot.index:
            if model == best_model:
                continue
            common = pivot.columns[pivot.loc[model].notna() & pivot.loc[best_model].notna()]
            if len(common) < 3:
                continue
            diffs = pivot.loc[model, common] - pivot.loc[best_model, common]
            try:
                stat, p_perf = wilcoxon(diffs, alternative="greater")
            except Exception:
                p_perf = np.nan
            results.append({
                "prep": prep,
                "candidate_model": model,
                "ref_model": best_model,
                "median_diff": np.median(diffs),
                "p_perf": p_perf
            })
    return pd.DataFrame(results)


def correlation_test_per_pair(pivot: pd.DataFrame, pairs: pd.DataFrame):
    """Compute Spearman correlation and p-value between candidate and best model profiles."""
    corrs = []
    for _, row in pairs.iterrows():
        cand = (row["candidate_model"], row["prep"])
        ref = (row["ref_model"], row["prep"])
        if cand not in pivot.index or ref not in pivot.index:
            continue
        v1, v2 = pivot.loc[cand].to_numpy(), pivot.loc[ref].to_numpy()
        mask = np.isfinite(v1) & np.isfinite(v2)
        if np.sum(mask) < 3:
            continue
        rho, p_corr = spearmanr(v1[mask], v2[mask], alternative="greater")
        corrs.append({
            "candidate_model": row["candidate_model"],
            "prep": row["prep"],
            "ref_model": row["ref_model"],
            "rho": rho,
            "p_corr": p_corr
        })
    return pd.DataFrame(corrs)


def combine_tests(perf_df: pd.DataFrame, corr_df: pd.DataFrame,
                  alpha_perf: float, alpha_div: float, apply_fdr: bool = True):
    """Combine both tests: a pair is uninformative if worse (Wilcoxon) and redundant (Spearman)."""
    df = pd.merge(perf_df, corr_df, on=["candidate_model", "prep", "ref_model"], how="inner")

    # Adjust p-values (FDR)
    if apply_fdr and len(df) > 0:
        df["p_perf_adj"] = multipletests(df["p_perf"], alpha=alpha_perf, method="fdr_bh")[1]
        df["p_corr_adj"] = multipletests(df["p_corr"], alpha=alpha_div, method="fdr_bh")[1]
    else:
        df["p_perf_adj"] = df["p_perf"]
        df["p_corr_adj"] = df["p_corr"]

    df["signif_worse"] = df["p_perf_adj"] < alpha_perf
    df["signif_corr"] = df["p_corr_adj"] < alpha_div
    df["uninformative"] = df["signif_worse"] & df["signif_corr"]
    return df


def plot_spearman_visuals(df, pivot, output_dir):
    """
    Generate rank-rank scatterplots showing Spearman correlation between
    each uninformative candidate and its reference model.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_plots = 0

    for _, row in df[df["uninformative"]].iterrows():
        cand = (row["candidate_model"], row["prep"])
        ref = (row["ref_model"], row["prep"])
        if cand not in pivot.index or ref not in pivot.index:
            continue

        # Extract values and datasets
        v1 = pivot.loc[cand]
        v2 = pivot.loc[ref]
        mask = v1.notna() & v2.notna()
        if mask.sum() < 3:
            continue

        datasets = v1.index[mask]
        vals_cand = v1[mask].rank().to_numpy()
        vals_ref = v2[mask].rank().to_numpy()

        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=vals_ref, y=vals_cand)
        plt.plot([vals_ref.min(), vals_ref.max()],
                 [vals_ref.min(), vals_ref.max()], 'r--', label='Perfect monotone')
        plt.xlabel(f"Ranks of {row['ref_model']} (reference)")
        plt.ylabel(f"Ranks of {row['candidate_model']}")
        plt.title(f"Spearman correlation ({row['prep']})\nρ={row['rho']:.2f}, p={row['p_corr_adj']:.3g}")
        for i, ds in enumerate(datasets):
            plt.text(vals_ref[i] + 0.05, vals_cand[i], ds, fontsize=7)
        plt.legend()
        plt.tight_layout()

        fname = f"{row['prep']}_{row['candidate_model']}_vs_{row['ref_model']}_spearman.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        n_plots += 1

    print(f"[INFO] Generated {n_plots} Spearman visualizations in {output_dir}")



def plot_global_performance_diversity_map(df, output_path, alpha_perf=0.05, alpha_div=0.05):
    """
    Create a global scatterplot showing performance vs. redundancy:
      - X-axis: Spearman correlation (rho)
      - Y-axis: Median performance difference (Δ = candidate - best)
      - Color encodes statistical significance (uninformative/worse/redundant/informative)
      - Marker shape encodes model type (PLS, LGBM, Ridge, CNN, etc.)
      - Annotations highlight extreme or interesting points (Δ>0.1, ρ>0.8, Δ<0)
    """

    if df.empty:
        print("[WARN] Empty dataframe, skipping global visualization.")
        return

    # ---------------- Categorization by test significance ----------------
    cond_worse = df["p_perf_adj"] < alpha_perf
    cond_corr = df["p_corr_adj"] < alpha_div

    def categorize(row):
        if cond_worse.loc[row.name] and cond_corr.loc[row.name]:
            return "uninformative"
        elif cond_worse.loc[row.name]:
            return "worse_only"
        elif cond_corr.loc[row.name]:
            return "redundant_only"
        else:
            return "informative"

    df["category"] = df.apply(categorize, axis=1)

    # ---------------- Assign marker shapes by model ----------------
    marker_map = {
        "PLS": "o",        # circle
        "Ridge": "P",      # plus (filled)
        "LGBM": "s",       # square
        "CNN": "*",        # star
    }

    def get_marker(model_name):
        for key, marker in marker_map.items():
            if key.lower() in model_name.lower():
                return marker
        return "o"

    df["marker"] = df["candidate_model"].apply(get_marker)

    # ---------------- Color palette ----------------
    palette = {
        "uninformative": "red",
        "worse_only": "blue",
        "redundant_only": "orange",
        "informative": "lightgray"
    }

    # ---------------- Plot ----------------
    plt.figure(figsize=(10, 7))

    # Plot by model for distinct markers
    for model_name, sub in df.groupby("candidate_model"):
        sns.scatterplot(
            data=sub,
            x="rho",
            y="median_diff",
            hue="category",
            palette=palette,
            alpha=0.85,
            edgecolor="black",
            marker=sub["marker"].iloc[0],
            s=100,
            legend=False
        )

    # ---------------- Annotate extreme points ----------------
    for _, row in df.iterrows():
        if (row["rho"] < 0) or (row["median_diff"] > 0.1) or (row["median_diff"] < -0.1):
            label = f"{row['prep']}"
            plt.text(
                row["rho"] + 0.01,
                row["median_diff"],
                label,
                fontsize=7,
                color="black",
                alpha=0.8
            )

    # ---------------- Legends ----------------
    from matplotlib.lines import Line2D

    color_legend = [
        Line2D([0], [0], marker='o', color='w',
               label=label, markerfacecolor=color, markersize=10, markeredgecolor='black')
        for label, color in palette.items()
    ]

    marker_legend = [
        Line2D([0], [0], marker=m, color='k', label=model, linestyle='None', markersize=9)
        for model, m in marker_map.items()
    ]

    plt.legend(
        handles=color_legend + marker_legend,
        title="Category (color) & Model (shape)",
        loc="best",
        frameon=True
    )

    # ---------------- Axes and title ----------------
    plt.axhline(0, color="black", linestyle="--", lw=1)
    plt.axvline(0.5, color="gray", linestyle=":", lw=1)
    plt.xlabel("Spearman correlation ρ (redundancy)")
    plt.ylabel("Median Δ (candidate - best model)")
    plt.title(
        "Global Performance vs. Redundancy Map\n"
        "(Color = statistical category, Shape = model, Label = extreme couples)",
        fontsize=12
    )
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Global map with model markers and annotations saved to {output_path}")



# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Detect and visualize statistically redundant (model, preprocessing) pairs.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Input CSV (dataset, candidate_model, prep, delta_value).")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output CSV for uninformative pairs.")
    parser.add_argument("--alpha_perf", type=float, default=DEFAULT_ALPHA_PERF,
                        help="Significance level for performance test.")
    parser.add_argument("--alpha_div", type=float, default=DEFAULT_ALPHA_DIVERSITY,
                        help="Significance level for correlation test.")
    parser.add_argument("--min_datasets", type=int, default=DEFAULT_MIN_DATASETS,
                        help="Minimum datasets per pair.")
    parser.add_argument("--normalize", action="store_true", default=DEFAULT_NORMALIZE,
                        help="Normalize RRMSE per dataset.")
    parser.add_argument("--fig_dir", type=str, default="Figures/assoc_pp_model/Spearman_visuals",
                        help="Directory to save correlation scatterplots.")
    args = parser.parse_args()

    print(f"[INFO] Loading data from {args.input}")
    df = pd.read_csv(args.input)
    if df.empty:
        print("[ERROR] Input CSV is empty or invalid.")
        return

    print("[INFO] Building performance profiles ...")
    pivot = build_profiles(df, normalize=args.normalize, min_datasets=args.min_datasets)

    print("[INFO] Running Wilcoxon performance tests ...")
    perf_df = wilcoxon_test_per_prep(df)

    print("[INFO] Running Spearman correlation tests ...")
    corr_df = correlation_test_per_pair(pivot, perf_df)

    print("[INFO] Combining performance and redundancy tests ...")
    combined = combine_tests(perf_df, corr_df, args.alpha_perf, args.alpha_div)
    combined.to_csv(args.output, index=False)
    print(f"[INFO] Results saved to {args.output}")

    print("[INFO] Generating Spearman rank-rank visualizations ...")
    plot_spearman_visuals(combined, pivot, args.fig_dir)

    print(f"[INFO] Done. Uninformative pairs: {combined['uninformative'].sum()}")

    # --- Global visualization ---
    plot_global_performance_diversity_map(
        combined,
        output_path=os.path.splitext(args.output)[0] + "_global_map.png",
        alpha_perf=args.alpha_perf,
        alpha_div=args.alpha_div
    )



if __name__ == "__main__":
    main()
