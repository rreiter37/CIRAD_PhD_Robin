#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_tabpfn_best_preproc.py

Create diagnostic plots from MOD7 "best_tabpfn_per_dataset.csv".

Goals:
- Understand which preprocessing choices are frequently selected (counts)
- Understand which choices tend to perform better/worse (distribution of final_test_nrmse)
- Provide categorical views for PCA and RFF (on/off)
- Provide parameter-specific views for PCA and RFF when enabled

Expected input columns (from MOD7 BestResult):
- dataset
- best_simple_preproc
- best_standardization
- best_pca
- best_rff
- val_nrmse
- final_test_nrmse

All comments are intentionally written in English.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# Parsing helpers
# ==============================

_RFF_RE = re.compile(r"rff_nc(?P<nc>\d+)_sg(?P<sg>[0-9.eE+-]+)_ar1")


def parse_rff(label: str):
    """
    Parse the RFF label produced by MOD7, e.g. 'rff_nc256_sg0.5_ar1'.
    Returns (use_rff, n_components, sigma) or (False, None, None).
    """
    if not isinstance(label, str):
        return False, None, None
    if label.strip().lower() == "no_rff":
        return False, None, None
    m = _RFF_RE.match(label.strip())
    if m is None:
        # Unknown format -> treat as "enabled but params missing"
        return True, None, None
    return True, int(m.group("nc")), float(m.group("sg"))


def parse_pca(label: str):
    """
    Parse PCA label into categorical flags.
    - on/off
    - pca_family: no_pca / pca_0.99 / pca_adapt / other
    - pca_fraction: for pca_adapt_0.10n -> 0.10, for pca_adapt_0.25n -> 0.25, else None
    """
    if not isinstance(label, str):
        return False, "unknown", None
    s = label.strip().lower()
    if s == "no_pca":
        return False, "no_pca", None
    if s.startswith("pca_0.99"):
        return True, "pca_0.99", None
    if s.startswith("pca_adapt_"):
        # Typical format: pca_adapt_0.25n
        frac = None
        m = re.search(r"pca_adapt_(\d*\.?\d+)n", s)
        if m:
            try:
                frac = float(m.group(1))
            except Exception:
                frac = None
        return True, "pca_adapt", frac
    return True, "pca_other", None


# ==============================
# Plot helpers
# ==============================

def save_fig(fig, outpath: Path, dpi: int = 200):
    """Save a matplotlib figure safely."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def bar_count(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: Path, top_k: int = 30):
    """
    Make a simple bar chart of counts for a categorical series.
    """
    counts = series.value_counts(dropna=False).head(top_k)
    fig = plt.figure(figsize=(max(10, 0.4 * len(counts)), 6))
    ax = fig.add_subplot(111)
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, outpath)


def boxplot_by_category(df: pd.DataFrame, cat_col: str, metric_col: str, title: str, outpath: Path, max_cats: int = 30):
    """
    Make a boxplot of metric distribution across categories.
    We limit to the most frequent categories to keep plots readable.
    """
    # Keep most frequent categories
    top = df[cat_col].value_counts().head(max_cats).index
    sub = df[df[cat_col].isin(top)].copy()

    # Order categories by median metric (lower is better)
    order = (
        sub.groupby(cat_col)[metric_col]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    fig = plt.figure(figsize=(max(12, 0.5 * len(order)), 6))
    ax = fig.add_subplot(111)

    data = [sub.loc[sub[cat_col] == c, metric_col].dropna().values for c in order]
    ax.boxplot(data, tick_labels=[str(c) for c in order], showfliers=False)

    ax.set_title(title)
    ax.set_xlabel(cat_col)
    ax.set_ylabel(metric_col + " (lower is better)")
    ax.tick_params(axis="x", rotation=45)

    save_fig(fig, outpath)


def two_by_two_boxplot(df: pd.DataFrame, metric_col: str, title: str, outpath: Path):
    """
    Make a 2x2 categorical view PCA(on/off) x RFF(on/off) with boxplots.
    """
    # Build four groups
    groups = [
        ("PCA=off, RFF=off", (False, False)),
        ("PCA=on,  RFF=off", (True,  False)),
        ("PCA=off, RFF=on",  (False, True)),
        ("PCA=on,  RFF=on",  (True,  True)),
    ]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    data = []
    labels = []
    for name, (p_on, r_on) in groups:
        vals = df[(df["pca_on"] == p_on) & (df["rff_on"] == r_on)][metric_col].dropna().values
        if len(vals) == 0:
            # Keep an empty placeholder to preserve layout
            vals = np.array([np.nan])
        data.append(vals)
        labels.append(name)

    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(metric_col + " (lower is better)")
    ax.tick_params(axis="x", rotation=20)

    save_fig(fig, outpath)


def heatmap_rff_nc_sigma(df: pd.DataFrame, metric_col: str, title: str, outpath: Path):
    """
    Create a heatmap-like plot for (n_components x sigma) using median metric.
    Implemented with matplotlib imshow to avoid seaborn dependency.
    """
    sub = df[df["rff_on"] & df["rff_n_components"].notna() & df["rff_sigma"].notna()].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="rff_n_components",
        columns="rff_sigma",
        values=metric_col,
        aggfunc="median"
    )

    # Sort axes for readability
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    fig = plt.figure(figsize=(max(10, 0.6 * pivot.shape[1]), max(6, 0.5 * pivot.shape[0])))
    ax = fig.add_subplot(111)

    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("rff_sigma")
    ax.set_ylabel("rff_n_components")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])

    # Add numeric annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label=f"median {metric_col} (lower is better)")
    save_fig(fig, outpath)


def scatter_rff_params(df: pd.DataFrame, metric_col: str, outdir: Path):
    """
    Scatter plots for RFF hyperparameters:
    - metric vs n_components
    - metric vs sigma (log-scale sigma can be useful, but we keep it linear by default)
    """
    sub = df[df["rff_on"] & df["rff_n_components"].notna() & df["rff_sigma"].notna()].copy()
    if sub.empty:
        return

    # metric vs n_components
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.scatter(sub["rff_n_components"].values, sub[metric_col].values)
    ax.set_title(f"{metric_col} vs RFF n_components")
    ax.set_xlabel("rff_n_components")
    ax.set_ylabel(metric_col + " (lower is better)")
    save_fig(fig, outdir / f"scatter_{metric_col}__rff_n_components.png")

    # metric vs sigma
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.scatter(sub["rff_sigma"].values, sub[metric_col].values)
    ax.set_title(f"{metric_col} vs RFF sigma")
    ax.set_xlabel("rff_sigma")
    ax.set_ylabel(metric_col + " (lower is better)")
    save_fig(fig, outdir / f"scatter_{metric_col}__rff_sigma.png")


def boxplot_pca_adapt_fraction(df: pd.DataFrame, metric_col: str, outpath: Path):
    """
    If PCAAdaptive fractions exist (e.g., 0.10n, 0.25n), plot their metric distribution.
    """
    sub = df[(df["pca_family"] == "pca_adapt") & df["pca_fraction"].notna()].copy()
    if sub.empty:
        return

    # Treat fractions as categories
    sub["pca_fraction_str"] = sub["pca_fraction"].map(lambda x: f"{x:.3f}")
    order = (
        sub.groupby("pca_fraction_str")[metric_col]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    fig = plt.figure(figsize=(max(10, 0.7 * len(order)), 6))
    ax = fig.add_subplot(111)
    data = [sub.loc[sub["pca_fraction_str"] == c, metric_col].dropna().values for c in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title(f"{metric_col} distribution for PCAAdaptive fractions")
    ax.set_xlabel("pca_fraction (as fraction of n_samples)")
    ax.set_ylabel(metric_col + " (lower is better)")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, outpath)


# ==============================
# Main
# ==============================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--csv",
        type=str,
        default="best_tabpfn_per_dataset.csv",
        help="Path to best_tabpfn_per_dataset.csv (MOD7 output)."
    )

    p.add_argument(
        "--outdir",
        type=str,
        default="Results/tabpfn_mod7_best_preproc_viz",
        help="Output directory for plots."
    )

    p.add_argument(
        "--metric",
        type=str,
        default="final_test_nrmse",
        choices=["final_test_nrmse", "val_nrmse"],
        help="Which metric to analyze. final_test_nrmse is recommended."
    )

    p.add_argument(
        "--max_categories",
        type=int,
        default=25,
        help="Max number of categories shown in boxplots to keep plots readable."
    )

    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "dataset",
        "best_simple_preproc",
        "best_standardization",
        "best_pca",
        "best_rff",
        "val_nrmse",
        "final_test_nrmse",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    metric_col = args.metric
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col]).copy()

    # Parse PCA and RFF fields into more analysis-friendly columns
    pca_parsed = df["best_pca"].apply(parse_pca)
    df["pca_on"] = pca_parsed.apply(lambda t: bool(t[0]))
    df["pca_family"] = pca_parsed.apply(lambda t: t[1])
    df["pca_fraction"] = pca_parsed.apply(lambda t: t[2])

    rff_parsed = df["best_rff"].apply(parse_rff)
    df["rff_on"] = rff_parsed.apply(lambda t: bool(t[0]))
    df["rff_n_components"] = rff_parsed.apply(lambda t: t[1])
    df["rff_sigma"] = rff_parsed.apply(lambda t: t[2])

    # Save a lightweight summary table for quick inspection
    summary = {
        "n_datasets": int(df.shape[0]),
        "metric": metric_col,
        "median_metric": float(df[metric_col].median()),
        "mean_metric": float(df[metric_col].mean()),
        "pca_on_rate": float(df["pca_on"].mean()),
        "rff_on_rate": float(df["rff_on"].mean()),
    }
    pd.DataFrame([summary]).to_csv(outdir / "summary_overall.csv", index=False)

    # ==========================
    # 1) Frequency (what is often selected)
    # ==========================
    bar_count(
        df["best_simple_preproc"],
        title="Selection frequency: best_simple_preproc",
        xlabel="best_simple_preproc",
        ylabel="count",
        outpath=outdir / "counts_best_simple_preproc.png",
        top_k=50
    )

    bar_count(
        df["best_standardization"],
        title="Selection frequency: best_standardization",
        xlabel="best_standardization",
        ylabel="count",
        outpath=outdir / "counts_best_standardization.png",
        top_k=50
    )

    bar_count(
        df["best_pca"],
        title="Selection frequency: best_pca",
        xlabel="best_pca",
        ylabel="count",
        outpath=outdir / "counts_best_pca.png",
        top_k=50
    )

    bar_count(
        df["best_rff"],
        title="Selection frequency: best_rff",
        xlabel="best_rff",
        ylabel="count",
        outpath=outdir / "counts_best_rff.png",
        top_k=50
    )

    # PCA on/off and RFF on/off frequencies
    bar_count(
        df["pca_on"].map({True: "PCA_ON", False: "PCA_OFF"}),
        title="Selection frequency: PCA on/off",
        xlabel="PCA",
        ylabel="count",
        outpath=outdir / "counts_pca_onoff.png",
        top_k=10
    )

    bar_count(
        df["rff_on"].map({True: "RFF_ON", False: "RFF_OFF"}),
        title="Selection frequency: RFF on/off",
        xlabel="RFF",
        ylabel="count",
        outpath=outdir / "counts_rff_onoff.png",
        top_k=10
    )

    # ==========================
    # 2) Performance distribution by category
    # ==========================
    boxplot_by_category(
        df, "best_simple_preproc", metric_col,
        title=f"{metric_col} by best_simple_preproc (top frequent categories)",
        outpath=outdir / f"box_{metric_col}__best_simple_preproc.png",
        max_cats=args.max_categories
    )

    boxplot_by_category(
        df, "best_standardization", metric_col,
        title=f"{metric_col} by best_standardization (top frequent categories)",
        outpath=outdir / f"box_{metric_col}__best_standardization.png",
        max_cats=args.max_categories
    )

    boxplot_by_category(
        df, "best_pca", metric_col,
        title=f"{metric_col} by best_pca (top frequent categories)",
        outpath=outdir / f"box_{metric_col}__best_pca.png",
        max_cats=args.max_categories
    )

    # PCA family and RFF on/off (more compact)
    boxplot_by_category(
        df, "pca_family", metric_col,
        title=f"{metric_col} by pca_family",
        outpath=outdir / f"box_{metric_col}__pca_family.png",
        max_cats=10
    )

    boxplot_by_category(
        df.assign(rff_on_str=df["rff_on"].map({True: "RFF_ON", False: "RFF_OFF"})),
        "rff_on_str",
        metric_col,
        title=f"{metric_col} by RFF on/off",
        outpath=outdir / f"box_{metric_col}__rff_onoff.png",
        max_cats=10
    )

    # 2x2 PCA x RFF interaction
    two_by_two_boxplot(
        df,
        metric_col=metric_col,
        title=f"{metric_col} for PCA(on/off) x RFF(on/off)",
        outpath=outdir / f"box_{metric_col}__pca_x_rff.png"
    )

    # ==========================
    # 3) Parameter-specific views (when PCA/RFF are enabled)
    # ==========================
    # PCAAdaptive fraction (if present)
    boxplot_pca_adapt_fraction(
        df,
        metric_col=metric_col,
        outpath=outdir / f"box_{metric_col}__pca_adapt_fraction.png"
    )

    # RFF parameter diagnostics
    scatter_rff_params(df, metric_col=metric_col, outdir=outdir)
    heatmap_rff_nc_sigma(
        df,
        metric_col=metric_col,
        title=f"Median {metric_col} for RFF (n_components x sigma)",
        outpath=outdir / f"heatmap_median_{metric_col}__rff_nc_x_sigma.png"
    )

    # Also export a compact table with medians per choice (useful to read quickly)
    # This is often the fastest way to see "often good vs often bad".
    agg_rows = []

    def add_group_stats(group_col: str):
        g = (
            df.groupby(group_col)[metric_col]
            .agg(["count", "median", "mean"])
            .sort_values("median")
            .reset_index()
        )
        g["group"] = group_col
        g = g.rename(columns={group_col: "choice"})
        return g[["group", "choice", "count", "median", "mean"]]

    agg_rows.append(add_group_stats("best_simple_preproc"))
    agg_rows.append(add_group_stats("best_standardization"))
    agg_rows.append(add_group_stats("best_pca"))
    agg_rows.append(add_group_stats("best_rff"))
    agg_rows.append(add_group_stats("pca_family"))
    agg_rows.append(add_group_stats("pca_on"))
    agg_rows.append(add_group_stats("rff_on"))

    agg = pd.concat(agg_rows, axis=0, ignore_index=True)
    agg.to_csv(outdir / f"summary_median_{metric_col}_by_groups.csv", index=False)

    print(f"✅ Done. Outputs saved to: {outdir}")
    print(f" - summary_overall.csv")
    print(f" - summary_median_{metric_col}_by_groups.csv")
    print(f" - multiple PNG figures")


if __name__ == "__main__":
    main()