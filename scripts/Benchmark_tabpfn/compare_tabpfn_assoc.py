#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic benchmark comparison between:
  - assoc_pp_model (PLS, Ridge, LGBM, CNN)
  - Multiple TabPFN workspaces (Raw, RFF, PCA, etc.)

Metric:
  NRMSE = RMSE_test / (max(Y_val) - min(Y_val))

All comments are intentionally written in English.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# CLI arguments
# ==============================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tabpfn_workspaces",
    nargs="+",
    required=True,
    help="List of TabPFN workspace directories to compare"
)

parser.add_argument(
    "--tabpfn_labels",
    nargs="+",
    required=True,
    help="Human-readable labels for each TabPFN workspace"
)

parser.add_argument(
    "--outdir",
    type=str,
    default="Results/comp_assoc_tabpfn",
    help="Output directory"
)

args = parser.parse_args()

if len(args.tabpfn_workspaces) != len(args.tabpfn_labels):
    raise ValueError("tabpfn_workspaces and tabpfn_labels must have the same length")

TABPFN_WORKSPACES = [
    (Path(ws), label)
    for ws, label in zip(args.tabpfn_workspaces, args.tabpfn_labels)
]


# ==============================
# Paths & configuration
# ==============================

ASSOC_ROOT = Path("Results/assoc_pp_model/per_dataset")
DATA_ROOT = Path("Data/Regression")
OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

BASE_MODELS = ["PLS", "Ridge", "LGBM", "CNN"]
MODELS_ORDER = BASE_MODELS + args.tabpfn_labels


# ==============================
# Utility functions
# ==============================

def normalize(name: str) -> str:
    """Normalize dataset names for fuzzy matching."""
    return name.lower().replace("-", "_").replace(" ", "_")


def find_tabpfn_meta(dataset: str, root: Path):
    """Find the TabPFN .meta.parquet file corresponding to a dataset."""
    norm = normalize(dataset)
    for p in root.glob("*.meta.parquet"):
        if normalize(p.stem).startswith(norm) or norm.startswith(normalize(p.stem)):
            return p
    return None


def load_Yval_range(dataset: str):
    """Load Yval.csv and compute its value range."""
    path = DATA_ROOT / dataset / "Yval.csv"
    if not path.exists():
        return None

    y = pd.read_csv(path, sep=";", header=None).iloc[:, 0]
    y = pd.to_numeric(y, errors="coerce").dropna()
    if y.empty:
        return None

    r = y.max() - y.min()
    return r if r > 0 else None


def extract_tabpfn_rmse(meta_path: Path) -> float:
    """Extract mean test RMSE from a TabPFN meta.parquet file."""
    df = pd.read_parquet(meta_path)
    test_df = df[
        (df["partition"] == "test") &
        (df["fold_id"].isin(["avg", "w_avg", "0", "1", "2"]))
    ]
    rmses = [json.loads(s)["test"]["rmse"] for s in test_df["scores"]]
    return float(np.mean(rmses))


def extract_assoc_best_per_model(csv_path: Path):
    """
    Extract the best (minimum) score per model across all preprocessings
    from assoc_pp_model result CSVs.
    """
    df = pd.read_csv(csv_path)
    if "Model" not in df.columns:
        raise ValueError("Missing 'Model' column")

    prep_cols = [c for c in df.columns if c.lower() not in ["model", "id"]]
    results = {}

    for model in df["Model"].unique():
        sub = df[df["Model"] == model][prep_cols]
        sub = sub.apply(pd.to_numeric, errors="coerce")
        if sub.isna().all().all():
            continue

        clean = model.replace("_reg", "").replace("_Reg", "")
        results[clean] = float(sub.min().min())

    return results


# ==============================
# Load all results
# ==============================

rows = []

for dataset_dir in ASSOC_ROOT.iterdir():
    if not dataset_dir.is_dir():
        continue

    dataset = dataset_dir.name
    y_range = load_Yval_range(dataset)
    if y_range is None:
        continue

    # ---- assoc_pp_model ----
    for csv_file in dataset_dir.glob("results_*_dynamic_batch_size.csv"):
        try:
            scores = extract_assoc_best_per_model(csv_file)
        except Exception:
            continue

        for model, rmse in scores.items():
            rows.append({
                "dataset": dataset,
                "model": model,
                "NRMSE": rmse
            })

    # ---- TabPFN workspaces (generic) ----
    for ws_path, ws_label in TABPFN_WORKSPACES:
        meta = find_tabpfn_meta(dataset, ws_path)
        if meta is None:
            continue

        rmse = extract_tabpfn_rmse(meta)
        rows.append({
            "dataset": dataset,
            "model": ws_label,
            "NRMSE": rmse / y_range
        })


df_all = pd.DataFrame(rows)

if df_all.empty:
    raise RuntimeError("No results collected. Check paths and dataset names.")

# ==============================
# Keep only datasets present in ALL TabPFN workspaces
# ==============================

# Collect datasets available for each TabPFN label
datasets_per_tabpfn = {}

for label in args.tabpfn_labels:
    datasets_per_tabpfn[label] = set(
        df_all[df_all["model"] == label]["dataset"].unique()
    )

# Compute strict intersection
common_datasets = set.intersection(*datasets_per_tabpfn.values())

if not common_datasets:
    raise RuntimeError(
        "No dataset has results for ALL TabPFN workspaces. "
        "Cannot perform a fair comparison."
    )

# Filter the full dataframe
df_all = df_all[df_all["dataset"].isin(common_datasets)].copy()


# ==============================
# Pivot table
# ==============================

pivot_df = (
    df_all
    .pivot_table(
        index="model",
        columns="dataset",
        values="NRMSE",
        aggfunc="mean"
    )
    .reindex(MODELS_ORDER)
)

pivot_df.to_csv(OUTDIR / "nrmse_table.csv")


# ==============================
# Heatmap (values + best highlight)
# ==============================

fig_w = max(30, 1.4 * pivot_df.shape[1])
fig_h = max(6, 1.2 * pivot_df.shape[0])

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

sns.heatmap(
    pivot_df,
    ax=ax,
    cmap="viridis",
    linewidths=0.3,
    linecolor="white",
    annot=True,
    fmt=".3f",
    annot_kws={"fontsize": 7},
    cbar_kws={"label": "NRMSE (lower is better)"}
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    fontsize=8
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=12,
    fontweight="bold"
)

# Highlight best model per dataset
for j, ds in enumerate(pivot_df.columns):
    col = pivot_df[ds]
    if col.isna().all():
        continue
    i = pivot_df.index.get_loc(col.idxmin())
    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=2))

plt.title("Model comparison (NRMSE)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTDIR / "heatmap_nrmse.png", dpi=300)
plt.close()


# ==============================
# Binary heatmap (best vs others)
# ==============================

binary = np.zeros_like(pivot_df.values, dtype=int)

for j, ds in enumerate(pivot_df.columns):
    col = pivot_df[ds]
    if col.isna().all():
        continue
    binary[pivot_df.index.get_loc(col.idxmin()), j] = 1

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#f5b7b1", "#b6f2c2"])

ax.imshow(binary, aspect="auto", cmap=cmap)

ax.set_xticks(np.arange(len(pivot_df.columns)))
ax.set_yticks(np.arange(len(pivot_df.index)))

ax.set_xticklabels(pivot_df.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(pivot_df.index, fontsize=12, fontweight="bold")

for i in range(pivot_df.shape[0]):
    for j in range(pivot_df.shape[1]):
        val = pivot_df.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

ax.set_title("Best model per dataset (NRMSE)", fontsize=14)
ax.set_xlim(-0.5, pivot_df.shape[1] - 0.5)
ax.set_ylim(pivot_df.shape[0] - 0.5, -0.5)

plt.tight_layout()
plt.savefig(OUTDIR / "binary_heatmap_nrmse.png", dpi=300)
plt.close()


# ==============================
# Wins / losses
# ==============================

wins = {m: 0 for m in pivot_df.index}
losses = {m: 0 for m in pivot_df.index}

for ds in pivot_df.columns:
    col = pivot_df[ds].dropna()
    if col.empty:
        continue
    wins[col.idxmin()] += 1
    losses[col.idxmax()] += 1

pd.DataFrame({
    "model": pivot_df.index,
    "wins": [wins[m] for m in pivot_df.index],
    "losses": [losses[m] for m in pivot_df.index]
}).to_csv(OUTDIR / "wins_losses_per_model.csv", index=False)


# ==============================
# Mean rank
# ==============================

rank_rows = []
for ds in pivot_df.columns:
    col = pivot_df[ds]
    ranks = col.rank(ascending=True, method="average")
    for model, r in ranks.items():
        rank_rows.append({"dataset": ds, "model": model, "rank": r})

df_ranks = pd.DataFrame(rank_rows)

df_mean_rank = (
    df_ranks.groupby("model")
    .agg(mean_rank=("rank", "mean"))
    .sort_values("mean_rank")
)

df_mean_rank.to_csv(OUTDIR / "global_model_ranking.csv")


# ==============================
# Critical Difference diagram
# ==============================

def plot_cd(mean_ranks: dict, n_datasets: int):
    """Plot a Nemenyi Critical Difference diagram."""
    import numpy as np
    import matplotlib.pyplot as plt

    models = list(mean_ranks.keys())
    ranks = np.array(list(mean_ranks.values()))
    k = len(models)

    q_alpha = 2.569  # alpha=0.05
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_datasets))

    order = np.argsort(ranks)
    ranks = ranks[order]
    models = [models[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(min(ranks) - 0.5, max(ranks) + 0.5)
    ax.set_ylim(0, 1)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Mean rank (lower is better)")

    y = 0.6
    for r, m in zip(ranks, models):
        ax.plot(r, y, "o", color="black")
        ax.text(r, y + 0.08, m, ha="center", va="bottom")

    ax.plot([min(ranks), min(ranks) + cd], [0.15, 0.15], lw=3, color="black")
    ax.text(min(ranks) + cd / 2, 0.18, f"CD = {cd:.2f}", ha="center")

    ax.set_title("Critical Difference Diagram (Nemenyi)")

    return fig


fig = plot_cd(
    df_mean_rank["mean_rank"].to_dict(),
    pivot_df.shape[1]
)

fig.savefig(OUTDIR / "critical_difference_diagram.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"All results saved to {OUTDIR}")
