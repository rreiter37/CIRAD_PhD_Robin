#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full benchmark comparison between:
  - assoc_pp_model (PLS, Ridge, LGBM, CNN)
  - TabPFN Raw
  - TabPFN RFF

Metric:
  NRMSE = RMSE_test / (max(Y_val) - min(Y_val))
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata


# ==============================
# Paths & configuration
# ==============================

ASSOC_ROOT = Path("Results/assoc_pp_model/per_dataset")
TABPFN_RAW_ROOT = Path("wk_tabpfn_raw")
TABPFN_RFF_ROOT = Path("wk_tabpfn_rff")
DATA_ROOT = Path("Data/Data/Regression")  # <-- FIXED
OUTDIR = Path("Results/comp_assoc_tabpfn")
OUTDIR.mkdir(parents=True, exist_ok=True)

MODELS_ORDER = ["PLS", "Ridge", "LGBM", "CNN", "TabPFN Raw", "TabPFN RFF"]


# ==============================
# Utility functions
# ==============================

def normalize(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def find_tabpfn_meta(dataset: str, root: Path):
    norm = normalize(dataset)
    for p in root.glob("*.meta.parquet"):
        if normalize(p.stem).startswith(norm) or norm.startswith(normalize(p.stem)):
            return p
    return None


def load_Yval_range(dataset: str):
    path = DATA_ROOT / dataset / "Yval.csv"
    if not path.exists():
        print(f"[WARN] Missing Yval.csv for {dataset}")
        return None

    y = pd.read_csv(path, sep=";", header=None).iloc[:, 0]
    y = pd.to_numeric(y, errors="coerce").dropna()
    if y.empty:
        return None

    r = y.max() - y.min()
    return r if r > 0 else None


def extract_tabpfn_rmse(meta_path):
    df = pd.read_parquet(meta_path)
    test_df = df[
        (df["partition"] == "test") &
        (df["fold_id"].isin(["avg", "w_avg", "0", "1", "2"]))
    ]
    rmses = [json.loads(s)["test"]["rmse"] for s in test_df["scores"]]
    return float(np.mean(rmses))


def extract_assoc_best_per_model(csv_path):
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
        except Exception as e:
            print(f"[WARN] Failed to read {csv_file}: {e}")
            continue

        for model, nrmse in scores.items():
            rows.append({
                "dataset": dataset,
                "model": model,
                "NRMSE": nrmse
            })

    # ---- TabPFN Raw ----
    raw_meta = find_tabpfn_meta(dataset, TABPFN_RAW_ROOT)
    if raw_meta:
        rmse = extract_tabpfn_rmse(raw_meta)
        rows.append({
            "dataset": dataset,
            "model": "TabPFN Raw",
            "NRMSE": rmse / y_range
        })

    # ---- TabPFN RFF ----
    rff_meta = find_tabpfn_meta(dataset, TABPFN_RFF_ROOT)
    if rff_meta:
        rmse = extract_tabpfn_rmse(rff_meta)
        rows.append({
            "dataset": dataset,
            "model": "TabPFN RFF",
            "NRMSE": rmse / y_range
        })


df_all = pd.DataFrame(rows)

if df_all.empty:
    raise RuntimeError("No results collected. Check paths and dataset names.")


# ==============================
# Keep only datasets with TabPFN
# ==============================

valid_datasets = set(df_all[df_all["model"].isin(["TabPFN Raw", "TabPFN RFF"])]["dataset"])
df_all = df_all[df_all["dataset"].isin(valid_datasets)].copy()

if df_all.empty:
    raise RuntimeError("No dataset contains TabPFN results.")


# ==============================
# Pivot table
# ==============================

pivot_df = df_all.pivot_table(
    index="model",
    columns="dataset",
    values="NRMSE",
    aggfunc="mean"
).reindex(MODELS_ORDER)

pivot_df.to_csv(OUTDIR / "nrmse_table.csv")


# ==============================
# Heatmap (fully readable labels)
# ==============================

import textwrap

def wrap_label(label, width=18):
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))

fig_w = max(30, 1.4 * pivot_df.shape[1])   # big width per dataset
fig_h = max(6, 1.2 * pivot_df.shape[0])

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

sns.heatmap(
    pivot_df,
    ax=ax,
    cmap="viridis",
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Normalized RMSE"},
    annot=True,
    fmt=".3f",
    annot_kws={
        "fontsize": 7,
        "ha": "center",
        "va": "center"
    }
)

wrapped_labels = [wrap_label(ds) for ds in pivot_df.columns]

ax.set_xticklabels(
    wrapped_labels,
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

# Highlight best per dataset
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

def plot_cd(mean_ranks: dict, n_datasets: int, alpha: float = 0.05):
    """
    Proper Critical Difference (Nemenyi) diagram, publication-ready.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    models = list(mean_ranks.keys())
    ranks = np.array(list(mean_ranks.values()))
    k = len(models)

    # Nemenyi critical value (alpha=0.05, large sample approx)
    q_alpha = 2.569
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_datasets))

    # Sort by rank
    order = np.argsort(ranks)
    ranks = ranks[order]
    models = [models[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 3))

    # Axis
    ax.set_xlim(min(ranks) - 0.5, max(ranks) + 0.5)
    ax.set_ylim(0, 1)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Mean rank (lower is better)", fontsize=11)

    # Plot model points and labels
    y = 0.6
    for r, m in zip(ranks, models):
        ax.plot(r, y, "o", color="black")
        ax.text(r, y + 0.08, m, ha="center", va="bottom", fontsize=11)

    # Plot CD bar
    x_start = min(ranks)
    ax.plot([x_start, x_start + cd], [0.15, 0.15], lw=3, color="black")
    ax.text(x_start + cd / 2, 0.18, f"CD = {cd:.2f}", ha="center", fontsize=10)

    # Plot non-significant groups
    y_level = 0.45
    step = 0.07

    for i in range(k):
        j = i
        while j < k and ranks[j] - ranks[i] <= cd:
            j += 1
        if j - i > 1:
            ax.plot(
                [ranks[i], ranks[j - 1]],
                [y_level, y_level],
                lw=4,
                color="black"
            )
            y_level -= step

    ax.set_title("Critical Difference Diagram (Nemenyi test)", fontsize=13)

    return fig

fig = plot_cd(df_mean_rank["mean_rank"].to_dict(), pivot_df.shape[1])
fig.savefig(OUTDIR / "critical_difference_diagram.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"All results saved to {OUTDIR}")
