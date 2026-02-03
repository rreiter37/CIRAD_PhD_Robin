#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_tabpfn_assoc_with_search_best.py

Compare regression results between:
  - assoc_pp_model (PLS, Ridge, LGBM, CNN) best-over-preprocessings
  - TabPFN workspaces (*.meta.parquet)
  - "search_best_tabpfn_preproc_MOD7.py" outputs (best_tabpfn_per_dataset.csv)

Metric:
  NRMSE = RMSE_test / (max(Y_val) - min(Y_val))
For MOD7 search:
  We use `final_test_nrmse` directly because MOD7 defines it as RMSE_test / range(Yval).

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
# CLI
# ==============================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tabpfn_workspaces",
        nargs="*",
        default=[],
        help="List of TabPFN workspace directories to compare (each contains *.meta.parquet)"
    )

    parser.add_argument(
        "--tabpfn_labels",
        nargs="*",
        default=[],
        help="Human-readable labels for each TabPFN workspace"
    )

    parser.add_argument(
        "--search_best_csvs",
        nargs="*",
        default=[],
        help="List of MOD7 'best_tabpfn_per_dataset.csv' files to include"
    )

    parser.add_argument(
        "--search_best_labels",
        nargs="*",
        default=[],
        help="Human-readable labels for each MOD7 best CSV"
    )

    parser.add_argument(
        "--assoc_root",
        type=str,
        default="Results/assoc_pp_model/per_dataset",
        help="Root folder for assoc_pp_model per-dataset results"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="Data/Regression",
        help="Root folder containing datasets with Yval.csv"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="Results/comp_assoc_tabpfn_searchbest",
        help="Output directory"
    )

    parser.add_argument(
        "--strict_intersection",
        action="store_true",
        help="If set, keep only datasets present in ALL sources (assoc + each TabPFN ws + each MOD7 csv)."
    )

    return parser.parse_args()


# ==============================
# Utilities
# ==============================

def normalize(name: str) -> str:
    """Normalize dataset names for fuzzy matching."""
    return name.lower().replace("-", "_").replace(" ", "_")


def find_tabpfn_meta(dataset: str, root: Path):
    """Find a TabPFN .meta.parquet file corresponding to a dataset (fuzzy match)."""
    norm = normalize(dataset)
    for p in root.glob("*.meta.parquet"):
        stem = p.stem  # e.g., DatasetName.meta -> stem includes ".meta" if double suffix, but glob matches anyway
        if normalize(stem).startswith(norm) or norm.startswith(normalize(stem)):
            return p
    return None


def load_yval_range(dataset: str, data_root: Path):
    """Load Yval.csv and compute its value range."""
    path = data_root / dataset / "Yval.csv"
    if not path.exists():
        return None

    # Your datasets use ';' separators in multiple scripts; we keep the same convention.
    y = pd.read_csv(path, sep=";", header=None).iloc[:, 0]
    y = pd.to_numeric(y, errors="coerce").dropna()
    if y.empty:
        return None

    r = float(y.max() - y.min())
    return r if r > 0 else None


def extract_tabpfn_rmse(meta_path: Path) -> float:
    """
    Extract mean test RMSE from a TabPFN meta.parquet file.
    This follows the logic from compare_tabpfn_assoc.py.
    """
    df = pd.read_parquet(meta_path)
    test_df = df[
        (df["partition"] == "test") &
        (df["fold_id"].isin(["avg", "w_avg", "0", "1", "2"]))
    ]
    if test_df.empty:
        raise ValueError(f"No test rows found in meta: {meta_path}")

    rmses = []
    for s in test_df["scores"].tolist():
        payload = json.loads(s)
        # Expected format: payload["test"]["rmse"]
        rmses.append(payload["test"]["rmse"])
    return float(np.mean(rmses))


def extract_assoc_best_per_model(csv_path: Path):
    """
    Extract the best (minimum) score per model across all preprocessings
    from assoc_pp_model result CSVs.
    """
    df = pd.read_csv(csv_path)
    if "Model" not in df.columns:
        raise ValueError(f"Missing 'Model' column in {csv_path}")

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


def load_mod7_best_csv(csv_path: Path):
    """
    Load MOD7 best results:
    expected columns include: dataset, final_test_nrmse
    (per MOD7 description, final_test_nrmse is normalized by range(Yval)).
    """
    df = pd.read_csv(csv_path)
    required = {"dataset", "final_test_nrmse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MOD7 CSV missing columns {missing}: {csv_path}")

    out = df[["dataset", "final_test_nrmse"]].copy()
    out["final_test_nrmse"] = pd.to_numeric(out["final_test_nrmse"], errors="coerce")
    out = out.dropna(subset=["final_test_nrmse"])
    return out


# ==============================
# Main
# ==============================

def main():
    args = parse_args()

    tabpfn_workspaces = [Path(p) for p in args.tabpfn_workspaces]
    tabpfn_labels = list(args.tabpfn_labels)

    if len(tabpfn_workspaces) != len(tabpfn_labels):
        raise ValueError("--tabpfn_workspaces and --tabpfn_labels must have the same length")

    mod7_csvs = [Path(p) for p in args.search_best_csvs]
    mod7_labels = list(args.search_best_labels)

    if len(mod7_csvs) != len(mod7_labels):
        raise ValueError("--search_best_csvs and --search_best_labels must have the same length")

    assoc_root = Path(args.assoc_root)
    data_root = Path(args.data_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Base models (assoc)
    base_models = ["PLS", "Ridge", "LGBM", "CNN"]

    # Combined order of rows in the heatmap
    models_order = base_models + tabpfn_labels + mod7_labels

    rows = []

    # ------------------------------
    # 1) Collect assoc_pp_model rows
    # ------------------------------
    if assoc_root.exists():
        for dataset_dir in assoc_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            y_range = load_yval_range(dataset, data_root)
            if y_range is None:
                continue

            for csv_file in dataset_dir.glob("results_*_dynamic_batch_size.csv"):
                try:
                    scores = extract_assoc_best_per_model(csv_file)
                except Exception:
                    continue

                for model, rmse in scores.items():
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "NRMSE": float(rmse)  # assoc already stores NRMSE in your script logic
                    })
    else:
        print(f"Warning: assoc_root not found: {assoc_root}")

    # ------------------------------------------------
    # 2) Collect TabPFN workspace meta.parquet rows
    # ------------------------------------------------
    for ws_path, ws_label in zip(tabpfn_workspaces, tabpfn_labels):
        if not ws_path.exists():
            print(f"Warning: TabPFN workspace not found: {ws_path}")
            continue

        # We iterate datasets from DATA root for a stable list
        if not data_root.exists():
            raise RuntimeError(f"data_root not found: {data_root}")

        for ds_dir in data_root.iterdir():
            if not ds_dir.is_dir():
                continue
            dataset = ds_dir.name
            y_range = load_yval_range(dataset, data_root)
            if y_range is None:
                continue

            meta = find_tabpfn_meta(dataset, ws_path)
            if meta is None:
                continue

            try:
                rmse = extract_tabpfn_rmse(meta)
            except Exception:
                continue

            rows.append({
                "dataset": dataset,
                "model": ws_label,
                "NRMSE": float(rmse) / float(y_range)
            })

    # -----------------------------------------
    # 3) Collect MOD7 "best per dataset" rows
    # -----------------------------------------
    for csv_path, label in zip(mod7_csvs, mod7_labels):
        if not csv_path.exists():
            print(f"Warning: MOD7 CSV not found: {csv_path}")
            continue

        df_mod7 = load_mod7_best_csv(csv_path)
        for _, r in df_mod7.iterrows():
            rows.append({
                "dataset": str(r["dataset"]),
                "model": label,
                "NRMSE": float(r["final_test_nrmse"])
            })

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        raise RuntimeError("No results collected. Check paths, dataset names, and file availability.")

    # -----------------------------------------
    # Optional: strict intersection of datasets
    # -----------------------------------------
    if args.strict_intersection:
        sources_labels = tabpfn_labels + mod7_labels

        if len(sources_labels) == 0:
            raise RuntimeError(
                "strict_intersection was requested, but no TabPFN workspaces or MOD7 CSVs were provided."
            )

        datasets_per_source = {}
        for lbl in sources_labels:
            datasets_per_source[lbl] = set(df_all[df_all["model"] == lbl]["dataset"].unique())

        common = set.intersection(*datasets_per_source.values())
        if not common:
            raise RuntimeError(
                "No dataset has results for ALL provided sources (TabPFN workspaces + MOD7 CSVs)."
            )

        df_all = df_all[df_all["dataset"].isin(common)].copy()

    # -----------------------------------------
    # Pivot table
    # -----------------------------------------
    pivot_df = (
        df_all
        .pivot_table(index="model", columns="dataset", values="NRMSE", aggfunc="mean")
        .reindex(models_order)
    )

    pivot_df.to_csv(outdir / "nrmse_table.csv")

    # -----------------------------------------
    # Heatmap 1: raw values
    # -----------------------------------------
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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title("NRMSE comparison (assoc_pp_model vs TabPFN vs MOD7 best)")
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_nrmse.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------
    # Heatmap 2: highlight best per dataset
    # -----------------------------------------
    # We create a mask that keeps only the best (minimum) per column visible in one overlay.
    best_mask = pivot_df.copy()
    for col in best_mask.columns:
        col_vals = best_mask[col]
        if col_vals.isna().all():
            continue
        m = col_vals.min(skipna=True)
        best_mask[col] = (col_vals == m)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # Base heatmap
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
    # Overlay: outline best cells
    # We draw rectangles manually for best cells to avoid colormap conflicts.
    for i, model in enumerate(pivot_df.index):
        for j, dataset in enumerate(pivot_df.columns):
            if bool(best_mask.loc[model, dataset]) is True:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", linewidth=2.0))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title("NRMSE heatmap (best per dataset outlined)")
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_best_highlight.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------
    # Per-dataset winner table
    # -----------------------------------------
    winners = []
    for ds in pivot_df.columns:
        s = pivot_df[ds].dropna()
        if s.empty:
            continue
        winner = s.idxmin()
        winners.append({"dataset": ds, "winner": winner, "best_nrmse": float(s.min())})

    df_winners = pd.DataFrame(winners)
    df_winners.to_csv(outdir / "rank_per_dataset.csv", index=False)

    print(f"✅ Saved: {outdir / 'nrmse_table.csv'}")
    print(f"✅ Saved: {outdir / 'heatmap_nrmse.png'}")
    print(f"✅ Saved: {outdir / 'heatmap_best_highlight.png'}")
    print(f"✅ Saved: {outdir / 'rank_per_dataset.csv'}")


if __name__ == "__main__":
    main()
