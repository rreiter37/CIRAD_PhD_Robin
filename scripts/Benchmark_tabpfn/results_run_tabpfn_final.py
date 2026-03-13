#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a consolidated results table from pipeline_tabpfn_final.py outputs,
and create a "wins" barplot per preprocessing component.

Expected files per dataset (in --results_dir):
- <dataset>__best_config.json
- <dataset>__final_predictions.csv   (optional, used to compute test RMSE if y_true exists)

Outputs:
- summary_table.csv
- wins_by_preprocessing.csv
- wins_by_preprocessing.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

def read_csv_strict(path: Path) -> pd.DataFrame:
    """Read CSV with the project convention: ';' separator and '.' decimal."""
    return pd.read_csv(path, sep=";", decimal=".")


def safe_float(x: Any) -> Optional[float]:
    """Convert to float when possible, otherwise return None."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def format_preprocessing(best_cfg: Dict[str, Any]) -> str:
    """
    Convert best_config dict into a compact string representation.

    Example:
      {"scaler":"None","baseline":"ASLSBaseline","simple":"SNV","pca":"None"}
    -> "scaler=None | baseline=ASLSBaseline | simple=SNV | pca=None"
    """
    keys = ["scaler", "baseline", "simple", "pca"]
    parts = []
    for k in keys:
        if k in best_cfg:
            parts.append(f"{k}={best_cfg.get(k)}")
    # Include unexpected keys deterministically
    extra_keys = sorted(set(best_cfg.keys()) - set(keys))
    for k in extra_keys:
        parts.append(f"{k}={best_cfg.get(k)}")
    return " | ".join(parts)


def parse_best_config_json(path: Path) -> Tuple[str, Optional[float], Dict[str, Any], str]:
    """
    Returns:
      dataset_name, rmse_fold, best_cfg_dict, preprocessing_str
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    dataset_name = str(d.get("dataset", path.name.split("__best_config.json")[0]))

    # CV score (RMSE for regression) is stored as best_mean_score in your outputs
    rmse_fold = safe_float(d.get("best_mean_score"))
    if rmse_fold is None:
        fold_scores = d.get("best_fold_scores", None)
        if isinstance(fold_scores, list) and len(fold_scores) > 0:
            vals = [safe_float(v) for v in fold_scores]
            vals = [v for v in vals if v is not None]
            if len(vals) > 0:
                rmse_fold = float(np.mean(vals))

    best_cfg = d.get("best_config", {})
    if not isinstance(best_cfg, dict):
        best_cfg = {}

    preprocessing_str = format_preprocessing(best_cfg)
    return dataset_name, rmse_fold, best_cfg, preprocessing_str


def parse_final_predictions_csv(path: Path) -> Optional[float]:
    """
    Compute RMSE test from <dataset>__final_predictions.csv when y_true is available.
    If y_true is missing, return None.
    """
    df = read_csv_strict(path)

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return None
    if df["y_true"].isna().all():
        return None

    return compute_rmse(df["y_true"].to_numpy(), df["y_pred"].to_numpy())


def discover_datasets(results_dir: Path) -> List[str]:
    """Discover dataset names by scanning for *__best_config.json files."""
    names = []
    for p in results_dir.glob("*__best_config.json"):
        stem = p.name.replace("__best_config.json", "")
        if stem:
            names.append(stem)
    return sorted(set(names))


def value_or_unknown(x: Any) -> str:
    """Convert a config value to a printable string."""
    if x is None:
        return "None"
    s = str(x)
    return s if s.strip() else "None"


def compute_wins_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count wins per preprocessing component.
    Each dataset contributes exactly 1 win to its chosen scaler/baseline/simple/pca option.
    """
    out_rows = []

    for category in ["scaler", "baseline", "simple", "pca"]:
        if category not in summary_df.columns:
            continue
        counts = summary_df[category].fillna("None").astype(str).value_counts(dropna=False)
        for option, wins in counts.items():
            out_rows.append(
                {
                    "category": category,
                    "option": option,
                    "wins": int(wins),
                }
            )

    wins_df = pd.DataFrame(out_rows)
    if not wins_df.empty:
        wins_df = wins_df.sort_values(["category", "wins", "option"], ascending=[True, False, True]).reset_index(drop=True)
    return wins_df


def plot_wins_barplot(wins_df: pd.DataFrame, output_path: Path) -> None:
    """
    Create a single figure with 4 panels (scaler/baseline/simple/pca).
    """
    categories = ["scaler", "baseline", "simple", "pca"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    for i, cat in enumerate(categories):
        ax = axes[i]
        sub = wins_df[wins_df["category"] == cat].copy()
        if sub.empty:
            ax.set_title(f"{cat} (no data)")
            ax.axis("off")
            continue

        # Keep a readable order: most wins first
        sub = sub.sort_values(["wins", "option"], ascending=[False, True])

        x = np.arange(len(sub))
        ax.bar(x, sub["wins"].to_numpy())

        ax.set_title(f"Wins by {cat}")
        ax.set_ylabel("Number of datasets (wins)")
        ax.set_xticks(x)

        # Rotate labels for readability
        ax.set_xticklabels(sub["option"].tolist(), rotation=45, ha="right")

        # Add value labels on bars
        for xi, wi in zip(x, sub["wins"].to_numpy()):
            ax.text(xi, wi + 0.02, str(int(wi)), ha="center", va="bottom", fontsize=9)

    # Hide any unused axes (in case categories list changes)
    for j in range(len(categories), len(axes)):
        axes[j].axis("off")

    fig.suptitle("TabPFN preprocessing wins (best config per dataset)", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing <dataset>__best_config.json and <dataset>__final_predictions.csv files.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save summary CSV + wins CSV + figure (default: --results_dir).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(str(results_dir))

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = discover_datasets(results_dir)

    rows: List[Dict[str, Any]] = []

    for ds in dataset_names:
        best_json = results_dir / f"{ds}__best_config.json"
        pred_csv = results_dir / f"{ds}__final_predictions.csv"

        dataset_name, rmse_fold, best_cfg, preprocessing_str = parse_best_config_json(best_json)

        # Extract the 4 preprocessing components as separate columns (useful for wins plots)
        scaler = value_or_unknown(best_cfg.get("scaler", None))
        baseline = value_or_unknown(best_cfg.get("baseline", None))
        simple = value_or_unknown(best_cfg.get("simple", None))
        pca = value_or_unknown(best_cfg.get("pca", None))

        rmse_test = None
        if pred_csv.exists():
            try:
                rmse_test = parse_final_predictions_csv(pred_csv)
            except Exception:
                rmse_test = None

        rows.append(
            {
                "dataset_name": dataset_name,
                "rmse_fold": rmse_fold,
                "rmse_test": rmse_test,
                "preprocessing": preprocessing_str,
                "scaler": scaler,
                "baseline": baseline,
                "simple": simple,
                "pca": pca,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("dataset_name").reset_index(drop=True)

    # 1) Save main summary CSV
    summary_csv = output_dir / "summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 2) Compute + save wins table
    wins_df = compute_wins_table(summary_df)
    wins_csv = output_dir / "wins_by_preprocessing.csv"
    wins_df.to_csv(wins_csv, index=False)

    # 3) Plot wins figure
    wins_fig = output_dir / "wins_by_preprocessing.png"
    if not wins_df.empty:
        plot_wins_barplot(wins_df, wins_fig)

    print(f"✅ Wrote summary: {summary_csv}")
    print(f"✅ Wrote wins:    {wins_csv}")
    if wins_df.empty:
        print("⚠️ Wins table is empty (no preprocessing columns found).")
    else:
        print(f"✅ Wrote figure:  {wins_fig}")

    # Console preview
    print("\nSummary (head):")
    print(summary_df.head(20).to_string(index=False))
    print("\nWins (head):")
    print(wins_df.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
