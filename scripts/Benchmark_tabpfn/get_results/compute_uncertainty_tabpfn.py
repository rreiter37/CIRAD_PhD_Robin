#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_uncertainty_tabpfn.py

Aggregate the per-sample predictive distributions produced by
``reinfer_tabpfn_distribution.py`` into dataset-level and benchmark-level
uncertainty metrics, for the TabPFN-Raw vs TabPFN-opt comparison
(manuscript_ACA, "Influence of preprocessing").

For each (model, dataset) it computes, from the native TabPFN predictive
distribution:

- **Calibration** (reliability): empirical coverage of the central prediction
  interval at a grid of nominal levels (derived from the symmetric quantile
  pairs). The mean empirical coverage across datasets, per nominal level, is the
  reliability curve plotted in panel (b) of the figure.
- **Sharpness**: normalized mean prediction-interval width (NMPIW = mean width /
  std of the training target), at the 50/80/90/95% levels.
- **Proper scores**: normalized CRPS (= mean CRPS / std of the training target)
  and mean NLL.
- **Miscalibration**: mean absolute deviation between empirical and nominal
  coverage across the nominal grid (lower is better).

It then performs a *paired* Raw-vs-opt comparison (datasets present for both
variants) with a Wilcoxon signed-rank test on sharpness, normalized CRPS, and
miscalibration, mirroring the statistical rigor used elsewhere in the paper.

Outputs (written to ``--output-dir``):
- ``uncertainty_datasetwise.csv`` : one row per (model, dataset).
- ``reliability_curve.csv``       : per (model, nominal level) mean/IQR coverage.
- ``uncertainty_summary.json``    : benchmark-level summary + paired tests.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:  # noqa: BLE001 - scipy optional at import time
    wilcoxon = None  # type: ignore


#: Standard central-interval levels highlighted in summaries.
HEADLINE_LEVELS = (0.50, 0.80, 0.90, 0.95)


# ==============================
# Quantile-column helpers
# ==============================

def parse_quantile_columns(columns: List[str]) -> Dict[float, str]:
    """Map quantile level -> column name for columns of the form ``q0.050``."""
    out: Dict[float, str] = {}
    for col in columns:
        if col.startswith("q"):
            try:
                out[round(float(col[1:]), 3)] = col
            except ValueError:
                continue
    return out


def central_levels(taus: List[float]) -> List[Tuple[float, float, float]]:
    """Build central-interval triplets (nominal_p, tau_low, tau_high).

    For each lower quantile ``tau < 0.5`` whose symmetric partner ``1 - tau`` is
    also available, the nominal central coverage is ``p = 1 - 2 * tau``.
    """
    available = set(taus)
    triplets: List[Tuple[float, float, float]] = []
    for tau in sorted(t for t in taus if t < 0.5):
        hi = round(1.0 - tau, 3)
        if hi in available:
            triplets.append((round(1.0 - 2.0 * tau, 3), tau, hi))
    return sorted(triplets)


# ==============================
# Per-dataset aggregation
# ==============================

def aggregate_one(
    df: pd.DataFrame, qmap: Dict[float, str], levels: List[Tuple[float, float, float]],
    y_train_std: float,
) -> Dict[str, object]:
    """Compute calibration/sharpness/score metrics for one (model, dataset)."""
    y = df["y_true"].to_numpy(dtype=np.float64)
    valid = np.isfinite(y)
    y = y[valid]
    scale = y_train_std if (y_train_std and y_train_std > 0) else 1.0

    picp_by_level: Dict[float, float] = {}
    nmpiw_by_level: Dict[float, float] = {}
    for p, tau_lo, tau_hi in levels:
        lo = df[qmap[tau_lo]].to_numpy(dtype=np.float64)[valid]
        hi = df[qmap[tau_hi]].to_numpy(dtype=np.float64)[valid]
        picp_by_level[p] = float(np.mean((y >= lo) & (y <= hi))) if y.size else np.nan
        nmpiw_by_level[p] = float(np.mean(hi - lo) / scale)

    # Mean absolute deviation between empirical and nominal coverage.
    miscal = float(np.mean([abs(picp_by_level[p] - p) for p, _, _ in levels])) \
        if levels else np.nan

    crps = df["crps"].to_numpy(dtype=np.float64)
    nll = df["nll"].to_numpy(dtype=np.float64)
    row: Dict[str, object] = {
        "crps_norm": float(np.nanmean(crps) / scale),
        "mean_crps": float(np.nanmean(crps)),
        "mean_nll": float(np.nanmean(nll)),
        "miscal": miscal,
        "n_test": int(y.size),
    }
    for p in HEADLINE_LEVELS:
        row[f"picp@{int(p * 100)}"] = picp_by_level.get(round(p, 3), np.nan)
        row[f"nmpiw@{int(p * 100)}"] = nmpiw_by_level.get(round(p, 3), np.nan)
    # Keep the full coverage vector for the reliability curve.
    row["_picp_by_level"] = picp_by_level
    return row


# ==============================
# Main
# ==============================

def load_meta(output_dir: Path, preds_dir: Path) -> pd.DataFrame:
    """Load the meta table (for y_train_std) or rebuild a minimal one."""
    meta_path = output_dir / "uncertainty_meta.csv"
    if meta_path.exists():
        return pd.read_csv(meta_path)
    # Minimal fallback: y_train_std unknown -> 1.0 (NMPIW/CRPS unnormalized).
    rows = []
    for f in sorted(glob.glob(str(preds_dir / "*.parquet"))):
        d = pd.read_parquet(f, columns=["model", "dataset"]).iloc[0]
        rows.append({"model": d["model"], "dataset": d["dataset"], "y_train_std": np.nan})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=str, default="Results/tabpfn_uncertainty",
                   help="Directory containing preds/ and uncertainty_meta.csv.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Where to write aggregated tables (default: --input-dir).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = in_dir / "preds"

    meta = load_meta(in_dir, preds_dir)
    std_lookup = {
        (str(r["model"]), str(r["dataset"])): float(r.get("y_train_std", np.nan))
        for _, r in meta.iterrows()
    }

    files = sorted(glob.glob(str(preds_dir / "*.parquet")))
    if not files:
        raise SystemExit(f"No prediction parquet files found in {preds_dir}")

    dataset_rows: List[Dict[str, object]] = []
    reliability_records: List[Dict[str, object]] = []
    levels_ref: Optional[List[Tuple[float, float, float]]] = None

    for f in files:
        df = pd.read_parquet(f)
        model = str(df["model"].iloc[0])
        dataset = str(df["dataset"].iloc[0])
        database = str(df["database"].iloc[0]) if "database" in df else ""
        qmap = parse_quantile_columns(list(df.columns))
        levels = central_levels(sorted(qmap.keys()))
        levels_ref = levels_ref or levels

        # Prefer the training-target scale embedded in the parquet (self-contained);
        # fall back to the meta table if an older parquet lacks the column.
        if "y_train_std" in df.columns and np.isfinite(df["y_train_std"].iloc[0]):
            y_std = float(df["y_train_std"].iloc[0])
        else:
            y_std = std_lookup.get((model, dataset), np.nan)
        agg = aggregate_one(df, qmap, levels, y_std)
        picp_by_level = agg.pop("_picp_by_level")  # type: ignore

        agg.update({"model": model, "database": database, "dataset": dataset,
                    "y_train_std": y_std})
        dataset_rows.append(agg)

        for p, picp in picp_by_level.items():  # type: ignore
            reliability_records.append({"model": model, "dataset": dataset,
                                        "nominal": p, "empirical": picp})

    dsw = pd.DataFrame(dataset_rows)
    # Order columns: identifiers first.
    id_cols = ["model", "database", "dataset", "n_test", "y_train_std"]
    metric_cols = [c for c in dsw.columns if c not in id_cols]
    dsw = dsw[id_cols + metric_cols]
    dsw.to_csv(out_dir / "uncertainty_datasetwise.csv", index=False)

    # Reliability curve: per (model, nominal) mean + IQR across datasets.
    rel = pd.DataFrame(reliability_records)
    curve = (rel.groupby(["model", "nominal"])["empirical"]
             .agg(mean="mean", median="median",
                  q25=lambda s: s.quantile(0.25),
                  q75=lambda s: s.quantile(0.75), n="count")
             .reset_index())
    curve.to_csv(out_dir / "reliability_curve.csv", index=False)

    # Benchmark-level summary + paired Raw-vs-opt tests.
    summary: Dict[str, object] = {
        "n_pred_files": len(files),
        "n_datasets_per_model": dsw.groupby("model")["dataset"].nunique().to_dict(),
        "per_model_median": {},
        "paired_raw_vs_opt": {},
    }
    for model, g in dsw.groupby("model"):
        summary["per_model_median"][model] = {  # type: ignore
            "nmpiw@90": float(g["nmpiw@90"].median()),
            "crps_norm": float(g["crps_norm"].median()),
            "miscal": float(g["miscal"].median()),
            "picp@90": float(g["picp@90"].median()),
        }

    raw = dsw[dsw["model"] == "TabPFN-Raw"].set_index("dataset")
    opt = dsw[dsw["model"] == "TabPFN-opt"].set_index("dataset")
    common = sorted(set(raw.index) & set(opt.index))
    summary["n_paired_datasets"] = len(common)
    if common:
        for metric in ["nmpiw@90", "crps_norm", "miscal", "picp@90"]:
            a = opt.loc[common, metric].to_numpy(dtype=np.float64)
            b = raw.loc[common, metric].to_numpy(dtype=np.float64)
            delta = a - b  # opt - raw
            entry = {"median_opt_minus_raw": float(np.nanmedian(delta)),
                     "median_raw": float(np.nanmedian(b)),
                     "median_opt": float(np.nanmedian(a))}
            mask = np.isfinite(delta) & (delta != 0)
            if wilcoxon is not None and mask.sum() >= 5:
                try:
                    stat, pval = wilcoxon(a[np.isfinite(delta)], b[np.isfinite(delta)])
                    entry["wilcoxon_stat"] = float(stat)
                    entry["wilcoxon_p"] = float(pval)
                except Exception:  # noqa: BLE001
                    pass
            summary["paired_raw_vs_opt"][metric] = entry  # type: ignore

    with open(out_dir / "uncertainty_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Console digest.
    print(f"Aggregated {len(files)} prediction files "
          f"({dsw.groupby('model')['dataset'].nunique().to_dict()}).")
    print(f"Paired datasets (Raw & opt): {len(common)}")
    print("\nPer-model medians:")
    print(json.dumps(summary["per_model_median"], indent=2))
    if common:
        print("\nPaired opt - raw (median delta, Wilcoxon p):")
        for k, v in summary["paired_raw_vs_opt"].items():  # type: ignore
            p = v.get("wilcoxon_p")
            print(f"  {k:10s}: {v['median_opt_minus_raw']:+.4f}"
                  + (f"  (p={p:.3g})" if p is not None else ""))
    print(f"\nWritten: uncertainty_datasetwise.csv, reliability_curve.csv, "
          f"uncertainty_summary.json -> {out_dir}")


if __name__ == "__main__":
    main()
