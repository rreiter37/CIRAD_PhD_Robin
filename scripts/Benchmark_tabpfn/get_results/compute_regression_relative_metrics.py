#!/usr/bin/env python3
"""
Compute regression relative metrics from the master results table.

This script is designed for the benchmark structure previously discussed.
It reads a master results table containing one final row per dataset/model,
filters regression datasets, and computes relative metrics such as:

- iRMSEP_preproc: relative improvement of a preprocessed model over its raw version
- iRMSEP_TabPFN_vs_<baseline>: relative gain/loss of TabPFN against each baseline
- ratio_to_best: RMSEP divided by the best RMSEP on the same dataset
- delta_to_best: RMSEP minus the best RMSEP on the same dataset
- relative_gain_vs_best: relative gap to the best model on the same dataset

The script is intentionally flexible:
- missing models are allowed
- missing raw variants are allowed
- the number of models is not fixed
- different model name conventions can be handled via CLI arguments

Expected master table columns
-----------------------------
Minimum required:
- dataset
- task
- model
- RMSEP

Optional but useful:
- preprocessing_pipeline
- status
- database_name

Important note
--------------
You previously decided to drop the explicit "variant" column from the master table.
Therefore, raw-vs-preprocessed comparisons cannot be inferred automatically unless the
raw rows are detectable from another column. This script supports three strategies:

1. Detect raw rows from the preprocessing pipeline string (default).
2. Detect raw rows from a custom boolean column.
3. Accept a separate raw master table and match it to the best-preprocessed table.

Outputs
-------
- regression_relative_metrics.parquet
- regression_relative_metrics.csv
- regression_relative_metrics_errors.csv
- tabpfn_pairwise_irmsep.parquet
- tabpfn_pairwise_irmsep.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute relative regression metrics from master_results.parquet/csv"
    )
    parser.add_argument(
        "--master_results",
        type=str,
        required=True,
        help="Path to master_results.parquet or master_results.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where output tables will be written",
    )
    parser.add_argument(
        "--raw_master_results",
        type=str,
        default=None,
        help=(
            "Optional separate master table containing raw-spectrum rows. "
            "Useful if raw results are stored separately and the main master table only contains best results."
        ),
    )
    parser.add_argument(
        "--dataset_col",
        type=str,
        default="dataset",
        help="Dataset column name in the master table",
    )
    parser.add_argument(
        "--task_col",
        type=str,
        default="task",
        help="Task column name in the master table",
    )
    parser.add_argument(
        "--model_col",
        type=str,
        default="model",
        help="Model column name in the master table",
    )
    parser.add_argument(
        "--rmsep_col",
        type=str,
        default="RMSEP",
        help="RMSEP column name in the master table",
    )
    parser.add_argument(
        "--preproc_col",
        type=str,
        default="preprocessing_pipeline",
        help="Preprocessing pipeline column used to detect raw rows",
    )
    parser.add_argument(
        "--status_col",
        type=str,
        default="status",
        help="Status column name. Rows with status different from ok/partial can be ignored if desired.",
    )
    parser.add_argument(
        "--keep_partial",
        action="store_true",
        help="Keep rows with status=partial. By default they are kept; this flag is just explicit.",
    )
    parser.add_argument(
        "--drop_non_ok",
        action="store_true",
        help="Drop rows whose status is neither ok nor partial",
    )
    parser.add_argument(
        "--regression_label",
        type=str,
        default="regression",
        help="Value used in the task column for regression datasets",
    )
    parser.add_argument(
        "--tabpfn_name",
        type=str,
        default="TabPFN",
        help="Model name to use as TabPFN reference in pairwise comparisons",
    )
    parser.add_argument(
        "--baseline_names",
        nargs="*",
        default=None,
        help=(
            "Optional list of baseline model names for pairwise TabPFN comparisons. "
            "If omitted, all models except TabPFN are used."
        ),
    )
    parser.add_argument(
        "--raw_detection_mode",
        choices=["pipeline_string", "boolean_column", "disabled"],
        default="pipeline_string",
        help="How to identify raw-spectrum rows in the main master table",
    )
    parser.add_argument(
        "--raw_boolean_col",
        type=str,
        default=None,
        help="Boolean column name to use when --raw_detection_mode=boolean_column",
    )
    parser.add_argument(
        "--raw_pipeline_patterns",
        nargs="*",
        default=["none", "raw", "identity", "passthrough"],
        help=(
            "Case-insensitive tokens used to detect raw preprocessing pipelines. "
            "Applied only when --raw_detection_mode=pipeline_string."
        ),
    )
    parser.add_argument(
        "--group_keys",
        nargs="*",
        default=None,
        help=(
            "Extra keys used to match rows across tables. "
            "Example: --group_keys database_name trait. "
            "The dataset column is always included automatically."
        ),
    )
    return parser.parse_args()


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)

    raise ValueError(f"Unsupported input format for {p}. Use .parquet or .csv")



def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()



def safe_float(x: object) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan



def ensure_columns(df: pd.DataFrame, required: List[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {table_name}: {missing}")



def canonical_group_cols(dataset_col: str, extra_group_keys: Optional[List[str]]) -> List[str]:
    cols = [dataset_col]
    if extra_group_keys:
        for c in extra_group_keys:
            if c not in cols:
                cols.append(c)
    return cols



def is_status_acceptable(status: object) -> bool:
    s = normalize_text(status)
    return s in {"", "ok", "partial"}



def detect_raw_pipeline(value: object, patterns: List[str]) -> bool:
    """
    Detect whether a preprocessing pipeline string looks like a raw-spectrum pipeline.

    The default logic is intentionally permissive:
    - exact None/nan/empty
    - strings such as 'None', 'raw', 'identity'
    - combinations where all steps appear to be 'None'
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True

    s = normalize_text(value)
    if s in {"", "none", "nan", "null", "raw", "identity", "passthrough"}:
        return True

    # Replace common separators and inspect tokens.
    tokens = re.split(r"[|,;+\-_/()\[\]{} ]+", s)
    tokens = [t for t in tokens if t]
    if tokens and all(t in patterns or t == "none" for t in tokens):
        return True

    # Detect repeated None-like pipelines such as "None | None | None".
    if "none" in s:
        cleaned = re.sub(r"none", "", s)
        cleaned = re.sub(r"[|,;+\-_/()\[\]{} ]+", "", cleaned)
        if cleaned == "":
            return True

    return False


# -----------------------------------------------------------------------------
# Core computations
# -----------------------------------------------------------------------------

def prepare_regression_master(df: pd.DataFrame, args: argparse.Namespace, table_name: str) -> pd.DataFrame:
    required = [args.dataset_col, args.task_col, args.model_col, args.rmsep_col]
    ensure_columns(df, required, table_name)

    work = df.copy()
    work[args.task_col] = work[args.task_col].astype(str)
    work[args.model_col] = work[args.model_col].astype(str)
    work[args.rmsep_col] = work[args.rmsep_col].map(safe_float)

    # Filter regression rows only.
    work = work[work[args.task_col].map(normalize_text) == normalize_text(args.regression_label)].copy()

    # Optionally drop problematic statuses.
    if args.drop_non_ok and args.status_col in work.columns:
        work = work[work[args.status_col].map(is_status_acceptable)].copy()

    # Remove rows without usable RMSEP.
    work = work[work[args.rmsep_col].notna()].copy()
    return work



def mark_raw_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    work = df.copy()
    work["is_raw_detected"] = False

    if args.raw_detection_mode == "disabled":
        return work

    if args.raw_detection_mode == "boolean_column":
        if not args.raw_boolean_col:
            raise ValueError("--raw_boolean_col must be provided when raw_detection_mode=boolean_column")
        if args.raw_boolean_col not in work.columns:
            raise KeyError(f"Column not found: {args.raw_boolean_col}")
        work["is_raw_detected"] = work[args.raw_boolean_col].fillna(False).astype(bool)
        return work

    # Default: pipeline_string detection.
    if args.preproc_col not in work.columns:
        # No preprocessing column available: keep all as False.
        return work

    patterns = [normalize_text(p) for p in args.raw_pipeline_patterns]
    work["is_raw_detected"] = work[args.preproc_col].apply(lambda x: detect_raw_pipeline(x, patterns))
    return work



def reduce_to_best_per_group(df: pd.DataFrame, group_cols: List[str], model_col: str, rmsep_col: str) -> pd.DataFrame:
    """
    Keep the best row (lowest RMSEP) for each dataset/model group.

    This is robust to accidental duplicates in the master table.
    """
    if df.empty:
        return df.copy()

    sort_cols = group_cols + [model_col, rmsep_col]
    work = df.sort_values(sort_cols, ascending=[True] * (len(group_cols) + 1) + [True]).copy()
    return work.groupby(group_cols + [model_col], as_index=False, sort=False).first()



def compute_dataset_best_metrics(df: pd.DataFrame, group_cols: List[str], rmsep_col: str) -> pd.DataFrame:
    """Attach best-on-dataset relative metrics for each row."""
    work = df.copy()

    best_rmsep = work.groupby(group_cols)[rmsep_col].transform("min")
    work["best_rmsep_dataset"] = best_rmsep
    work["delta_to_best"] = work[rmsep_col] - best_rmsep
    work["ratio_to_best"] = work[rmsep_col] / best_rmsep

    work["relative_gap_to_best_pct"] = np.where(
        best_rmsep > 0,
        100.0 * (work[rmsep_col] - best_rmsep) / best_rmsep,
        np.nan,
    )

    work["is_best_on_dataset"] = np.isclose(work[rmsep_col], best_rmsep, equal_nan=False)
    return work



def compute_preproc_irmsep(
    best_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    group_cols: List[str],
    model_col: str,
    rmsep_col: str,
) -> pd.DataFrame:
    """
    Compute iRMSEP_preproc for each model/dataset pair.

    Formula:
        100 * (RMSEP_raw - RMSEP_best_preproc) / RMSEP_raw
    """
    if best_df.empty:
        out = best_df.copy()
        out["rmsep_raw_reference"] = np.nan
        out["iRMSEP_preproc"] = np.nan
        out["has_raw_reference"] = False
        return out

    left = best_df.copy()
    right = raw_df[group_cols + [model_col, rmsep_col]].copy()
    right = right.rename(columns={rmsep_col: "rmsep_raw_reference"})

    merged = left.merge(right, on=group_cols + [model_col], how="left")
    merged["has_raw_reference"] = merged["rmsep_raw_reference"].notna()
    merged["iRMSEP_preproc"] = np.where(
        merged["rmsep_raw_reference"] > 0,
        100.0 * (merged["rmsep_raw_reference"] - merged[rmsep_col]) / merged["rmsep_raw_reference"],
        np.nan,
    )
    return merged



def build_tabpfn_pairwise_table(
    df: pd.DataFrame,
    args: argparse.Namespace,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Build one row per dataset x baseline for TabPFN pairwise iRMSEP.

    Formula:
        100 * (RMSEP_baseline - RMSEP_TabPFN) / RMSEP_baseline
    Positive values mean TabPFN is better.
    """
    model_col = args.model_col
    rmsep_col = args.rmsep_col

    if df.empty:
        return pd.DataFrame()

    tabpfn_df = df[df[model_col] == args.tabpfn_name].copy()
    if tabpfn_df.empty:
        return pd.DataFrame()

    if args.baseline_names:
        baseline_names = [m for m in args.baseline_names if m != args.tabpfn_name]
    else:
        baseline_names = [m for m in sorted(df[model_col].dropna().unique()) if m != args.tabpfn_name]

    baseline_df = df[df[model_col].isin(baseline_names)].copy()
    if baseline_df.empty:
        return pd.DataFrame()

    tabpfn_cols = group_cols + [rmsep_col]
    tabpfn_ref = tabpfn_df[group_cols + [rmsep_col]].rename(columns={rmsep_col: "RMSEP_TabPFN"})

    merged = baseline_df.merge(tabpfn_ref, on=group_cols, how="left")
    merged = merged.rename(columns={rmsep_col: "RMSEP_baseline"})
    merged["baseline_model"] = merged[model_col]
    merged["tabpfn_model"] = args.tabpfn_name
    merged["has_tabpfn_reference"] = merged["RMSEP_TabPFN"].notna()

    merged["iRMSEP_TabPFN_vs_baseline"] = np.where(
        merged["RMSEP_baseline"] > 0,
        100.0 * (merged["RMSEP_baseline"] - merged["RMSEP_TabPFN"]) / merged["RMSEP_baseline"],
        np.nan,
    )
    merged["delta_rmsep_tabpfn_minus_baseline"] = merged["RMSEP_TabPFN"] - merged["RMSEP_baseline"]

    keep_cols = group_cols + [
        "baseline_model",
        "tabpfn_model",
        "RMSEP_baseline",
        "RMSEP_TabPFN",
        "iRMSEP_TabPFN_vs_baseline",
        "delta_rmsep_tabpfn_minus_baseline",
        "has_tabpfn_reference",
    ]
    return merged[keep_cols].sort_values(group_cols + ["baseline_model"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_cols = canonical_group_cols(args.dataset_col, args.group_keys)

    master = read_table(args.master_results)
    master = prepare_regression_master(master, args, "master_results")
    master = mark_raw_rows(master, args)

    # Reduce to one best row per dataset/model in case duplicates remain.
    best_main = reduce_to_best_per_group(master, group_cols, args.model_col, args.rmsep_col)

    # Resolve raw-reference table.
    raw_ref_df: Optional[pd.DataFrame] = None
    if args.raw_master_results:
        raw_tbl = read_table(args.raw_master_results)
        raw_tbl = prepare_regression_master(raw_tbl, args, "raw_master_results")
        raw_tbl = mark_raw_rows(raw_tbl, args)
        raw_tbl = raw_tbl[raw_tbl["is_raw_detected"]].copy() if args.raw_detection_mode != "disabled" else raw_tbl.copy()
        raw_ref_df = reduce_to_best_per_group(raw_tbl, group_cols, args.model_col, args.rmsep_col)
    else:
        raw_ref_df = best_main[best_main["is_raw_detected"]].copy()

    # Compute dataset-best relative metrics.
    enriched = compute_dataset_best_metrics(best_main, group_cols, args.rmsep_col)

    # Compute iRMSEP_preproc if possible.
    enriched = compute_preproc_irmsep(enriched, raw_ref_df, group_cols, args.model_col, args.rmsep_col)

    # Build TabPFN pairwise comparisons.
    pairwise = build_tabpfn_pairwise_table(enriched, args, group_cols)

    # Add per-baseline pairwise iRMSEP columns back to the model-level table.
    if not pairwise.empty:
        for baseline in pairwise["baseline_model"].dropna().unique():
            sub = pairwise[pairwise["baseline_model"] == baseline][group_cols + ["iRMSEP_TabPFN_vs_baseline"]].copy()
            col_name = f"iRMSEP_{args.tabpfn_name}_vs_{baseline}"
            sub = sub.rename(columns={"iRMSEP_TabPFN_vs_baseline": col_name})
            enriched = enriched.merge(sub, on=group_cols, how="left")

    # Create a compact error report.
    errors = []
    if enriched.empty:
        errors.append({"issue": "no_regression_rows_found", "source": str(args.master_results)})
    else:
        missing_rmsep = enriched[enriched[args.rmsep_col].isna()]
        if not missing_rmsep.empty:
            errors.append({"issue": "rows_with_missing_rmsep_after_processing", "count": int(len(missing_rmsep))})

        missing_raw = enriched[~enriched["has_raw_reference"]]
        if not missing_raw.empty:
            errors.append({"issue": "rows_without_raw_reference", "count": int(len(missing_raw))})

        if args.tabpfn_name not in set(enriched[args.model_col].dropna().unique()):
            errors.append({"issue": "tabpfn_model_not_found", "tabpfn_name": args.tabpfn_name})

    errors_df = pd.DataFrame(errors)

    # Save outputs.
    enriched = enriched.sort_values(group_cols + [args.model_col]).reset_index(drop=True)
    enriched.to_parquet(output_dir / "regression_relative_metrics.parquet", index=False)
    enriched.to_csv(output_dir / "regression_relative_metrics.csv", index=False)

    if not pairwise.empty:
        pairwise.to_parquet(output_dir / "tabpfn_pairwise_irmsep.parquet", index=False)
        pairwise.to_csv(output_dir / "tabpfn_pairwise_irmsep.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "tabpfn_pairwise_irmsep.csv", index=False)

    errors_df.to_csv(output_dir / "regression_relative_metrics_errors.csv", index=False)

    summary = {
        "n_rows_input": int(len(master)),
        "n_rows_output": int(len(enriched)),
        "n_pairwise_rows": int(len(pairwise)),
        "tabpfn_name": args.tabpfn_name,
        "baseline_names": args.baseline_names,
        "group_cols": group_cols,
        "raw_detection_mode": args.raw_detection_mode,
        "raw_master_results": args.raw_master_results,
    }
    with open(output_dir / "regression_relative_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {output_dir / 'regression_relative_metrics.parquet'}")
    print(f"Saved: {output_dir / 'regression_relative_metrics.csv'}")
    print(f"Saved: {output_dir / 'tabpfn_pairwise_irmsep.csv'}")
    print(f"Saved: {output_dir / 'regression_relative_metrics_errors.csv'}")


if __name__ == "__main__":
    main()
