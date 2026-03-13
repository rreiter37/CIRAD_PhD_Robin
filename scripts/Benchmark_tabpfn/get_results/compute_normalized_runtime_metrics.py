#!/usr/bin/env python3
"""
compute_runtime_normalized_metrics.py

Compute runtime normalization metrics for model x dataset calibration results.

This script is designed for benchmarking settings where:
- total calibration time is available per dataset x model,
- different model families evaluate different numbers of preprocessing configs,
- execution environments may differ (CPU, GPU, parallel CPU, etc.).

The script produces:
1. observed raw runtime metrics,
2. config-normalized runtime metrics,
3. optional hardware-aware normalized runtime metrics.

Important note
--------------
The hardware-aware normalization implemented here is intentionally conservative.
It should be interpreted as a comparative indicator, not as a perfectly
hardware-independent runtime reconstruction.

Expected inputs
---------------
- master_results.parquet or master_results.csv
- optional runtime_metadata.parquet/csv
- optional model_configs.parquet/csv

Expected outputs
----------------
- runtime_normalized.parquet / .csv
- runtime_model_summary.parquet / .csv
- runtime_task_summary.parquet / .csv
- runtime_errors.csv

All code comments are in English, as requested.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# I/O helpers
# -------------------------------------------------------------------------

def read_table(path):
    """Read a parquet or CSV file."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def ensure_dir(path):
    """Create output directory if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


def first_existing_column(df, candidates):
    """Return the first existing column among candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute runtime-normalized metrics from master results."
    )

    parser.add_argument(
        "--master_results",
        required=True,
        help="Path to master_results.parquet or master_results.csv"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where outputs will be written"
    )

    parser.add_argument(
        "--runtime_metadata",
        default=None,
        help=(
            "Optional runtime metadata table (.parquet or .csv) containing "
            "device / parallelization / worker information"
        )
    )
    parser.add_argument(
        "--model_configs",
        default=None,
        help=(
            "Optional table mapping each model to its number of evaluated configurations. "
            "Expected columns: model, n_configs"
        )
    )

    parser.add_argument("--dataset_col", default="dataset", help="Dataset column name")
    parser.add_argument("--model_col", default="model", help="Model column name")
    parser.add_argument("--task_col", default="task", help="Task column name")

    parser.add_argument(
        "--runtime_col",
        default=None,
        help=(
            "Total runtime column in master_results. If omitted, the script tries common names."
        )
    )
    parser.add_argument(
        "--n_configs_col",
        default=None,
        help=(
            "Optional column in master_results containing the number of tested configurations."
        )
    )

    parser.add_argument(
        "--device_col",
        default=None,
        help="Optional device column in runtime_metadata or master_results"
    )
    parser.add_argument(
        "--parallel_col",
        default=None,
        help="Optional parallelization flag column in runtime_metadata or master_results"
    )
    parser.add_argument(
        "--workers_col",
        default=None,
        help="Optional workers count column in runtime_metadata or master_results"
    )

    parser.add_argument(
        "--task_filter",
        default="all",
        choices=["all", "regression", "classification"],
        help="Restrict analysis to one task if needed"
    )

    # Default number of configurations by model name if model_configs is not provided
    parser.add_argument(
        "--default_model_configs",
        nargs="*",
        default=[],
        help=(
            "Optional mapping 'ModelName=n_configs'. "
            "Example: PLS=72 Ridge=72 TabPFN=36 CatBoost=36"
        )
    )

    # Conservative hardware normalization factors
    parser.add_argument(
        "--gpu_factor",
        type=float,
        default=1.0,
        help=(
            "Conservative multiplicative factor applied to GPU runs "
            "when building hardware-aware normalized runtime. "
            "Use 1.0 to keep observed runtimes unchanged."
        )
    )
    parser.add_argument(
        "--parallel_cpu_factor",
        type=float,
        default=1.0,
        help=(
            "Conservative multiplicative factor applied to parallel CPU runs. "
            "Use 1.0 to keep observed runtimes unchanged."
        )
    )
    parser.add_argument(
        "--sequential_cpu_factor",
        type=float,
        default=1.0,
        help=(
            "Conservative multiplicative factor applied to sequential CPU runs. "
            "Use 1.0 to keep observed runtimes unchanged."
        )
    )

    parser.add_argument(
        "--use_worker_adjustment",
        action="store_true",
        help=(
            "If set, divide the hardware-aware runtime by an approximate worker factor "
            "for parallel CPU runs. This remains heuristic."
        )
    )
    parser.add_argument(
        "--worker_adjustment_cap",
        type=float,
        default=8.0,
        help=(
            "Maximum effective worker count used in heuristic worker adjustment."
        )
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Column inference
# -------------------------------------------------------------------------

def infer_runtime_col(df, runtime_col=None):
    """Infer the total runtime column."""
    if runtime_col is not None:
        if runtime_col not in df.columns:
            raise ValueError(f"Requested runtime column '{runtime_col}' not found.")
        return runtime_col

    col = first_existing_column(
        df,
        [
            "runtime_total",
            "total_runtime",
            "calibration_time",
            "runtime_seconds",
            "wall_time",
            "elapsed_time",
            "fit_time",
            "time_total_sec",
            "time_sec",
        ]
    )
    if col is None:
        raise ValueError(
            "Could not infer the runtime column. "
            "Use --runtime_col explicitly."
        )
    return col


def infer_aux_col(df, requested, candidates):
    """Infer an auxiliary column with optional explicit override."""
    if requested is not None:
        return requested if requested in df.columns else None
    return first_existing_column(df, candidates)


# -------------------------------------------------------------------------
# Mapping helpers
# -------------------------------------------------------------------------

def parse_default_model_configs(pairs):
    """
    Parse CLI mappings like:
        PLS=72 Ridge=72 TabPFN=36 CatBoost=36
    """
    mapping = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(
                f"Invalid --default_model_configs entry '{item}'. "
                "Expected format ModelName=n_configs"
            )
        model_name, value = item.split("=", 1)
        mapping[str(model_name).strip()] = int(value)
    return mapping


def load_model_config_mapping(model_configs_path, model_col):
    """
    Load model -> n_configs mapping from a table.
    Expected columns:
        - model_col
        - n_configs
    """
    df = read_table(model_configs_path)
    if model_col not in df.columns:
        raise ValueError(
            f"Column '{model_col}' not found in model_configs table."
        )
    if "n_configs" not in df.columns:
        raise ValueError(
            "Column 'n_configs' not found in model_configs table."
        )

    mapping = {}
    for _, row in df.iterrows():
        if pd.notna(row[model_col]) and pd.notna(row["n_configs"]):
            mapping[str(row[model_col]).strip()] = int(row["n_configs"])
    return mapping


# -------------------------------------------------------------------------
# Merge helpers
# -------------------------------------------------------------------------

def merge_runtime_metadata(master_df, runtime_df, dataset_col, model_col):
    """
    Merge runtime metadata into the master table using dataset x model.
    Runtime metadata is expected to have at least:
        dataset_col, model_col
    """
    required = [dataset_col, model_col]
    missing = [c for c in required if c not in runtime_df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in runtime_metadata: {missing}"
        )

    # Keep the first row per dataset x model if duplicates exist
    runtime_df = runtime_df.copy()
    runtime_df = runtime_df.drop_duplicates(subset=[dataset_col, model_col], keep="first")

    merged = master_df.merge(
        runtime_df,
        on=[dataset_col, model_col],
        how="left",
        suffixes=("", "_runtime_meta")
    )
    return merged


# -------------------------------------------------------------------------
# Hardware normalization helpers
# -------------------------------------------------------------------------

def normalize_device_string(device_value):
    """Normalize device labels."""
    if pd.isna(device_value):
        return "unknown"

    text = str(device_value).strip().lower()

    if "gpu" in text or "cuda" in text:
        return "gpu"
    if "cpu" in text:
        return "cpu"
    return text


def normalize_parallel_flag(value):
    """Normalize a parallelization flag."""
    if pd.isna(value):
        return "unknown"

    if isinstance(value, (bool, np.bool_)):
        return "parallel" if value else "sequential"

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "parallel", "multiprocessing", "joblib"}:
        return "parallel"
    if text in {"false", "0", "no", "n", "sequential", "serial"}:
        return "sequential"

    return text


def safe_worker_factor(workers, cap=8.0):
    """
    Build a conservative effective worker factor.

    We do not assume linear speedup. We use:
        effective = min(max(workers, 1), cap)
    and later only divide by that value if requested.

    This is intentionally simple and conservative.
    """
    if pd.isna(workers):
        return 1.0

    try:
        w = float(workers)
    except Exception:
        return 1.0

    if w < 1:
        return 1.0

    return min(w, cap)


def compute_hardware_factor(device_norm, parallel_norm, gpu_factor, parallel_cpu_factor, sequential_cpu_factor):
    """
    Compute a simple multiplicative factor for hardware-aware normalization.

    Interpretation:
    - values > 1 inflate the observed runtime,
    - values = 1 keep the observed runtime unchanged.

    By default all factors are 1.0, meaning no adjustment.
    """
    if device_norm == "gpu":
        return gpu_factor

    if device_norm == "cpu":
        if parallel_norm == "parallel":
            return parallel_cpu_factor
        if parallel_norm == "sequential":
            return sequential_cpu_factor

    return 1.0


# -------------------------------------------------------------------------
# Core computation
# -------------------------------------------------------------------------

def build_runtime_normalized_table(
    df,
    dataset_col,
    model_col,
    task_col,
    runtime_col,
    n_configs_col,
    device_col,
    parallel_col,
    workers_col,
    default_model_configs,
    gpu_factor,
    parallel_cpu_factor,
    sequential_cpu_factor,
    use_worker_adjustment,
    worker_adjustment_cap,
):
    """
    Build dataset-level runtime normalization metrics.

    Output columns include:
    - runtime_total_observed
    - n_configs
    - runtime_per_config
    - runtime_per_sample_train (if available)
    - runtime_hw_factor
    - runtime_hardware_aware
    - runtime_per_config_hardware_aware
    """
    rows = []
    errors = []

    required = [dataset_col, model_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Optional train size column for extra normalization
    n_train_col = first_existing_column(
        df,
        ["n_samples_train", "n_train", "train_size"]
    )

    for _, row in df.iterrows():
        dataset_name = row.get(dataset_col)
        model_name = row.get(model_col)
        task_name = row.get(task_col, np.nan)

        if pd.isna(dataset_name) or pd.isna(model_name):
            errors.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "task": task_name,
                    "status": "missing_dataset_or_model",
                }
            )
            continue

        runtime_value = row.get(runtime_col, np.nan)
        if pd.isna(runtime_value):
            errors.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "task": task_name,
                    "status": "missing_runtime",
                }
            )
            continue

        try:
            runtime_total = float(runtime_value)
        except Exception:
            errors.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "task": task_name,
                    "status": "runtime_not_numeric",
                    "runtime_value": runtime_value,
                }
            )
            continue

        if runtime_total < 0:
            errors.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "task": task_name,
                    "status": "negative_runtime",
                    "runtime_value": runtime_total,
                }
            )
            continue

        # Resolve number of configurations
        n_configs = np.nan
        if n_configs_col is not None and n_configs_col in row.index and pd.notna(row[n_configs_col]):
            try:
                n_configs = int(row[n_configs_col])
            except Exception:
                n_configs = np.nan

        if pd.isna(n_configs):
            n_configs = default_model_configs.get(str(model_name).strip(), np.nan)

        if pd.isna(n_configs) or n_configs <= 0:
            errors.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "task": task_name,
                    "status": "missing_or_invalid_n_configs",
                }
            )
            continue

        # Hardware metadata
        device_norm = "unknown"
        parallel_norm = "unknown"
        workers = np.nan

        if device_col is not None and device_col in row.index:
            device_norm = normalize_device_string(row[device_col])

        if parallel_col is not None and parallel_col in row.index:
            parallel_norm = normalize_parallel_flag(row[parallel_col])

        if workers_col is not None and workers_col in row.index:
            workers = row[workers_col]

        runtime_hw_factor = compute_hardware_factor(
            device_norm=device_norm,
            parallel_norm=parallel_norm,
            gpu_factor=gpu_factor,
            parallel_cpu_factor=parallel_cpu_factor,
            sequential_cpu_factor=sequential_cpu_factor,
        )

        runtime_hardware_aware = runtime_total * runtime_hw_factor

        worker_factor = 1.0
        if use_worker_adjustment and device_norm == "cpu" and parallel_norm == "parallel":
            worker_factor = safe_worker_factor(workers, cap=worker_adjustment_cap)
            runtime_hardware_aware = runtime_hardware_aware * worker_factor

        runtime_per_config = runtime_total / n_configs
        runtime_per_config_hardware_aware = runtime_hardware_aware / n_configs

        n_samples_train = np.nan
        runtime_per_train_sample = np.nan
        runtime_per_train_sample_per_config = np.nan

        if n_train_col is not None and pd.notna(row.get(n_train_col, np.nan)):
            try:
                n_samples_train = float(row[n_train_col])
            except Exception:
                n_samples_train = np.nan

            if pd.notna(n_samples_train) and n_samples_train > 0:
                runtime_per_train_sample = runtime_total / n_samples_train
                runtime_per_train_sample_per_config = runtime_per_config / n_samples_train

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "task": task_name,
                "runtime_total_observed": runtime_total,
                "n_configs": int(n_configs),
                "runtime_per_config": runtime_per_config,
                "device_norm": device_norm,
                "parallel_norm": parallel_norm,
                "workers": workers,
                "runtime_hw_factor": runtime_hw_factor,
                "worker_adjustment_factor": worker_factor,
                "runtime_hardware_aware": runtime_hardware_aware,
                "runtime_per_config_hardware_aware": runtime_per_config_hardware_aware,
                "n_samples_train": n_samples_train,
                "runtime_per_train_sample": runtime_per_train_sample,
                "runtime_per_train_sample_per_config": runtime_per_train_sample_per_config,
                "status": "ok",
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(errors)


def build_model_summary(runtime_df):
    """Aggregate runtime metrics at the model level."""
    if runtime_df.empty:
        return pd.DataFrame()

    summary = (
        runtime_df.groupby(["task", "model"], as_index=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            mean_runtime_total_observed=("runtime_total_observed", "mean"),
            median_runtime_total_observed=("runtime_total_observed", "median"),
            mean_runtime_per_config=("runtime_per_config", "mean"),
            median_runtime_per_config=("runtime_per_config", "median"),
            mean_runtime_hardware_aware=("runtime_hardware_aware", "mean"),
            median_runtime_hardware_aware=("runtime_hardware_aware", "median"),
            mean_runtime_per_config_hardware_aware=("runtime_per_config_hardware_aware", "mean"),
            median_runtime_per_config_hardware_aware=("runtime_per_config_hardware_aware", "median"),
            mean_n_configs=("n_configs", "mean"),
            gpu_share=("device_norm", lambda s: np.mean(pd.Series(s) == "gpu")),
            parallel_cpu_share=(
                "parallel_norm",
                lambda s: np.mean(pd.Series(s) == "parallel")
            ),
        )
    )

    return summary.sort_values(
        ["task", "mean_runtime_per_config_hardware_aware", "model"],
        ascending=[True, True, True]
    ).reset_index(drop=True)


def build_task_summary(runtime_df):
    """Aggregate runtime metrics at the task level."""
    if runtime_df.empty:
        return pd.DataFrame()

    summary = (
        runtime_df.groupby(["task"], as_index=False)
        .agg(
            n_rows=("dataset", "size"),
            n_datasets=("dataset", "nunique"),
            n_models=("model", "nunique"),
            mean_runtime_total_observed=("runtime_total_observed", "mean"),
            median_runtime_total_observed=("runtime_total_observed", "median"),
            mean_runtime_per_config=("runtime_per_config", "mean"),
            median_runtime_per_config=("runtime_per_config", "median"),
            mean_runtime_hardware_aware=("runtime_hardware_aware", "mean"),
            median_runtime_hardware_aware=("runtime_hardware_aware", "median"),
            mean_runtime_per_config_hardware_aware=("runtime_per_config_hardware_aware", "mean"),
            median_runtime_per_config_hardware_aware=("runtime_per_config_hardware_aware", "median"),
        )
    )

    return summary.sort_values("task").reset_index(drop=True)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    master_df = read_table(args.master_results)

    if args.task_filter != "all":
        if args.task_col not in master_df.columns:
            raise ValueError(f"Column '{args.task_col}' not found in master_results.")
        master_df = master_df[
            master_df[args.task_col].astype(str).str.lower() == args.task_filter
        ].copy()

    # Merge runtime metadata if provided
    if args.runtime_metadata is not None:
        runtime_meta_df = read_table(args.runtime_metadata)
        master_df = merge_runtime_metadata(
            master_df=master_df,
            runtime_df=runtime_meta_df,
            dataset_col=args.dataset_col,
            model_col=args.model_col,
        )

    runtime_col = infer_runtime_col(master_df, args.runtime_col)

    n_configs_col = None
    if args.n_configs_col is not None:
        if args.n_configs_col not in master_df.columns:
            raise ValueError(f"Column '{args.n_configs_col}' not found in master_results.")
        n_configs_col = args.n_configs_col
    else:
        n_configs_col = first_existing_column(
            master_df,
            ["n_configs", "n_tested_configs", "n_preprocessing_configs", "num_configs"]
        )

    device_col = infer_aux_col(
        master_df,
        args.device_col,
        ["device", "device_type", "compute_device", "hardware_device"]
    )
    parallel_col = infer_aux_col(
        master_df,
        args.parallel_col,
        ["is_parallel", "parallel", "parallel_mode", "cpu_parallel"]
    )
    workers_col = infer_aux_col(
        master_df,
        args.workers_col,
        ["n_workers", "num_workers", "workers", "n_jobs"]
    )

    # Build model -> n_configs mapping
    default_model_configs = parse_default_model_configs(args.default_model_configs)

    if args.model_configs is not None:
        file_mapping = load_model_config_mapping(
            model_configs_path=args.model_configs,
            model_col=args.model_col,
        )
        default_model_configs.update(file_mapping)

    runtime_df, errors_df = build_runtime_normalized_table(
        df=master_df,
        dataset_col=args.dataset_col,
        model_col=args.model_col,
        task_col=args.task_col,
        runtime_col=runtime_col,
        n_configs_col=n_configs_col,
        device_col=device_col,
        parallel_col=parallel_col,
        workers_col=workers_col,
        default_model_configs=default_model_configs,
        gpu_factor=args.gpu_factor,
        parallel_cpu_factor=args.parallel_cpu_factor,
        sequential_cpu_factor=args.sequential_cpu_factor,
        use_worker_adjustment=args.use_worker_adjustment,
        worker_adjustment_cap=args.worker_adjustment_cap,
    )

    model_summary_df = build_model_summary(runtime_df)
    task_summary_df = build_task_summary(runtime_df)

    settings = {
        "master_results": str(args.master_results),
        "runtime_metadata": args.runtime_metadata,
        "model_configs": args.model_configs,
        "dataset_col": args.dataset_col,
        "model_col": args.model_col,
        "task_col": args.task_col,
        "runtime_col": runtime_col,
        "n_configs_col": n_configs_col,
        "device_col": device_col,
        "parallel_col": parallel_col,
        "workers_col": workers_col,
        "task_filter": args.task_filter,
        "default_model_configs": default_model_configs,
        "gpu_factor": args.gpu_factor,
        "parallel_cpu_factor": args.parallel_cpu_factor,
        "sequential_cpu_factor": args.sequential_cpu_factor,
        "use_worker_adjustment": args.use_worker_adjustment,
        "worker_adjustment_cap": args.worker_adjustment_cap,
    }
    settings_json = json.dumps(settings, ensure_ascii=False)

    for out_df in [runtime_df, model_summary_df, task_summary_df]:
        if not out_df.empty:
            out_df["settings_json"] = settings_json

    if errors_df.empty:
        errors_df = pd.DataFrame(
            columns=["dataset", "model", "task", "status", "settings_json"]
        )
    else:
        errors_df["settings_json"] = settings_json

    runtime_df.to_parquet(output_dir / "runtime_normalized.parquet", index=False)
    runtime_df.to_csv(output_dir / "runtime_normalized.csv", index=False)

    model_summary_df.to_parquet(output_dir / "runtime_model_summary.parquet", index=False)
    model_summary_df.to_csv(output_dir / "runtime_model_summary.csv", index=False)

    task_summary_df.to_parquet(output_dir / "runtime_task_summary.parquet", index=False)
    task_summary_df.to_csv(output_dir / "runtime_task_summary.csv", index=False)

    errors_df.to_csv(output_dir / "runtime_errors.csv", index=False)

    print(f"Saved: {output_dir / 'runtime_normalized.parquet'}")
    print(f"Saved: {output_dir / 'runtime_normalized.csv'}")
    print(f"Saved: {output_dir / 'runtime_model_summary.parquet'}")
    print(f"Saved: {output_dir / 'runtime_model_summary.csv'}")
    print(f"Saved: {output_dir / 'runtime_task_summary.parquet'}")
    print(f"Saved: {output_dir / 'runtime_task_summary.csv'}")
    print(f"Saved: {output_dir / 'runtime_errors.csv'}")


if __name__ == "__main__":
    main()