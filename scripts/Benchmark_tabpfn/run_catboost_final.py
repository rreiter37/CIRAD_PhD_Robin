#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_catboost_final_on_excel.py

Same launcher logic as run_tabpfn_final.py, but calling pipeline_catboost_final.py.

Expected dataset layout (regression by default):
    Data/regression/<Database>/<Dataset>/
        Xtrain.csv, Ytrain.csv, Xtest.csv, Ytest.csv

This launcher:
- Reads DatabaseDetail.xlsx and discovers dataset folders (columns: "Database", "Dataset").
- Skips missing/invalid folders.
- Runs pipeline_catboost_final.py dataset-by-dataset (safer than batching all datasets in one call).
- Writes one log file per dataset and a summary CSV.

All comments are in English (per your requirement).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm


REQUIRED_FILES = ("Xtrain.csv", "Ytrain.csv", "Xtest.csv", "Ytest.csv")


@dataclass
class RunRow:
    database: str
    dataset: str
    folder: str
    status: str
    returncode: int
    elapsed_sec: float
    log_path: str
    cmd: str


def discover_datasets_from_excel(
    xlsx_path: Path,
    data_root: Path,
    databases_filter: Optional[List[str]] = None,
    datasets_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, Path]]:
    """Discover dataset folders from DatabaseDetail.xlsx."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Database detail file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    required_cols = {"Database", "Dataset"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {xlsx_path}: {sorted(missing)}")

    if databases_filter:
        df = df[df["Database"].astype(str).isin(set(databases_filter))]

    if datasets_filter:
        df = df[df["Dataset"].astype(str).isin(set(datasets_filter))]

    out: List[Tuple[str, str, Path]] = []
    seen = set()

    for _, row in df.iterrows():
        db = str(row["Database"]).strip()
        ds = str(row["Dataset"]).strip()
        folder = data_root / db / ds
        key = str(folder)

        if key in seen:
            continue
        seen.add(key)

        missing_files = [f for f in REQUIRED_FILES if not (folder / f).exists()]
        if missing_files:
            continue

        out.append((db, ds, folder))

    return out


def build_pipeline_cmd(
    pipeline_script: Path,
    dataset_folder: Path,
    output_dir: Path,
    task_type: str,
    seed: int,
    n_splits: int,
    n_estimators_search: int,
    n_estimators_final: int,
    parallel: bool,
    n_jobs: int,
    use_tmp_dir: bool,
    extra_args: List[str],
) -> List[str]:
    """Build the command line to call pipeline_catboost_final.py for a single dataset."""
    cmd = [
        sys.executable,
        str(pipeline_script),
        "--datasets",
        str(dataset_folder),
        "--output_dir",
        str(output_dir),
        "--task_type",
        str(task_type),
        "--seed",
        str(int(seed)),
        "--n_splits",
        str(int(n_splits)),
        "--n_estimators_search",
        str(int(n_estimators_search)),
        "--n_estimators_final",
        str(int(n_estimators_final)),
    ]

    if parallel:
        cmd += ["--parallel", "--n_jobs", str(int(n_jobs))]

    if use_tmp_dir:
        cmd += ["--use_tmp_dir"]

    cmd += list(extra_args)
    return cmd


def run_one_dataset(db: str, ds: str, folder: Path, cmd: List[str], logs_dir: Path) -> RunRow:
    """Run one dataset and capture stdout+stderr to a dedicated log file."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{db}__{ds}.log"

    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("CMD:\n")
            f.write(" ".join(shlex.quote(x) for x in cmd) + "\n\n")
            f.flush()

            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
                env=os.environ.copy(),
            )

        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else "FAIL"
        return RunRow(
            database=db,
            dataset=ds,
            folder=str(folder),
            status=status,
            returncode=int(proc.returncode),
            elapsed_sec=float(elapsed),
            log_path=str(log_path),
            cmd=" ".join(shlex.quote(x) for x in cmd),
        )
    except Exception as e:
        elapsed = time.time() - t0
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n[LAUNCHER ERROR]\n")
            f.write(repr(e) + "\n")
        return RunRow(
            database=db,
            dataset=ds,
            folder=str(folder),
            status="LAUNCHER_FAIL",
            returncode=999,
            elapsed_sec=float(elapsed),
            log_path=str(log_path),
            cmd=" ".join(shlex.quote(x) for x in cmd),
        )


def append_run_row_to_csv(summary_csv: Path, row: RunRow) -> None:
    """Append exactly one run row to the summary CSV."""
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = True
    if summary_csv.exists():
        try:
            write_header = summary_csv.stat().st_size == 0
        except OSError:
            write_header = False

    df_one = pd.DataFrame([row.__dict__])
    df_one.to_csv(summary_csv, mode="a", header=write_header, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Dataset discovery
    p.add_argument("--database_detail_xlsx", type=str, default="Data/DatabaseDetail.xlsx",
                   help="Path to DatabaseDetail.xlsx.")
    p.add_argument("--data_root", type=str, default="Data/regression",
                   help="Root folder with datasets: Data/<task>/<Database>/<Dataset>/")
    p.add_argument("--databases", nargs="*", default=None,
                   help="Optional list of database names to include.")
    p.add_argument("--datasets_include", nargs="*", default=None,
                   help="Optional list of dataset names to include.")
    p.add_argument("--after_dataset", type=str, default=None,
                   help="If provided, skip all datasets before this dataset name in DATA_PATH.")

    # Where pipeline_catboost_final.py is located
    p.add_argument("--pipeline_script", type=str, default="scripts/Benchmark_tabpfn/pipeline_catboost_final.py",
                   help="Path to pipeline_catboost_final.py.")

    # Output
    p.add_argument("--output_dir", type=str, default="Results/catboost_reg_final",
                   help="Output directory passed to pipeline_catboost_final.py.")
    p.add_argument("--logs_dir", type=str, default="Results/catboost_reg_final/logs",
                   help="Directory for per-dataset logs.")
    p.add_argument("--summary_csv", type=str, default="Results/catboost_reg_final/summary_runs.csv",
                   help="CSV summarizing launcher runs.")

    # Pipeline args (mirrors pipeline_catboost_final.py)
    p.add_argument("--task_type", type=str, default="regression", choices=["regression", "classification"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_splits", type=int, default=3)
    p.add_argument("--n_estimators_search", type=int, default=200)
    p.add_argument("--n_estimators_final", type=int, default=500)
    p.add_argument("--parallel", action="store_true",
                   help="Forward --parallel to pipeline_catboost_final.py (parallel inside each dataset search).")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Forward --n_jobs to pipeline_catboost_final.py (workers inside each dataset search).")
    p.add_argument("--use_tmp_dir", action="store_true",
                   help="Forward --use_tmp_dir to pipeline_catboost_final.py.")

    # Advanced: forward unknown args to the pipeline script (future-proofing)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra arguments forwarded to pipeline_catboost_final.py (must come last).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    xlsx_path = Path(args.database_detail_xlsx)
    data_root = Path(args.data_root)
    pipeline_script = Path(args.pipeline_script)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    summary_csv = Path(args.summary_csv)

    if not pipeline_script.exists():
        raise FileNotFoundError(f"pipeline script not found: {pipeline_script}")

    datasets = discover_datasets_from_excel(
        xlsx_path=xlsx_path,
        data_root=data_root,
        databases_filter=args.databases,
        datasets_filter=args.datasets_include,
    )

    # AFTER_DATASET logic (same behavior as your TabPFN launcher)
    if args.after_dataset is not None:
        target = str(args.after_dataset).strip()

        start_idx = None
        for i, (db, ds, _) in enumerate(datasets):
            if str(ds).strip() == target:
                start_idx = i
                break

        if start_idx is None:
            available = sorted({str(ds).strip() for _, ds, _ in datasets})
            raise ValueError(
                f"--after_dataset '{target}' not found among discovered datasets. "
                f"Example available dataset names: {available[:20]}"
            )

        datasets = datasets[start_idx:]

        print(f"⚙️ --after_dataset activated: starting from '{target}' (inclusive).")
        print(f"➡️ Remaining datasets to run: {len(datasets)}")
        for db, ds, _ in datasets[:10]:
            print(f"   - {db}/{ds}")
        if len(datasets) > 10:
            print("   ...")

    print(f"Selected datasets (valid folders): {len(datasets)}")
    for (db, ds, folder) in datasets[:10]:
        print(f"  • {db}/{ds} -> {folder}")
    if len(datasets) > 10:
        print("  • ...")

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    if summary_csv.exists() and summary_csv.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(summary_csv)
            print(f"Existing summary CSV found with {len(existing_df)} rows: {summary_csv}")
        except Exception:
            print(f"Existing summary CSV found (unreadable), will append anyway: {summary_csv}")

    # Run sequentially (recommended). The pipeline can already do internal parallelism per dataset.
    for db, ds, folder in tqdm(datasets, desc="Datasets", unit="ds"):
        cmd = build_pipeline_cmd(
            pipeline_script=pipeline_script,
            dataset_folder=folder,
            output_dir=output_dir,
            task_type=args.task_type,
            seed=args.seed,
            n_splits=args.n_splits,
            n_estimators_search=args.n_estimators_search,
            n_estimators_final=args.n_estimators_final,
            parallel=bool(args.parallel),
            n_jobs=int(args.n_jobs),
            use_tmp_dir=bool(args.use_tmp_dir),
            extra_args=list(args.extra),
        )

        print("\n" + "=" * 100)
        print(f"Running: {db}/{ds}")
        print("=" * 100)
        print("CMD:", " ".join(shlex.quote(x) for x in cmd), flush=True)

        row = run_one_dataset(db=db, ds=ds, folder=folder, cmd=cmd, logs_dir=logs_dir)

        print(f"-> status={row.status} returncode={row.returncode} elapsed={row.elapsed_sec:.1f}s")
        print(f"-> log: {row.log_path}")

        append_run_row_to_csv(summary_csv=summary_csv, row=row)

    print("\nDONE.")
    print(f"Summary CSV: {summary_csv}")
    print(f"Logs dir:    {logs_dir}")
    print(f"Output dir:  {output_dir}")


if __name__ == "__main__":
    main()