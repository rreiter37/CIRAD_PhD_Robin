#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
search_best_tabpfn_preproc.py

Two-stage search for the best (preprocessing + PCA + optional RFF) TabPFN configuration per dataset.

Key requirement (disk safety):
- If --no_artifacts_during_search is enabled:
    * Stage 1 and Stage 2 runs are executed in a temporary workspace (preferably /dev/shm)
    * The temporary workspace is deleted immediately after each run
    * => NO artifacts are written to the real workspace during search (Optuna included)
- The FINAL best run is executed in the real workspace and kept.

Stage 1 (fast screening):
- No RFF
- Small PCA space (no_pca + 2 PCA variants)
- 1-fold CV (n_splits=1) on calibration set (Xcal/Ycal)
- Select top-K configurations based on validation NRMSE normalized by CALIBRATION target range.

Stage 2 (refinement):
- Default: Optuna (50 trials by default)
- Candidate base configs are the top-K from Stage 1 (base_preproc + standardization)
- Expanded PCA space
- Optional RFF with append_raw ALWAYS TRUE (not a hyperparameter, as requested)
- Two-fidelity evaluation for pruning:
    1) quick eval with 1-fold CV
    2) if not pruned, full eval with n_splits=stage2_n_splits (default 3)
- Selection is based on validation NRMSE normalized by CALIBRATION target range.

Final:
- Train best config on the full calibration set (no folds)
- Evaluate on the test set (Xval/Yval)
- Report test NRMSE normalized by TEST target range.

Outputs:
- tabpfn_search_results.csv        : all evaluated configs (stage1 + optuna trials)
- best_tabpfn_per_dataset.csv      : selected best per dataset + final test metrics
- optuna_trials_<dataset>.csv      : trial-level details for Stage 2 (if Optuna)
- optuna_best_<dataset>.json       : best trial params summary (if Optuna)

Notes on metric normalization (important):
- Stage 1 / Stage 2: normalize RMSE by range(Ycal) because validation folds come from calibration set.
- Final test: normalize RMSE by range(Yval) because it is evaluated on the test set.
"""

from __future__ import annotations

import os
import re
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import shutil
import tempfile
import multiprocessing as mp
import concurrent.futures
import traceback

import numpy as np
import pandas as pd
import torch

from dotenv import load_dotenv
from huggingface_hub import login

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from nirs4all.data import DatasetConfigs, Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import ASLSBaseline
import nirs4all.operators.transforms as pp
from tabpfn import TabPFNRegressor, TabPFNClassifier


# ===================== TabPFN device safety ===================== #

def get_safe_tabpfn_device() -> str:
    """
    Returns 'cuda' if the GPU architecture is supported by TabPFN kernels,
    otherwise falls back to 'cpu' to prevent CUDA kernel crashes.
    """
    if not torch.cuda.is_available():
        return "cpu"
    major, _ = torch.cuda.get_device_capability(0)
    # TabPFN v2.5 kernels are not compiled for SM>=90 (e.g., Blackwell).
    if major >= 9:
        print("⚠️ Unsupported GPU architecture detected for TabPFN CUDA kernels → forcing CPU.")
        return "cpu"
    return "cuda"


TABPFN_DEVICE = get_safe_tabpfn_device()


# ===================== Determinism helpers ===================== #

def set_deterministic_env(seed: int) -> None:
    """
    Configure common environment variables to reduce non-determinism and avoid CPU oversubscription.
    This is especially important when using multi-processing.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Torch-level determinism knobs (mainly relevant if CUDA is used).
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== Simple spectral preprocessings ===================== #

class SNV(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) per-sample normalization:
    x' = (x - mean(x)) / std(x)
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        return (X - mu) / (sd + self.eps)



# --------------------- simple_preprocs (aligned with association_pp_model.py) --------------------- #
# We import these spectral preprocessors from nirs4all.operators.transformations (aliased as `pp`).
# In Stage 1, we want to explore each preprocessing from `simple_preprocs` (excluding PCA),
# but we avoid an unreadable "soup" of combinations by:
#   (1) searching the best (base_preproc x std_option x pca_option) for each simple preprocessing, and
#   (2) comparing the per-preprocessing best scores to keep the global top-K.
#
# IMPORTANT: we keep PCA handled separately by `pca_space_stage1()`.

def simple_preprocs_factories() -> List[Tuple[str, Any]]:
    """Return the list of (name, factory) for simple spectral preprocessors (excluding PCA)."""
    return [
        ("id", lambda: pp.IdentityTransformer()),
        ("baseline", lambda: pp.Baseline()),
        ("derivate", lambda: pp.Derivate()),
        ("detrend", lambda: pp.Detrend()),
        ("MSC", lambda: pp.MultiplicativeScatterCorrection()),
        ("normalize", lambda: pp.Normalize()),
        ("RNV", lambda: pp.RobustStandardNormalVariate()),
        ("savgol", lambda: pp.SavitzkyGolay()),
        ("simplescale", lambda: pp.SimpleScale()),
        ("SNV", lambda: pp.StandardNormalVariate()),
        ("haar", lambda: pp.Wavelet("haar")),
        ("gaussian", lambda: pp.Gaussian(order=2, sigma=1)),
    ]


def build_simple_preproc(name: str) -> List[BaseEstimator]:
    """Instantiate the transformer(s) for a given simple preprocessing name."""
    for n, fac in simple_preprocs_factories():
        if n == name:
            return [fac()]
    raise KeyError(f"Unknown simple preprocessing: {name}")


class PCAAdaptive(BaseEstimator, TransformerMixin):
    """
    PCA with n_components chosen as a fraction of n_samples seen at fit time:
    n_components = clamp(int(fraction * n_samples), min_components, max_components)
    """
    def __init__(
        self,
        fraction: float = 0.25,
        whiten: bool = True,
        min_components: int = 5,
        max_components: Optional[int] = None,
        random_state: int = 42,
    ):
        self.fraction = fraction
        self.whiten = whiten
        self.min_components = min_components
        self.max_components = max_components
        self.random_state = random_state
        self._pca = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        n_comp = int(self.fraction * n_samples)
        n_comp = max(self.min_components, n_comp)
        n_comp = min(n_comp, n_features)

        if self.max_components is not None:
            n_comp = min(n_comp, self.max_components)

        self._pca = PCA(
            n_components=n_comp,
            whiten=self.whiten,
            random_state=self.random_state
        )
        self._pca.fit(X)
        return self

    def transform(self, X):
        if self._pca is None:
            raise RuntimeError("PCAAdaptive was not fitted before transform().")
        return self._pca.transform(np.asarray(X))


class RFFEncoding(BaseEstimator, TransformerMixin):
    """
    Random Fourier Feature encoding for RBF kernels.

    IMPORTANT:
    - append_raw is ALWAYS TRUE in this project (per user requirement).
      Therefore, append_raw is NOT a hyperparameter.
    """
    def __init__(
        self,
        n_components: int = 256,
        sigma: float = 1.0,
        append_raw: bool = True,   # kept for API clarity; always True here
        random_state: int = 42,
    ):
        self.n_components = int(n_components)
        self.sigma = float(sigma)
        self.append_raw = bool(append_raw)
        self.random_state = int(random_state)
        self.W_ = None
        self.b_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        rng = np.random.RandomState(self.random_state)
        d = X.shape[1]
        self.W_ = rng.normal(0.0, 1.0 / self.sigma, size=(d, self.n_components))
        self.b_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        X = np.asarray(X)
        Z = X @ self.W_ + self.b_
        Z = np.sqrt(2.0 / self.n_components) * np.concatenate([np.cos(Z), np.sin(Z)], axis=1)
        # append_raw is always True here
        return np.hstack([X, Z]) if self.append_raw else Z


# ===================== Ranking helpers (top-N datasets) ===================== #

def load_ranking_json(json_path: str) -> dict:
    """Load the difficulty ranking JSON file."""
    jp = Path(json_path)
    if not jp.exists():
        raise FileNotFoundError(f"Ranking file not found: {jp}")
    with open(jp, "r") as f:
        data = json.load(f)
    if "rankings" not in data:
        raise ValueError("Ranking JSON is missing the 'rankings' key.")
    return data["rankings"]


def pick_top_datasets(ranking_list: List[str], top_n: str) -> List[str]:
    """Return the top-N datasets. If top_n == 'all', return full list."""
    if top_n == "all":
        return list(ranking_list)
    return list(ranking_list[: int(top_n)])


def filter_after_dataset(dataset_list: List[str], after_dataset: Optional[str]) -> List[str]:
    """If after_dataset is provided, return the sublist starting from that dataset."""
    if after_dataset is None:
        return dataset_list
    if after_dataset not in dataset_list:
        raise ValueError(f"--after_dataset '{after_dataset}' not found in ranking list.")
    start_index = dataset_list.index(after_dataset)
    return dataset_list[start_index:]


# ===================== Robust CSV reading (inspired by the old working behavior) ===================== #

def read_vector_csv(path: Path) -> np.ndarray:
    """
    Robust target reader:
    - auto-detect separator (',' vs ';')
    - tolerate headers / ids / extra columns
    - select the most numeric column
    - handle malformed CSV files with error_bad_lines
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # Try to read with error_bad_lines='skip' to handle inconsistent field counts
    try:
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines='skip')
    except Exception:
        # Fallback: try with a specific separator
        try:
            df = pd.read_csv(path, sep=",", on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(path, sep=";", on_bad_lines='skip')
    
    if df.shape[0] == 0:
        raise RuntimeError(f"Empty target file: {path}")

    df_num = df.apply(pd.to_numeric, errors="coerce")

    valid_counts = df_num.notna().sum(axis=0)
    best_col = valid_counts.idxmax()

    if valid_counts[best_col] == 0:
        raise RuntimeError(f"No numeric column found in target file: {path}")

    y = df_num[best_col].dropna().to_numpy(dtype=float)
    if y.size == 0:
        raise RuntimeError(f"Parsed target is empty after cleaning: {path}")

    return y


def read_matrix_csv(path: Path) -> np.ndarray:
    """
    Robust matrix reader:
    - auto-detect separator (',' vs ';')
    - tolerate wavelength header rows (e.g., '852.78_nm')
    - tolerate an ID/index first column
    - handle malformed CSV files with error_bad_lines
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # Try to read with on_bad_lines='skip' to handle inconsistent field counts
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str, on_bad_lines='skip')
    except Exception:
        # Fallback: try with a specific separator
        try:
            df = pd.read_csv(path, sep=",", dtype=str, on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(path, sep=";", dtype=str, on_bad_lines='skip')
    
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise RuntimeError(f"Empty matrix file: {path}")

    df_num = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows that are mostly non-numeric (e.g., wavelength header row)
    row_numeric_rate = df_num.notna().mean(axis=1)
    df_num = df_num.loc[row_numeric_rate > 0.5].copy()
    if df_num.shape[0] == 0:
        raise RuntimeError(f"No numeric rows found in matrix file: {path}")

    # Drop columns that are mostly non-numeric (e.g., sample id column)
    col_numeric_rate = df_num.notna().mean(axis=0)
    df_num = df_num.loc[:, col_numeric_rate > 0.5].copy()
    if df_num.shape[1] == 0:
        raise RuntimeError(f"No numeric columns found in matrix file: {path}")

    # Keep only fully numeric columns (avoid NaNs leaking into models)
    df_num = df_num.dropna(axis=1, how="any")
    if df_num.shape[1] == 0:
        raise RuntimeError(f"All numeric columns contain NaNs in: {path}")

    return df_num.to_numpy(dtype=float)


def compute_range_from_file(dataset_dir: Path, filename: str) -> float:
    """
    Compute the target range from a given CSV filename (e.g., Ycal.csv or Yval.csv).
    """
    y = read_vector_csv(dataset_dir / filename)
    r = float(np.max(y) - np.min(y))
    return max(r, 1e-12)


def cleanup_run_dir(workspace: Path, dataset_norm: str, run_name: str) -> None:
    """
    Best-effort deletion of nirs4all run directories matching a run_name.
    This is only used when runs are executed in the REAL workspace.
    """
    runs_root = workspace / "runs" / dataset_norm
    if not runs_root.exists():
        return
    for p in runs_root.glob(f"*{run_name}*"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)


# --- Add this helper somewhere in the script (e.g. near other helpers) ---
def append_or_merge_csv(csv_path: Path, new_df: pd.DataFrame, dedup_subset: list[str] | None) -> None:
    """
    Append new rows to an existing CSV without losing previous content.
    If the CSV already exists, we merge (concat) and optionally drop duplicates
    (keeping the last occurrence).
    """
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)

        # Align schemas (in case columns evolved)
        for c in new_df.columns:
            if c not in old_df.columns:
                old_df[c] = pd.NA
        for c in old_df.columns:
            if c not in new_df.columns:
                new_df[c] = pd.NA

        # Keep the old column order for stability
        new_df = new_df[old_df.columns]

        merged = pd.concat([old_df, new_df], ignore_index=True)

        if dedup_subset is not None and len(dedup_subset) > 0:
            # Keep the most recent row if duplicates exist
            merged = merged.drop_duplicates(subset=dedup_subset, keep="last")

        merged.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)


# ===================== No-artifact search workspace helpers ===================== #

def get_tmpfs_root() -> Path:
    """
    Prefer a RAM-backed filesystem if available (Linux).
    This avoids filling disk even if the pipeline writes artifacts.
    """
    p = Path("/dev/shm")
    if p.exists() and os.access(str(p), os.W_OK):
        return p
    return Path(tempfile.gettempdir())


def make_search_workspace(dataset_norm: str) -> Path:
    """
    Create a temporary workspace for search runs only.
    This workspace is deleted after each run (strict mode).
    """
    root = get_tmpfs_root()
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"tabpfn_search_{dataset_norm}_", dir=str(root)))
    return tmp_dir


def nuke_workspace(ws: Path) -> None:
    """Delete the whole workspace directory (best-effort)."""
    try:
        shutil.rmtree(ws, ignore_errors=True)
    except Exception:
        pass


def run_pipeline_no_artifacts(
    args: argparse.Namespace,
    real_workspace: Path,
    dataset_norm: str,
    run_name: str,
    pipe_cfg: PipelineConfigs,
    ds_cfg: DatasetConfigs,
    force_tmp_workspace: bool = False,
) -> dict:
    """
    Execute a nirs4all pipeline run.

    If --no_artifacts_during_search is enabled:
      - run in a temporary workspace (preferably /dev/shm)
      - delete the temp workspace immediately after completion
      => NO artifacts are saved to the real workspace during Stage 1/2 (Optuna included)

    Otherwise:
      - run in the real workspace
      - optionally cleanup the run folder if --keep_only_best and not --keep_search_runs
    """
    if force_tmp_workspace or args.no_artifacts_during_search:
        tmp_ws = make_search_workspace(dataset_norm)
        tmp_runner = PipelineRunner(verbose=0, workspace_path=str(tmp_ws))
        try:
            _, preds_per_ds = tmp_runner.run(pipe_cfg, ds_cfg)
            return preds_per_ds
        finally:
            nuke_workspace(tmp_ws)

    # Real workspace run
    real_runner = PipelineRunner(verbose=0, workspace_path=str(real_workspace))
    _, preds_per_ds = real_runner.run(pipe_cfg, ds_cfg)

    if args.keep_only_best and not args.keep_search_runs:
        cleanup_run_dir(real_workspace, dataset_norm, run_name)

    return preds_per_ds


# ===================== nirs4all Predictions handling (old script style) ===================== #

def get_single_dataset_key(preds_per_ds: dict) -> str:
    """
    Old-script style: when running one dataset at a time, nirs4all returns a dict
    with a single key. We simply pick it.
    """
    if not isinstance(preds_per_ds, dict) or len(preds_per_ds) == 0:
        raise RuntimeError("preds_per_ds is empty or not a dict.")
    if len(preds_per_ds) != 1:
        raise RuntimeError(
            f"Expected 1 dataset in preds_per_ds, got {len(preds_per_ds)} keys: {list(preds_per_ds.keys())}"
        )
    return next(iter(preds_per_ds.keys()))


def get_top_rmse(preds: Predictions, partition: str) -> Optional[float]:
    """
    Extract RMSE from a Predictions object using the official API:
    preds.top(1, 'rmse', rank_partition=partition, display_metrics=['rmse'])
    """
    try:
        top = preds.top(1, "rmse", rank_partition=partition, display_metrics=["rmse"])
    except Exception:
        tb = traceback.format_exc()
        return ('__ERROR__', job, tb)

    if not isinstance(top, list) or len(top) == 0:
        return None

    rmse = top[0].get("rmse", None)
    return None if rmse is None else float(rmse)


# ===================== RFF bounds helper ===================== #

def rff_bounds_from_ncal(n_cal: int) -> Tuple[int, int]:
    """
    Define reasonable bounds for RFF n_components from the calibration size.
    This is used by Optuna to constrain the search space.
    """
    base = int(np.clip(np.sqrt(max(n_cal, 10)) * 8, 64, 512))
    lo = int(np.clip(base // 2, 32, 512))
    hi = int(np.clip(base * 2, 32, 512))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


# ===================== Results schemas ===================== #

@dataclass
class SearchResult:
    dataset: str
    stage: str
    config_name: str
    preprocessing: str
    simple_preproc: str
    standardization: str
    pca: str
    rff: str
    val_rmse: float
    val_nrmse: float
    normalizer: str  # "range_ycal" or "range_yval"
    test_rmse: Optional[float] = None
    test_nrmse: Optional[float] = None


@dataclass
class BestResult:
    dataset: str
    best_config_name: str
    best_preprocessing: str
    best_simple_preproc: str
    best_standardization: str
    best_pca: str
    best_rff: str
    selected_on: str
    val_rmse: float
    val_nrmse: float
    val_normalizer: str
    final_test_rmse: float
    final_test_nrmse: float
    test_normalizer: str


# ===================== TabPFN / pipeline builders ===================== #

def make_tabpfn_model(task_type: str, seed: int, n_estimators: int, model_path: str):
    """Create the TabPFN model object."""
    if task_type == "regression":
        return TabPFNRegressor(
            n_estimators=n_estimators,
            device=TABPFN_DEVICE,
            random_state=seed,
            model_path=model_path,
            ignore_pretraining_limits=True,
        )
    return TabPFNClassifier(
        n_estimators=n_estimators,
        device=TABPFN_DEVICE,
        random_state=seed,
        model_path=model_path,
        ignore_pretraining_limits=True,
    )


def build_pipeline(
    preprocessing_steps: List,
    use_split: bool,
    n_splits: int,
    seed: int,
    task_type: str,
    model_path: str,
    n_estimators: int,
) -> List:
    """
    Build a pipeline list compatible with nirs4all PipelineConfigs.
    - use_split=True adds SPXYGFold for validation folds.
    """
    pipe = []
    pipe.extend(preprocessing_steps)
    # Keep target scaling consistent with your earlier pipelines
    pipe.append({"y_processing": StandardScaler()})

    if use_split:
        pipe.append({"split": SPXYGFold(n_splits=n_splits, random_state=seed)})

    pipe.append({
        "model": make_tabpfn_model(task_type, seed, n_estimators, model_path),
        "name": "TabPFN",
    })
    return pipe


# ===================== Search spaces ===================== #

def base_preproc_space() -> List[Tuple[str, List]]:
    """
    Stage-1 base preprocessing space (coherent with the original script).

    Keep Stage 1 small and fast:
    - identity (no preprocessing)
    - ASLS baseline correction
    """
    return [
        ("id", []),
        ("asls_baseline", [ASLSBaseline()]),
    ]


def std_preproc_space() -> List[Tuple[str, List]]:
    """Stage-1 standardizations (coherent with the original script)."""
    return [
        ("no_std", []),
        ("std_scaler", [StandardScaler()]),
        ("snv", [SNV()]),
        ("snv+std", [SNV(), StandardScaler()]),
    ]

def pca_space_stage1(seed: int) -> List[Tuple[str, Optional[BaseEstimator]]]:
    """Small PCA space for Stage 1."""
    return [
        ("no_pca", None),
        ("pca_adapt_0.25n", PCAAdaptive(fraction=0.25, whiten=True, random_state=seed)),
    ]

def pca_space_stage2(seed: int) -> List[Tuple[str, Optional[BaseEstimator]]]:
    """Expanded PCA space for Stage 2."""
    return [
        ("no_pca", None),
        ("pca_0.99", PCA(n_components=0.99, random_state=seed, whiten=True)),
        ("pca_adapt_0.10n", PCAAdaptive(fraction=0.10, whiten=True, random_state=seed)),
        ("pca_adapt_0.25n", PCAAdaptive(fraction=0.25, whiten=True, random_state=seed)),
    ]

# ===================== CLI ===================== #

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--datasets", nargs="+", default=None,
                   help="Explicit dataset folder paths. If provided, ranking arguments are ignored.")
    p.add_argument("--data_root", type=str, default="Data/Regression",
                   help="Root folder containing dataset subfolders.")

    # Ranking-based selection
    p.add_argument("--ranking_json", type=str,
                   default="Results/assoc_pp_model/All_datasets/Rank_datasets_difficulty/dataset_difficulty_ranking__ALL.json")
    p.add_argument("--difficulty_ranking", type=str, default="dataset_size_ascending",
                   choices=["best_rrmse", "mean_rrmse", "mean_best_model", "dataset_size_ascending"])
    p.add_argument("--top_n", type=str, default="all",
                   help='How many datasets to run from the ranking: integer or "all".')
    p.add_argument("--after_dataset", type=str, default=None)

    p.add_argument("--workspace", type=str, default="wk_tabpfn_search")
    p.add_argument("--output_dir", type=str, default="Results/tabpfn_search_best")

    p.add_argument("--task_type", type=str, default="regression", choices=["regression", "classification"])
    p.add_argument("--seed", type=int, default=42)

    # Stage 1 controls
    p.add_argument("--stage1_parallel", action="store_true",
                   help="Evaluate Stage 1 configurations in parallel (deterministic).")
    p.add_argument("--stage1_n_jobs", type=int, default=1,
                   help="Number of worker processes for Stage 1 when --stage1_parallel is set.")
    p.add_argument("--stage1_n_splits", type=int, default=1,
                   help="Number of folds used in Stage 1 (default: 1 for fast screening).")
    p.add_argument("--stage1_top_k", type=int, default=5,
                   help="How many best configs (Stage 1) to keep for Stage 2 refinement.")

    # Stage 2 controls
    p.add_argument("--stage2_method", type=str, default="optuna", choices=["optuna", "grid"],
                   help="Stage 2 refinement method.")
    p.add_argument("--stage2_n_splits", type=int, default=3,
                   help="Number of folds used for Stage 2 scoring (default: 3).")

    # Optuna parameters (default requested: 50 trials)
    p.add_argument("--optuna_trials", type=int, default=20,
                   help="Number of Optuna trials for Stage 2 (default: 50).")
    p.add_argument("--optuna_timeout", type=int, default=0,
                   help="Optional timeout in seconds for Optuna Stage 2 (0 = no timeout).")
    p.add_argument("--optuna_seed", type=int, default=42,
                   help="Optuna sampler seed (defaults to --seed).")

    # TabPFN compute controls
    p.add_argument("--stage1_n_estimators", type=int, default=2,
                   help="TabPFN n_estimators used in Stage 1 (faster).")
    p.add_argument("--stage2_n_estimators", type=int, default=2,
                   help="TabPFN n_estimators used in Stage 2.")
    p.add_argument("--final_n_estimators", type=int, default=4,
                   help="TabPFN n_estimators used for the final full-calibration training.")

    # HF / model
    p.add_argument("--hf_token_env", type=str, default="HF_TOKEN")
    p.add_argument("--model_path", type=str, default="tabpfn-v2.5-regressor-v2.5_real.ckpt")

    # Disk / artifact control
    p.add_argument(
        "--keep_only_best",
        action="store_true",
        help="If NOT using --no_artifacts_during_search, delete real-workspace run folders during search."
    )
    p.add_argument(
        "--keep_search_runs",
        action="store_true",
        help="Debug option: keep Stage 1 / Stage 2 runs even if --keep_only_best is set."
    )
    p.add_argument(
        "--no_artifacts_during_search",
        action="store_true",
        help="Run Stage 1/2 in a temporary workspace (preferably /dev/shm) and delete it immediately. "
             "Only the FINAL best run is kept in the real workspace."
    )

    return p.parse_args()


def resolve_dataset_paths(args) -> List[Path]:
    """
    Resolve dataset paths:
    - If --datasets is provided: use them as-is.
    - Else: use ranking JSON to pick top-N dataset names, then map to data_root/name.
    """
    if args.datasets is not None:
        return sorted([Path(d) for d in args.datasets], key=lambda x: x.name)

    data_root = Path(args.data_root)
    rankings = load_ranking_json(args.ranking_json)

    if args.difficulty_ranking not in rankings:
        raise KeyError(f"Ranking key '{args.difficulty_ranking}' not found. Available: {list(rankings.keys())}")

    ranking_list = rankings[args.difficulty_ranking]
    ranking_list = filter_after_dataset(ranking_list, args.after_dataset)
    selected = pick_top_datasets(ranking_list, args.top_n)

    print("\nSelected datasets (from ranking):")
    for ds in selected:
        print("  •", ds)

    return [data_root / ds for ds in selected]


# ===================== Helper to rebuild steps from names ===================== #

def sanitize_name(s: str) -> str:
    """Make a string safe for file / run names."""
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def steps_from_names(
    base_name: str,
    simple_name: str,
    std_name: str,
    pca_name: str,
    pca_obj: Optional[BaseEstimator],
    use_rff: bool,
    rff_n_components: int,
    rff_sigma: float,
    seed: int,
) -> Tuple[List, str, str]:
    """
    Build preprocessing steps list and a readable 'rff label' for logging/CSV.
    """
    steps: List[Any] = []

    if base_name == "asls_baseline":
        steps.append(ASLSBaseline())

    # Apply the selected simple spectral preprocessing (may be identity)
    if simple_name is not None:
        steps.extend(build_simple_preproc(simple_name))

    if std_name == "std_scaler":
        steps.append(StandardScaler())
    elif std_name == "snv":
        steps.append(SNV())
    elif std_name == "snv+std":
        steps.extend([SNV(), StandardScaler()])
    # no_std -> nothing

    if pca_obj is not None:
        steps.append(pca_obj)

    rff_label = "no_rff"
    if use_rff:
        # append_raw is always True
        steps.append(RFFEncoding(
            n_components=int(rff_n_components),
            sigma=float(rff_sigma),
            append_raw=True,
            random_state=seed,
        ))
        rff_label = f"rff_nc{int(rff_n_components)}_sg{float(rff_sigma):.6g}_ar1"

    config_name = sanitize_name(f"{base_name}__{simple_name}__{std_name}__{pca_name}__{rff_label}")
    return steps, rff_label, config_name


# ===================== Stage 1 parallel worker ===================== #

def _stage1_eval_one(job: Dict[str, Any]) -> Any:
    """
    Evaluate a single Stage-1 configuration.

    Returns:
        (SearchResult, candidate_dict, log_line) on success, or an error dict on failure.

    Notes on determinism:
    - Each job receives explicit seed and uses deterministic operators.
    - We force Stage-1 runs to execute in a temporary workspace to avoid I/O races.
    """
    try:
        seed = int(job["seed"])
        set_deterministic_env(seed)

        args = job["args"]
        # Rebuild minimal Namespace-like access: args is already a dict-like object
        class _A:
            pass
        a = _A()
        for k, v in args.items():
            setattr(a, k, v)

        dataset_name = job["dataset_name"]
        dataset_norm = job["dataset_norm"]
        cal_range = float(job["cal_range"])
        workspace = Path(job["workspace"])
        ds_cfg = DatasetConfigs([job["ds_path"]], task_type=job["task_type"])

        base_name = job["base_name"]
        simple_name = job["simple_name"]
        std_name = job["std_name"]
        pca_name = job["pca_name"]
        pca_obj = job["pca_obj"]

        steps, rff_label, config_name = steps_from_names(
            base_name=base_name,
            simple_name=job["simple_name"],
            std_name=std_name,
            pca_name=pca_name,
            pca_obj=pca_obj,
            use_rff=False,
            rff_n_components=0,
            rff_sigma=0.0,
            seed=seed,
        )
        run_name = f"STAGE1_{dataset_name}__{config_name}"

        pipeline = build_pipeline(
            preprocessing_steps=steps,
            use_split=True,
            n_splits=int(job["stage1_n_splits"]),
            seed=seed,
            task_type=job["task_type"],
            model_path=job["model_path"],
            n_estimators=int(job["stage1_n_estimators"]),
        )
        pipe_cfg = PipelineConfigs(pipeline, run_name)

        preds_per_ds = run_pipeline_no_artifacts(
            args=a,  # uses only the artifact-related flags
            real_workspace=workspace,
            dataset_norm=dataset_norm,
            run_name=run_name,
            pipe_cfg=pipe_cfg,
            ds_cfg=ds_cfg,
            force_tmp_workspace=True,  # critical for safe deterministic parallel runs
        )

        ds_key = get_single_dataset_key(preds_per_ds)
        ds_pred = preds_per_ds[ds_key]["run_predictions"]

        val_rmse = get_top_rmse(ds_pred, partition="val")
        if val_rmse is None:
            return None

        val_nrmse = float(val_rmse) / cal_range

        sr = SearchResult(
            dataset=dataset_name,
            stage="stage1",
            config_name=config_name,
            preprocessing=base_name,
            simple_preproc=job["simple_name"],
            standardization=std_name,
            pca=pca_name,
            rff=rff_label,
            val_rmse=float(val_rmse),
            val_nrmse=float(val_nrmse),
            normalizer="range_ycal",
            test_rmse=None,
            test_nrmse=None,
        )

        cand = {
            "preprocessing": base_name,
            "simple_preproc": job["simple_name"],
            "standardization": std_name,
            "pca": pca_name,
            "val_nrmse": float(val_nrmse),
        }

        log_line = f"  - {config_name}: val_nrmse(ycal)={val_nrmse:.6f}"
        return sr, cand, log_line

    except Exception:
        # Return an error payload so the parent process can surface the root cause.
        tb = traceback.format_exc()
        return {"__error__": True, "traceback": tb, "job": {k: (str(v)[:200] if not isinstance(v,(int,float,str,bool,type(None))) else v) for k,v in job.items() if k not in ('pca_obj',)} }

# ===================== Main ===================== #

def main():
    args = parse_args()

    # Configure determinism as early as possible.
    set_deterministic_env(int(args.seed))

    # --- Env / HF login ---
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    hf_token = os.environ.get(args.hf_token_env, None)
    if hf_token:
        login(token=hf_token)
    else:
        print(f"⚠️ {args.hf_token_env} not found. TabPFN may fail if the checkpoint requires HF auth.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset_paths = resolve_dataset_paths(args)
    dataset_paths = [p for p in dataset_paths if p.exists() and p.is_dir()]
    if len(dataset_paths) == 0:
        raise RuntimeError("No valid dataset folders found after resolution.")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    base_space = base_preproc_space()
    std_space = std_preproc_space()

    all_results: List[SearchResult] = []
    best_results: List[BestResult] = []

    for ds_path in dataset_paths:
        dataset_name = ds_path.name
        dataset_norm = dataset_name.lower()

        print(f"\n{'=' * 90}\nDataset: {dataset_name}\n{'=' * 90}")

        # Compute normalizers:
        # - calibration range is used for Stage 1/2 validation normalization (no leakage)
        # - test range is used for final test normalization
        cal_range = compute_range_from_file(ds_path, "Ycal.csv")
        test_range = compute_range_from_file(ds_path, "Yval.csv")

        # n_cal is used only to define the RFF search bounds
        Xcal = read_matrix_csv(ds_path / "Xcal.csv")
        n_cal = int(Xcal.shape[0])

        ds_cfg = DatasetConfigs([str(ds_path)], task_type=args.task_type)

        # =========================
        # Stage 1: fast screening (no RFF)
        # =========================
        print("\n[Stage 1] Fast screening (no RFF)")

        pca1 = pca_space_stage1(args.seed)


        # Build the Stage-1 search jobs in a stable order.
        #
        # To keep Stage 1 readable (and not a combinatorial soup), we use the following strategy:
        #   - For each simple preprocessing from `simple_preprocs` (as in association_pp_model.py, excluding PCA),
        #     we search the best configuration over:
        #         base_preproc_space()  x  std_options  x  pca_space_stage1()
        #   - Then we compare the *per-preprocessing best* scores and keep the global top-K as candidates for Stage 2.
        #
        # This gives broad coverage of meaningful preprocessings without exploding the number of configs.

        simple_space = [name for name, _ in simple_preprocs_factories()]

        # A small std space for Stage 1:
        # - no_std
        # - std_scaler
        # We optionally skip std_scaler for preprocessings that already normalize/scale strongly.
        scaling_like = {"normalize", "RNV", "simplescale", "SNV"}

        jobs: List[Dict[str, Any]] = []
        for simple_name in simple_space:
            for (base_name, _) in base_space:
                std_names = ["no_std", "std_scaler"] if simple_name not in scaling_like else ["no_std"]
                for std_name in std_names:
                    for (pca_name, pca_obj) in pca1:
                        jobs.append({
                            "seed": int(args.seed),
                            "args": {
                                # Only artifact-control flags are needed inside the worker.
                                "no_artifacts_during_search": bool(args.no_artifacts_during_search),
                                "keep_only_best": bool(args.keep_only_best),
                                "keep_search_runs": bool(args.keep_search_runs),
                            },
                            "dataset_name": dataset_name,
                            "dataset_norm": dataset_norm,
                            "ds_path": str(ds_path),
                            "task_type": args.task_type,
                            "workspace": str(workspace),
                            "cal_range": float(cal_range),
                            "stage1_n_splits": int(args.stage1_n_splits),
                            "stage1_n_estimators": int(args.stage1_n_estimators),
                            "model_path": args.model_path,
                            "base_name": base_name,
                            "simple_name": simple_name,
                            "std_name": std_name,
                            "pca_name": pca_name,
                            "pca_obj": pca_obj,
                        })

        stage1_candidates: List[Dict[str, Any]] = []
        stage1_rows: List[SearchResult] = []
        stage1_logs: List[str] = []
        stage1_errors: List[dict] = []

        if args.stage1_parallel and int(args.stage1_n_jobs) > 1:
            n_jobs = int(args.stage1_n_jobs)

            # Use 'spawn' to avoid fork-related non-determinism with torch and to keep workers isolated.
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
                futures = [ex.submit(_stage1_eval_one, j) for j in jobs]
                for fut in concurrent.futures.as_completed(futures):
                    out = fut.result()
                    if out is None:
                        continue
                    # Worker may return an error payload to help debugging.
                    if isinstance(out, dict) and out.get("__error__", False):
                        stage1_errors.append(out)
                        continue
                    sr, cand, log_line = out
                    stage1_rows.append(sr)
                    stage1_candidates.append(cand)
                    stage1_logs.append(log_line)

            # Deterministic display / selection:
            # - Sort search rows by config_name (stable across runs)
            # - Sort candidates by (val_nrmse, names) for deterministic top-K selection
            stage1_rows = sorted(stage1_rows, key=lambda r: r.config_name)
            stage1_candidates = sorted(
                stage1_candidates,
                key=lambda c: (c["val_nrmse"], c["preprocessing"], c["standardization"], c["pca"])
            )
            stage1_logs = sorted(stage1_logs)

            for sr in stage1_rows:
                all_results.append(sr)
            for line in stage1_logs:
                print(line)

        else:
            # Sequential deterministic evaluation (baseline behavior).
            for j in jobs:
                out = _stage1_eval_one(j)
                if out is None:
                    continue
                if isinstance(out, dict) and out.get("__error__", False):
                    stage1_errors.append(out)
                    continue
                sr, cand, log_line = out
                all_results.append(sr)
                stage1_candidates.append(cand)
                print(log_line)

            stage1_candidates = sorted(
                stage1_candidates,
                key=lambda c: (c["val_nrmse"], c["preprocessing"], c["standardization"], c["pca"])
            )


        # Reduce Stage-1 results to one best configuration per simple preprocessing.
        # This prevents a combinatorial explosion and makes the Stage-1 candidate set easier to interpret.
        best_by_simple: Dict[str, Dict[str, Any]] = {}
        for c in stage1_candidates:
            k = c.get("simple_preproc", "unknown")
            cur = best_by_simple.get(k)
            if cur is None:
                best_by_simple[k] = c
            else:
                # Deterministic tie-breaker.
                cur_key = (cur["val_nrmse"], cur["preprocessing"], cur["standardization"], cur["pca"])
                new_key = (c["val_nrmse"], c["preprocessing"], c["standardization"], c["pca"])
                if new_key < cur_key:
                    best_by_simple[k] = c

        stage1_candidates = sorted(
            list(best_by_simple.values()),
            key=lambda c: (c["val_nrmse"], c["simple_preproc"], c["preprocessing"], c["standardization"], c["pca"])
        )

        if len(stage1_candidates) == 0:
            print(f"❌ No successful Stage 1 runs for dataset: {dataset_name}")
            # Surface one representative traceback to diagnose failures.
            if stage1_errors:
                err = stage1_errors[-1]
                job = err.get("job", {})
                print("\n[Stage 1] Example error (most recent):")
                print(f"config: base={job.get('base_name')} simple={job.get('simple_name')} std={job.get('std_name')} pca={job.get('pca_name')}")
                print(err.get("traceback", ""))
            continue

        topk = stage1_candidates[: int(args.stage1_top_k)]

        print(f"\n[Stage 1] Top-{args.stage1_top_k} configs to refine:")
        for i, c in enumerate(topk, 1):
            print(f"  {i:>2}. simple={c['simple_preproc']} | base={c['preprocessing']} | std={c['standardization']} | pca={c['pca']} -> {c['val_nrmse']:.6f}")

        # =========================
        # Stage 2: refinement (Optuna by default)
        # =========================
        pca2 = pca_space_stage2(args.seed)
        pca2_dict = {k: v for (k, v) in pca2}

        # Optuna search bounds for RFF
        nc_lo, nc_hi = rff_bounds_from_ncal(n_cal)

        stage2_results: List[SearchResult] = []

        if args.stage2_method == "grid":
            # Optional fallback: exhaustive grid over (topK base configs) x (pca2) x (rff small grid)
            print("\n[Stage 2] Grid refinement (debug mode)")

            rff_nc_list = sorted(set([nc_lo, nc_hi]))
            rff_sigma_list = [0.5, 2.0]
            use_rff_list = [False, True]

            for cand in topk:
                base_name = cand["preprocessing"]
                simple_name = cand["simple_preproc"]
                std_name = cand["standardization"]

                for pca_name, pca_obj in pca2:
                    for use_rff in use_rff_list:
                        for nc in (rff_nc_list if use_rff else [0]):
                            for sg in (rff_sigma_list if use_rff else [0.0]):
                                steps, rff_label, config_name = steps_from_names(
                                    base_name=base_name,
                                    simple_name=simple_name,
                                    std_name=std_name,
                                    pca_name=pca_name,
                                    pca_obj=pca_obj,
                                    use_rff=use_rff,
                                    rff_n_components=nc,
                                    rff_sigma=sg,
                                    seed=args.seed,
                                )
                                run_name = f"STAGE2_GRID_{dataset_name}__{config_name}"

                                pipeline = build_pipeline(
                                    preprocessing_steps=steps,
                                    use_split=True,
                                    n_splits=args.stage2_n_splits,
                                    seed=args.seed,
                                    task_type=args.task_type,
                                    model_path=args.model_path,
                                    n_estimators=args.stage2_n_estimators,
                                )
                                pipe_cfg = PipelineConfigs(pipeline, run_name)

                                try:
                                    preds_per_ds = run_pipeline_no_artifacts(
                                        args=args,
                                        real_workspace=workspace,
                                        dataset_norm=dataset_norm,
                                        run_name=run_name,
                                        pipe_cfg=pipe_cfg,
                                        ds_cfg=ds_cfg,
                                    )
                                except Exception as e:
                                    print(f"⚠️ Stage 2 grid failed for {dataset_name} / {config_name}: {e}")
                                    continue

                                ds_key = get_single_dataset_key(preds_per_ds)
                                ds_pred = preds_per_ds[ds_key]["run_predictions"]

                                val_rmse = get_top_rmse(ds_pred, partition="val")
                                if val_rmse is None:
                                    continue

                                val_nrmse = float(val_rmse) / float(cal_range)

                                sr = SearchResult(
                                    dataset=dataset_name,
                                    stage="stage2_grid",
                                    config_name=config_name,
                                    preprocessing=base_name,
                                    simple_preproc=row.get("simple_name", ""),
                                    standardization=std_name,
                                    pca=pca_name,
                                    rff=rff_label,
                                    val_rmse=float(val_rmse),
                                    val_nrmse=float(val_nrmse),
                                    normalizer="range_ycal",
                                    test_rmse=None,
                                    test_nrmse=None,
                                )
                                stage2_results.append(sr)
                                all_results.append(sr)

                                print(f"  - {config_name}: val_nrmse(ycal)={val_nrmse:.6f}")

        else:
            print("\n[Stage 2] Optuna refinement (pruning: 1-fold -> 3-fold)")

            base_choices = [(c["preprocessing"], c["simple_preproc"], c["standardization"]) for c in topk]
            base_choice_labels = [f"{b}__{sp}__{s}" for (b, sp, s) in base_choices]

            base_choice_to_triplet = {lbl: trip for lbl, trip in zip(base_choice_labels, base_choices)}

            sampler_seed = args.seed if args.optuna_seed is None else args.optuna_seed
            sampler = TPESampler(seed=sampler_seed)
            pruner = MedianPruner(n_warmup_steps=10)

            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                study_name=f"optuna_stage2_{dataset_name}",
            )

            trial_rows: List[Dict[str, Any]] = []
            seen_configs = set()

            def eval_config_on_folds(
                base_name: str,
                simple_name: str,
                std_name: str,
                pca_name: str,
                use_rff: bool,
                rff_nc: int,
                rff_sigma: float,
                n_splits: int,
            ) -> float:
                """
                Evaluate a configuration with a given number of CV splits,
                returning validation NRMSE normalized by calibration range.
                """
                pca_obj = pca2_dict[pca_name]
                steps, rff_label, config_name = steps_from_names(
                    base_name=base_name,
                    simple_name=simple_name,
                    std_name=std_name,
                    pca_name=pca_name,
                    pca_obj=pca_obj,
                    use_rff=use_rff,
                    rff_n_components=rff_nc,
                    rff_sigma=rff_sigma,
                    seed=args.seed,
                )

                run_name = f"STAGE2_OPTUNA_{dataset_name}__{config_name}__spl{n_splits}"

                pipeline = build_pipeline(
                    preprocessing_steps=steps,
                    use_split=True,
                    n_splits=n_splits,
                    seed=args.seed,
                    task_type=args.task_type,
                    model_path=args.model_path,
                    n_estimators=args.stage2_n_estimators,
                )
                pipe_cfg = PipelineConfigs(pipeline, run_name)

                preds_per_ds = run_pipeline_no_artifacts(
                    args=args,
                    real_workspace=workspace,
                    dataset_norm=dataset_norm,
                    run_name=run_name,
                    pipe_cfg=pipe_cfg,
                    ds_cfg=ds_cfg,
                )

                ds_key = get_single_dataset_key(preds_per_ds)
                ds_pred = preds_per_ds[ds_key]["run_predictions"]

                val_rmse = get_top_rmse(ds_pred, partition="val")
                if val_rmse is None:
                    return float("inf")

                return float(val_rmse) / float(cal_range)

            def objective(trial: optuna.Trial) -> float:
                """
                Two-fidelity objective:
                - First compute 1-fold score (fast) and allow pruning.
                - If not pruned, compute full score with stage2_n_splits (default 3).
                """
                base_choice = trial.suggest_categorical("base_choice", base_choice_labels)
                base_name, simple_name, std_name = base_choice_to_triplet[base_choice]

                pca_name = trial.suggest_categorical("pca", list(pca2_dict.keys()))
                use_rff = trial.suggest_categorical("use_rff", [False, True])

                if use_rff:
                    rff_nc = int(trial.suggest_categorical("rff_n_components", sorted(set([nc_lo, nc_hi]))))
                    rff_sigma = float(trial.suggest_categorical("rff_sigma", [0.5, 2.0]))
                    rff_sigma_key = round(float(rff_sigma), 6)
                else:
                    rff_nc = 0
                    rff_sigma = 0.0
                    rff_sigma_key = None

                config_key = (base_choice, pca_name, bool(use_rff), int(rff_nc) if use_rff else None, rff_sigma_key)
                if config_key in seen_configs:
                    raise optuna.TrialPruned()
                seen_configs.add(config_key)

                score_1 = eval_config_on_folds(
                    base_name=base_name,
                    simple_name=simple_name,
                    std_name=std_name,
                    pca_name=pca_name,
                    use_rff=use_rff,
                    rff_nc=rff_nc,
                    rff_sigma=rff_sigma,
                    n_splits=1,
                )
                trial.report(score_1, step=0)
                if trial.should_prune():
                    trial_rows.append({
                        "trial": trial.number,
                        "base_name": base_name,
                        "simple_name": simple_name,
                        "std_name": std_name,
                        "pca": pca_name,
                        "use_rff": use_rff,
                        "rff_n_components": rff_nc,
                        "rff_sigma": rff_sigma,
                        "val_nrmse_1fold_ycal": score_1,
                        "val_nrmse_full_ycal": None,
                        "state": "PRUNED",
                    })
                    raise optuna.TrialPruned()

                score_full = eval_config_on_folds(
                    base_name=base_name,
                    simple_name=simple_name,
                    std_name=std_name,
                    pca_name=pca_name,
                    use_rff=use_rff,
                    rff_nc=rff_nc,
                    rff_sigma=rff_sigma,
                    n_splits=args.stage2_n_splits,
                )
                trial.report(score_full, step=1)

                trial_rows.append({
                    "trial": trial.number,
                    "base_name": base_name,
                    "simple_name": simple_name,
                    "std_name": std_name,
                    "pca": pca_name,
                    "use_rff": use_rff,
                    "rff_n_components": rff_nc,
                    "rff_sigma": rff_sigma,
                    "val_nrmse_1fold_ycal": score_1,
                    "val_nrmse_full_ycal": score_full,
                    "state": "COMPLETE",
                })
                return score_full

            # Warm-start: enqueue topK configs (no RFF)
            for cand in topk:
                base_name = cand["preprocessing"]
                std_name = cand["standardization"]
                pca_name = cand["pca"]
                base_choice = f"{base_name}__{cand['simple_preproc']}__{std_name}"
                if pca_name not in pca2_dict:
                    pca_name = "no_pca"
                study.enqueue_trial({"base_choice": base_choice, "pca": pca_name, "use_rff": False})

            timeout = None if args.optuna_timeout <= 0 else int(args.optuna_timeout)

            study.optimize(
                objective,
                n_trials=int(args.optuna_trials),
                timeout=timeout,
                gc_after_trial=True,
            )

            # Convert Optuna trial rows into SearchResult rows
            for row in trial_rows:
                if row["state"] != "COMPLETE":
                    continue

                base_name = row["base_name"]
                std_name = row["std_name"]
                pca_name = row["pca"]
                use_rff = bool(row["use_rff"])

                if use_rff:
                    rff_nc = int(row["rff_n_components"])
                    rff_sigma = float(row["rff_sigma"])
                    rff_label = f"rff_nc{rff_nc}_sg{rff_sigma:.6g}_ar1"
                else:
                    rff_label = "no_rff"

                config_name = sanitize_name(f"{base_name}__{row.get('simple_name','')}__{std_name}__{pca_name}__{rff_label}")
                val_nrmse = float(row["val_nrmse_full_ycal"])
                val_rmse = val_nrmse * float(cal_range)

                sr = SearchResult(
                    dataset=dataset_name,
                    stage="stage2_optuna",
                    config_name=f"{config_name}__trial{row['trial']}",
                    preprocessing=base_name,
                    simple_preproc=row.get("simple_name", ""),
                    standardization=std_name,
                    pca=pca_name,
                    rff=rff_label,
                    val_rmse=float(val_rmse),
                    val_nrmse=float(val_nrmse),
                    normalizer="range_ycal",
                    test_rmse=None,
                    test_nrmse=None,
                )
                stage2_results.append(sr)
                all_results.append(sr)

            # Save Optuna per-dataset artifacts (small CSV/JSON, not nirs4all runs)
            trials_csv = outdir / f"optuna_trials_{dataset_name}.csv"
            pd.DataFrame(trial_rows).to_csv(trials_csv, index=False)

            best_json = outdir / f"optuna_best_{dataset_name}.json"
            if study.best_trial is not None:
                best_payload = {
                    "dataset": dataset_name,
                    "best_value_val_nrmse_full_ycal": study.best_value,
                    "best_params": study.best_trial.params,
                    "n_trials": len(study.trials),
                }
                with open(best_json, "w") as f:
                    json.dump(best_payload, f, indent=2)

            print(f"Saved Optuna trials → {trials_csv}")
            print(f"Saved Optuna best → {best_json}")

        if len(stage2_results) == 0:
            print(f"❌ No successful Stage 2 results for dataset: {dataset_name}")
            continue

        # Select best configuration based on Stage 2 validation NRMSE
        best = sorted(stage2_results, key=lambda x: x.val_nrmse)[0]
        print(f"\n✅ Best config on validation (Stage 2) for {dataset_name}: {best.config_name}")
        print(f"   val_rmse={best.val_rmse:.6f} val_nrmse(ycal)={best.val_nrmse:.6f}")

        # =========================
        # Final: retrain on full calibration and test on Xval/Yval
        # =========================
        pca2_dict_final = {k: v for (k, v) in pca_space_stage2(args.seed)}
        pca_obj = pca2_dict_final.get(best.pca, None)

        use_rff = best.rff != "no_rff"
        rff_nc = 0
        rff_sigma = 0.0
        if use_rff:
            m = re.match(r"rff_nc(\d+)_sg([0-9.eE+-]+)_ar1", best.rff)
            if m is None:
                raise RuntimeError(f"Cannot parse RFF params from label: {best.rff}")
            rff_nc = int(m.group(1))
            rff_sigma = float(m.group(2))

        best_steps, _, _ = steps_from_names(
            base_name=best.preprocessing,
            simple_name=best.simple_preproc,
            std_name=best.standardization,
            pca_name=best.pca,
            pca_obj=pca_obj,
            use_rff=use_rff,
            rff_n_components=rff_nc,
            rff_sigma=rff_sigma,
            seed=args.seed,
        )

        final_run_name = f"FINAL_{dataset_name}__{sanitize_name(best.config_name)}"
        final_pipeline = build_pipeline(
            preprocessing_steps=best_steps,
            use_split=False,     # train on full cal, evaluate on test
            n_splits=0,
            seed=args.seed,
            task_type=args.task_type,
            model_path=args.model_path,
            n_estimators=args.final_n_estimators,
        )
        final_cfg = PipelineConfigs(final_pipeline, final_run_name)

        # FINAL is executed in the REAL workspace and kept
        _, final_preds_per_ds = PipelineRunner(verbose=0, workspace_path=str(workspace)).run(final_cfg, ds_cfg)
        ds_key = get_single_dataset_key(final_preds_per_ds)
        final_ds_pred = final_preds_per_ds[ds_key]["run_predictions"]

        final_test_rmse = get_top_rmse(final_ds_pred, partition="test")
        if final_test_rmse is None:
            final_test_rmse = get_top_rmse(final_ds_pred, partition="val")
        if final_test_rmse is None:
            raise RuntimeError(f"Final run has no RMSE in 'test' nor 'val' for dataset: {dataset_name}.")

        final_test_nrmse = float(final_test_rmse) / float(test_range)
        print(f"🎯 Final test metrics for {dataset_name}: test_rmse={final_test_rmse:.6f} test_nrmse(yval)={final_test_nrmse:.6f}")

        best_results.append(BestResult(
            dataset=dataset_name,
            best_config_name=best.config_name,
            best_preprocessing=best.preprocessing,
            best_simple_preproc=best.simple_preproc,
            best_standardization=best.standardization,
            best_pca=best.pca,
            best_rff=best.rff,
            selected_on="stage2_val_nrmse_range_ycal",
            val_rmse=best.val_rmse,
            val_nrmse=best.val_nrmse,
            val_normalizer="range_ycal",
            final_test_rmse=float(final_test_rmse),
            final_test_nrmse=float(final_test_nrmse),
            test_normalizer="range_yval",
        ))

        # Persist intermediate CSVs after each dataset (safe for long runs).
    df_all = pd.DataFrame([asdict(r) for r in all_results])
    df_best = pd.DataFrame([asdict(r) for r in best_results])

    all_csv = outdir / "tabpfn_search_results.csv"
    best_csv = outdir / "best_tabpfn_per_dataset.csv"

    # Merge with existing instead of overwriting:
    # - all_results: unique per (dataset, stage, config_name, normalizer)
    append_or_merge_csv(
        all_csv,
        df_all,
        dedup_subset=["dataset", "stage", "config_name", "normalizer"],
    )

    # - best_per_dataset: one row per dataset (keep latest if re-run)
    append_or_merge_csv(
        best_csv,
        df_best,
        dedup_subset=["dataset"],
    )

    print(f"\nSaved search results → {all_csv}")
    print(f"Saved best per dataset → {best_csv}")
    print(f"Workspace used → {workspace}")
    if args.no_artifacts_during_search:
        print("Disk policy: Stage 1/2 ran in temporary workspaces and were deleted immediately (no artifacts saved).")

if __name__ == "__main__":
    main()