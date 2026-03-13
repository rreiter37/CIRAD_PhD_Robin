#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_tabpfn_raw.py

Run TabPFN "raw" (no preprocessing, no validation phase):
- Fit TabPFN on the full training set (Xtrain/Ytrain)
- Predict on the test set (Xtest)
- Save ONLY test predictions (no CV fold predictions)
- Save a small JSON metadata file and a per-dataset summary CSV

Expected dataset layout:
    <dataset_folder>/
        Xtrain.csv, Ytrain.csv, Xtest.csv, optionally Ytest.csv

Output layout (relative to --output_dir):
    <output_dir>/
        <dataset_name>__raw_predictions.csv
        <dataset_name>__raw_run.json
        <dataset_name>__raw_summary.csv
        preds/<dataset_name>/test/Raw_fold_0_preds.csv

All comments are in English (per your requirement).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from sklearn.metrics import mean_squared_error, accuracy_score

from tabpfn import TabPFNRegressor, TabPFNClassifier


# ==============================
# Determinism helpers
# ==============================

def set_deterministic(seed: int) -> None:
    """Set as much determinism as possible (Python/NumPy/PyTorch)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic if it is ever used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================
# TabPFN device safety (RTX 50xx guard)
# ==============================

def get_safe_tabpfn_device() -> str:
    """
    Returns 'cuda' if the GPU architecture is supported by TabPFN CUDA kernels,
    otherwise falls back to 'cpu' to prevent kernel crashes (e.g., RTX 50xx).
    """
    if not torch.cuda.is_available():
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)

    # TabPFN kernels may not be compiled for very recent architectures (e.g., SM >= 90).
    if major >= 9:
        return "cpu"

    return "cuda"


TABPFN_DEVICE = get_safe_tabpfn_device()


# ==============================
# I/O helpers (project conventions)
# ==============================

def read_csv_strict(path: Path) -> pd.DataFrame:
    """Read CSV with the project convention: ';' separator and '.' decimal."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, sep=";", decimal=".")


def load_y_series(path: Path) -> pd.Series:
    """
    Load a Y file as a 1D series.
    In your datasets, Ytrain/Ytest are typically one-column CSVs.
    """
    df = read_csv_strict(path)
    return df.iloc[:, 0]


def load_dataset_folder(folder: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:
    """Load Xtrain/Ytrain/Xtest/(optional Ytest) from a dataset folder."""
    xtr = read_csv_strict(folder / "Xtrain.csv")
    ytr = load_y_series(folder / "Ytrain.csv")
    xte = read_csv_strict(folder / "Xtest.csv")

    yte_path = folder / "Ytest.csv"
    yte = load_y_series(yte_path) if yte_path.exists() else None
    return xtr, ytr, xte, yte


# ==============================
# Prediction saving helpers
# ==============================

def _sanitize_token(token: str) -> str:
    """Sanitize a string token so it is safe to use in filenames."""
    token = str(token).strip().replace(" ", "_")
    token = re.sub(r"[^A-Za-z0-9_.+-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "NA"


def save_test_predictions_vector(
    base_outdir: Path,
    dataset_name: str,
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
) -> Path:
    """
    Save test predictions only (no validation predictions).

    Output layout:
        preds/<dataset_name>/test/Raw_fold_0_preds.csv
    """
    outdir = Path(base_outdir) / "preds" / str(dataset_name) / "test"
    outdir.mkdir(parents=True, exist_ok=True)

    fname = f"{_sanitize_token('Raw')}_fold_0_preds.csv"
    fpath = outdir / fname

    df = pd.DataFrame({"y_pred": np.asarray(y_pred).reshape(-1)})
    if y_true is not None:
        df.insert(0, "y_true", np.asarray(y_true).reshape(-1))

    df.to_csv(fpath, sep=";", decimal=".", index=False)
    return fpath


# ==============================
# Model + scoring
# ==============================

def make_model(task_type: str, seed: int, n_estimators: int, model_path: Optional[str]) -> Any:
    """Create a TabPFN model (regressor or classifier)."""
    ModelCls = TabPFNRegressor if task_type == "regression" else TabPFNClassifier
    kwargs = dict(
        n_estimators=int(n_estimators),
        device=TABPFN_DEVICE,
        random_state=int(seed),
        ignore_pretraining_limits=True,
    )
    if model_path is not None and str(model_path).strip():
        kwargs["model_path"] = str(model_path).strip()
    return ModelCls(**kwargs)


def compute_test_metric(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the main test metric:
    - regression: RMSE
    - classification: accuracy
    """
    if task_type == "regression":
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    return float(accuracy_score(y_true, y_pred))


@dataclass
class RawRunSummary:
    dataset: str
    task_type: str
    seed: int
    tabpfn_device: str
    n_estimators: int
    model_path: Optional[str]
    has_ytest: bool
    test_metric_name: str
    test_metric_value: Optional[float]


def run_one_dataset_raw(
    dataset_folder: Path,
    outdir: Path,
    task_type: str,
    seed: int,
    n_estimators: int,
    model_path: Optional[str],
) -> None:
    """
    Fit TabPFN on full train, predict on test, save artifacts.
    """
    set_deterministic(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_dataset_folder(dataset_folder)

    model = make_model(task_type=task_type, seed=seed, n_estimators=n_estimators, model_path=model_path)

    # Fit on full train (no preprocessing).
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    # Predict on test.
    y_pred = model.predict(X_test.to_numpy())

    # Save the "raw" predictions CSV (final-style file).
    pred_df = pd.DataFrame({"y_pred": np.asarray(y_pred).reshape(-1)})
    if y_test is not None and len(y_test) == len(pred_df):
        pred_df["y_true"] = np.asarray(y_test).reshape(-1)

    pred_path = outdir / f"{dataset_folder.name}__raw_predictions.csv"
    pred_df.to_csv(pred_path, sep=";", decimal=".", index=False)

    # Save also in the preds/<dataset>/test/ layout (like your fold-based layout),
    # but we only have one "pseudo-fold" (fold_0) and no validation.
    save_test_predictions_vector(
        base_outdir=outdir,
        dataset_name=dataset_folder.name,
        y_true=(y_test.to_numpy() if y_test is not None else None),
        y_pred=y_pred,
    )

    # Compute optional test metric if Ytest exists.
    metric_name = "rmse" if task_type == "regression" else "accuracy"
    metric_value: Optional[float] = None
    if y_test is not None:
        metric_value = compute_test_metric(task_type, y_test.to_numpy(), np.asarray(y_pred))

    # Save a JSON metadata file.
    run_json = outdir / f"{dataset_folder.name}__raw_run.json"
    summary = RawRunSummary(
        dataset=dataset_folder.name,
        task_type=task_type,
        seed=int(seed),
        tabpfn_device=str(TABPFN_DEVICE),
        n_estimators=int(n_estimators),
        model_path=(str(model_path) if model_path else None),
        has_ytest=bool(y_test is not None),
        test_metric_name=metric_name,
        test_metric_value=(float(metric_value) if metric_value is not None else None),
    )
    with open(run_json, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    # Save a tiny per-dataset summary CSV (handy for quick parsing).
    summary_csv = outdir / f"{dataset_folder.name}__raw_summary.csv"
    pd.DataFrame([asdict(summary)]).to_csv(summary_csv, sep=";", decimal=".", index=False)

    print(f"✅ Saved: {pred_path.name}, {run_json.name}, {summary_csv.name}", flush=True)


# ==============================
# CLI / Main
# ==============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset folder paths (each must contain Xtrain.csv/Ytrain.csv/Xtest.csv).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="Results/tabpfn_raw",
        help="Output directory for CSV/JSON/predictions.",
    )
    p.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type for TabPFN model.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--n_estimators",
        type=int,
        default=8,
        help="TabPFN n_estimators for the raw fit.",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Optional TabPFN checkpoint path (e.g., tabpfn-v2.5-regressor-v2.5_real.ckpt).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(int(args.seed))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path.strip() if isinstance(args.model_path, str) else ""
    model_path = model_path if model_path else None

    for ds in tqdm(args.datasets, desc="Datasets", unit="ds"):
        dataset_folder = Path(ds)
        if not dataset_folder.exists():
            raise FileNotFoundError(str(dataset_folder))

        print("\n" + "=" * 100)
        print(f"Dataset: {dataset_folder.name}")
        print("=" * 100)
        print(f"TabPFN device: {TABPFN_DEVICE}", flush=True)

        run_one_dataset_raw(
            dataset_folder=dataset_folder,
            outdir=outdir,
            task_type=args.task_type,
            seed=int(args.seed),
            n_estimators=int(args.n_estimators),
            model_path=model_path,
        )

    print("\nDONE.")


if __name__ == "__main__":
    main() 