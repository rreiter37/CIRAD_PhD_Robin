#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_catboost_final.py

Same pipeline logic as pipeline_tabpfn_final.py, but using CatBoost instead of TabPFN.

What this script does (per dataset folder):
- Cartesian search over preprocessing configurations using SPXYG K-fold CV on the training set
- Select best preprocessing configuration (based on CV metric)
- Refit on the full training set with the best configuration
- Predict on Xtest and save final predictions + (optional) fitted pipeline

Artifacts (relative to --output_dir):
- <dataset>__search_results.csv
- <dataset>__best_config.json
- <dataset>__final_predictions.csv

Notes:
- CSV convention: ';' separator and '.' decimal.
"""

from __future__ import annotations

import os
import json
import math
import argparse
import tempfile
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
from tqdm.auto import tqdm
import sys
import numpy as np
import pandas as pd
import re

import torch  # used only for determinism helpers (consistent with your existing style)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score

# NIRS4ALL (for SPXYG folds + spectral operators)
from nirs4all.operators.splitters import SPXYGFold
import nirs4all.operators.transforms as pp
from nirs4all.operators.transforms import ASLSBaseline

# CatBoost
from catboost import CatBoostRegressor, CatBoostClassifier

# Global list to store all predictions before writing to parquet
ALL_PREDS = []

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
# I/O helpers
# ==============================

def read_csv_strict(path: Path) -> pd.DataFrame:
    """Read CSV with the project convention: ';' separator and '.' decimal."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, sep=";", decimal=".")


def load_y_series(path: Path) -> pd.Series:
    df = read_csv_strict(path)
    # Ytrain/Ytest are one-column in your project
    return df.iloc[:, 0]


def load_dataset_folder(folder: Path):
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
    token = str(token)
    token = token.strip().replace(" ", "_")
    token = re.sub(r"[^A-Za-z0-9_.+-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "NA"


def build_preproc_tag(cfg: "SearchConfig") -> str:
    """Build a compact, filesystem-safe tag for a preprocessing configuration."""
    used = []
    if cfg.shape != "None":
        used.append(cfg.shape)
    if cfg.scatter != "None":
        used.append(cfg.scatter)
    if cfg.pca != "None":
        used.append(cfg.pca)

    tag = "+".join(used) if used else "None"
    return _sanitize_token(tag)



# ==============================
# PCA_features(fraction) transformer
# ==============================

class PCAFeaturesFraction(BaseEstimator, TransformerMixin):
    """
    PCA where n_components is a fraction of the number of input features:
        n_components = clamp(int(fraction * n_features), 1, n_features)
    """

    def __init__(self, fraction: float = 0.25, whiten: bool = True, random_state: int = 42):
        self.fraction = float(fraction)
        self.whiten = bool(whiten)
        self.random_state = int(random_state)
        self._pca: Optional[PCA] = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # n_components must be <= min(n_samples, n_features)
        n_comp = int(self.fraction * n_features)
        n_comp = max(1, min(n_comp, n_features, n_samples))

        self._pca = PCA(
            n_components=n_comp,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        self._pca.fit(X)
        return self

    def transform(self, X):
        if self._pca is None:
            raise RuntimeError("PCAFeaturesFraction must be fitted before transform().")
        return self._pca.transform(np.asarray(X))


# ==============================
# Search definitions
# ==============================

@dataclass(frozen=True)
class SearchConfig:
    shape: str
    scatter: str
    pca: str


def _make_savgol_from_name(name: str):
    """
    Build a Savitzky-Golay transformer from its encoded name.

    Supported names:
    - SG_11_2_1
    - SG_15_2_1
    - SG_21_2_1
    - SG_15_3_2
    - SG_21_3_2
    """
    if name == "SG_11_2_1":
        return pp.SavitzkyGolay(window_length=11, polyorder=2, deriv=1)
    if name == "SG_15_2_1":
        return pp.SavitzkyGolay(window_length=15, polyorder=2, deriv=1)
    if name == "SG_21_2_1":
        return pp.SavitzkyGolay(window_length=21, polyorder=2, deriv=1)
    if name == "SG_15_3_2":
        return pp.SavitzkyGolay(window_length=15, polyorder=3, deriv=2)
    if name == "SG_21_3_2":
        return pp.SavitzkyGolay(window_length=21, polyorder=3, deriv=2)
    raise ValueError(f"Unknown Savitzky-Golay config: {name}")


def build_transformers(cfg: SearchConfig, seed: int) -> List[Tuple[str, Any]]:
    """Instantiate sklearn-compatible transformers based on SearchConfig."""
    steps: List[Tuple[str, Any]] = []

    # 1) Shape correction block
    if cfg.shape == "ASLSBaseline":
        steps.append(("asls", ASLSBaseline()))
    elif cfg.shape in {"SG_11_2_1", "SG_15_2_1", "SG_21_2_1", "SG_15_3_2", "SG_21_3_2"}:
        steps.append(("savgol", _make_savgol_from_name(cfg.shape)))
    elif cfg.shape == "None":
        pass
    else:
        raise ValueError(f"Unknown shape preprocessing: {cfg.shape}")

    # 2) Scatter correction block
    if cfg.scatter == "SNV":
        steps.append(("snv", pp.StandardNormalVariate()))
    elif cfg.scatter == "EMSC":
        steps.append(("emsc", pp.nirs.ExtendedMultiplicativeScatterCorrection()))
    elif cfg.scatter == "None":
        pass
    else:
        raise ValueError(f"Unknown scatter preprocessing: {cfg.scatter}")

    # 3) Optional PCA
    if cfg.pca == "PCA_features_0.25":
        steps.append(("pca", PCAFeaturesFraction(fraction=0.25, whiten=True, random_state=int(seed))))
    elif cfg.pca != "None":
        raise ValueError(f"Unknown PCA preprocessing: {cfg.pca}")

    return steps


def enumerate_search_space() -> List[SearchConfig]:
    """
    Cartesian product used for TabPFN and CatBoost:

      {None, ASLSBaseline, SG(11,2,1), SG(15,2,1), SG(21,2,1), SG(15,3,2), SG(21,3,2)}
      x {None, SNV, EMSC}
      x {None, PCA_0.25F}
    """
    shapes = [
        "None",
        "ASLSBaseline",
        "SG_11_2_1",
        "SG_15_2_1",
        "SG_21_2_1",
        "SG_15_3_2",
        "SG_21_3_2",
    ]
    scatters = ["None", "SNV", "EMSC"]
    pcas = ["None", "PCA_features_0.25"]

    out: List[SearchConfig] = []
    for shape in shapes:
        for scatter in scatters:
            for pca in pcas:
                out.append(SearchConfig(shape=shape, scatter=scatter, pca=pca))
    return out


# ==============================
# Model factory (CatBoost)
# ==============================

def make_model(task_type: str, seed: int, n_estimators: int) -> Any:
    """
    Build a CatBoost model with deterministic-ish settings.

    - We map n_estimators_* args to CatBoost 'iterations'
    - We force CPU and thread_count=1 for reproducibility across machines
    """
    common = dict(
        iterations=int(n_estimators),
        random_seed=int(seed),
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
    )

    if task_type == "regression":
        return CatBoostRegressor(loss_function="RMSE", **common)

    return CatBoostClassifier(loss_function="Logloss", **common)


def score_one_split(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the metric for a single split (loss-like: minimize)."""
    if task_type == "regression":
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return float(rmse)  # minimize RMSE
    acc = accuracy_score(y_true, y_pred)
    return float(-acc)     # minimize (-ACC) == maximize ACC


@dataclass
class EvalResult:
    config: Dict[str, Any]
    mean_score: float
    fold_scores: List[float]


def cv_evaluate_config(
    dataset_folder: Path,
    cfg: SearchConfig,
    task_type: str,
    seed: int,
    n_splits: int,
    n_estimators: int,
    outdir: Path,
    dataset_name: str,
    tmp_root: Optional[Path] = None,
) -> EvalResult:
    """
    Evaluate a configuration with SPXYG K-fold CV on the training set.
    """
    set_deterministic(seed)

    X_train, y_train, X_test, y_test = load_dataset_folder(dataset_folder)
    X = X_train.to_numpy()
    y = y_train.to_numpy()

    split = SPXYGFold(n_splits=int(n_splits), random_state=int(seed))
    folds = list(split.split(X, y))

    preproc_tag = build_preproc_tag(cfg)
    fold_scores: List[float] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        steps = build_transformers(cfg, seed=seed)
        model = make_model(task_type=task_type, seed=seed, n_estimators=n_estimators)
        pipe = Pipeline(steps + [("model", model)])

        # Fit and predict
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xva)

        # Save validation predictions for this fold
        for i in range(len(yhat)):
            ALL_PREDS.append({
                "dataset": dataset_name,
                "split": "valid",
                "fold": fold_id,
                "config": preproc_tag,
                "y_true": float(yva[i]),
                "y_pred": float(yhat[i])
            })

        # Save test predictions for this fold (model trained on the fold's training subset)
        yhat_test = pipe.predict(X_test.to_numpy())
        if y_test is not None:
            for i in range(len(yhat_test)):
                ALL_PREDS.append({
                    "dataset": dataset_name,
                    "split": "test",
                    "fold": fold_id,
                    "config": preproc_tag,
                    "y_true": float(y_test.to_numpy()[i]),
                    "y_pred": float(yhat_test[i])
                })

        fold_scores.append(score_one_split(task_type, yva, yhat))

    mean_score = float(np.mean(fold_scores))
    return EvalResult(config=asdict(cfg), mean_score=mean_score, fold_scores=fold_scores)


def refit_and_predict_best(
    dataset_folder: Path,
    best_cfg: SearchConfig,
    task_type: str,
    seed: int,
    n_estimators: int,
    outdir: Path,
) -> None:
    """Refit the best pipeline on full train and predict on Xtest; save artifacts."""
    set_deterministic(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_dataset_folder(dataset_folder)

    steps = build_transformers(best_cfg, seed=seed)
    model = make_model(task_type=task_type, seed=seed, n_estimators=n_estimators)
    pipe = Pipeline(steps + [("model", model)])

    pipe.fit(X_train.to_numpy(), y_train.to_numpy())
    y_pred = pipe.predict(X_test.to_numpy())

    pred_df = pd.DataFrame({"y_pred": np.asarray(y_pred).reshape(-1)})
    if y_test is not None and len(y_test) == len(pred_df):
        pred_df.insert(0, "y_true", np.asarray(y_test).reshape(-1))

    pred_path = outdir / f"{dataset_folder.name}__final_predictions.csv"
    pred_df.to_csv(pred_path, sep=";", decimal=".", index=False)

    # Optional: save fitted pipeline
    try:
        import joblib
        joblib.dump(pipe, outdir / f"{dataset_folder.name}__final_pipeline.joblib")
    except Exception:
        pass


def worker_eval_one_config(payload: Dict[str, Any]) -> EvalResult:
    """Picklable worker for multiprocessing."""
    dataset_folder = Path(payload["dataset_folder"])
    cfg = SearchConfig(**payload["cfg"])

    task_type = payload["task_type"]
    seed = int(payload["seed"])
    n_splits = int(payload["n_splits"])
    n_estimators = int(payload["n_estimators_search"])
    use_tmp_dir = bool(payload["use_tmp_dir"])
    dataset_name = payload["dataset_name"]
    outdir = Path(payload["outdir"])

    set_deterministic(seed)

    tmp_root = None
    if use_tmp_dir:
        tmp_root = Path(tempfile.mkdtemp(prefix=f"catboost_search_{dataset_name}_"))

    try:
        return cv_evaluate_config(
            dataset_folder=dataset_folder,
            cfg=cfg,
            task_type=task_type,
            seed=seed,
            n_splits=n_splits,
            n_estimators=n_estimators,
            outdir=outdir,
            dataset_name=dataset_name,
            tmp_root=tmp_root,
        )
    finally:
        if tmp_root is not None and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)


# ==============================
# CLI / Main
# ==============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--datasets", nargs="+", required=True,
                   help="List of dataset folder paths (each must contain Xtrain.csv/Ytrain.csv/Xtest.csv).")
    p.add_argument("--output_dir", type=str, default="Results/catboost_cartesian_search",
                   help="Output directory for CSV/JSON/predictions.")
    p.add_argument("--task_type", type=str, default="regression", choices=["regression", "classification"],
                   help="Task type for CatBoost model.")

    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--n_splits", type=int, default=3, help="Number of SPXYG folds for search (default: 3).")

    # Keep same flag names as TabPFN pipeline for drop-in replacement.
    p.add_argument("--n_estimators_search", type=int, default=200,
                   help="CatBoost iterations during CV search.")
    p.add_argument("--n_estimators_final", type=int, default=500,
                   help="CatBoost iterations for the final refit.")

    p.add_argument("--parallel", action="store_true",
                   help="Enable parallel evaluation of the cartesian search.")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Number of worker processes if --parallel is enabled.")
    p.add_argument("--use_tmp_dir", action="store_true",
                   help="If set, each worker uses an isolated temp dir (recommended).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(int(args.seed))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    space = enumerate_search_space()

    for ds in args.datasets:
        dataset_folder = Path(ds)
        if not dataset_folder.exists():
            raise FileNotFoundError(str(dataset_folder))

        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_folder.name}")
        print("=" * 100)

        results: List[EvalResult] = []

        payloads: List[Dict[str, Any]] = []
        for cfg in space:
            payloads.append({
                "dataset_folder": str(dataset_folder),
                "dataset_name": dataset_folder.name,
                "cfg": asdict(cfg),
                "task_type": args.task_type,
                "seed": int(args.seed),
                "n_splits": int(args.n_splits),
                "n_estimators_search": int(args.n_estimators_search),
                "use_tmp_dir": bool(args.use_tmp_dir),
                "outdir": str(outdir),
            })

        if args.parallel and int(args.n_jobs) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.n_jobs)) as ex:
                futs = [ex.submit(worker_eval_one_config, pl) for pl in payloads]
                print(f"Submitted {len(futs)} jobs to the process pool.", flush=True)

                with tqdm(
                    total=len(futs),
                    desc="Evaluating configs",
                    unit="cfg",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    mininterval=0.5,
                    disable=False,
                ) as pbar:
                    for fut in concurrent.futures.as_completed(futs):
                        results.append(fut.result())
                        pbar.update(1)
        else:
            for pl in payloads:
                results.append(worker_eval_one_config(pl))

        df = pd.DataFrame([{
            **r.config,
            "mean_score": r.mean_score,
            "fold_scores": json.dumps(r.fold_scores),
        } for r in results])

        # For regression: minimize RMSE; for classification: minimize (-ACC) => still minimize
        df = df.sort_values("mean_score", ascending=True).reset_index(drop=True)

        # Save full results
        res_csv = outdir / f"{dataset_folder.name}__search_results.csv"
        df.to_csv(res_csv, sep=";", decimal=".", index=False)

        best_row = df.iloc[0].to_dict()
        best_cfg = SearchConfig(
            shape=str(best_row["shape"]),
            scatter=str(best_row["scatter"]),
            pca=str(best_row["pca"]),
        )

        best_json = outdir / f"{dataset_folder.name}__best_config.json"
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset_folder.name,
                    "task_type": args.task_type,
                    "seed": int(args.seed),
                    "n_splits": int(args.n_splits),
                    "model": "CatBoost",
                    "n_estimators_search": int(args.n_estimators_search),
                    "n_estimators_final": int(args.n_estimators_final),
                    "best_config": asdict(best_cfg),
                    "best_mean_score": float(best_row["mean_score"]),
                    "best_fold_scores": json.loads(best_row["fold_scores"]),
                },
                f,
                indent=2,
            )

        print(f"✅ Best config (CV): {asdict(best_cfg)} | mean_score={float(best_row['mean_score']):.6f}")

        # Final refit + predict
        refit_and_predict_best(
            dataset_folder=dataset_folder,
            best_cfg=best_cfg,
            task_type=args.task_type,
            seed=int(args.seed),
            n_estimators=int(args.n_estimators_final),
            outdir=outdir,
        )

        if ALL_PREDS:
            preds_df = pd.DataFrame(ALL_PREDS)
            parquet_path = outdir / "all_predictions.parquet"
            preds_df.to_parquet(parquet_path, index=False)

            print(f"Saved predictions parquet: {parquet_path}")

        print("✅ Final refit done. Saved final predictions.")


if __name__ == "__main__":
    main()