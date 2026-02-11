#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tabpfn_cartesian_spxyg_cv3_refit.py

Cartesian hyperparameter search around TabPFN with:
- Deterministic parallel evaluation (ProcessPoolExecutor)
- 3-fold SPXYG split for model selection
- Final refit on the full training set with the best hyperparameters
- Final prediction on Xtest

Search space (cartesian product):
1) scaler: None OR StandardScaler
2) baseline: None OR ASLSBaseline
3) simple spectral preproc: None OR {SavitzkyGolay, Baseline, StandardNormalVariate, Gaussian, Normalize}
4) PCA: None OR PCA_features(0.25) where n_components = int(0.25 * n_features)

Notes:
- Comments are in English (per your requirement).
- The script assumes dataset folders contain Xtrain.csv, Ytrain.csv, Xtest.csv, optionally Ytest.csv.
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

import torch

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score

# NIRS4ALL (for SPXYG folds + spectral operators)
from nirs4all.operators.splitters import SPXYGFold
import nirs4all.operators.transforms as pp
from nirs4all.operators.transforms import ASLSBaseline


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
# Inspired by your existing code style in baseline/pipelines.
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

# TabPFN
from tabpfn import TabPFNRegressor, TabPFNClassifier  # noqa: E402


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
    # Ytrain/Ytest in your files are one-column with header 'x'
    return df.iloc[:, 0]

def load_dataset_folder(folder: Path):
    xtr = read_csv_strict(folder / "Xtrain.csv")
    ytr = load_y_series(folder / "Ytrain.csv")
    xte = read_csv_strict(folder / "Xtest.csv")

    yte_path = folder / "Ytest.csv"
    yte = load_y_series(yte_path) if yte_path.exists() else None
    return xtr, ytr, xte, yte



# ==============================
# PCA_features(fraction) transformer
# ==============================

class PCAFeaturesFraction(BaseEstimator, TransformerMixin):
    """
    PCA where n_components is a fraction of the number of input features:
        n_components = clamp(int(fraction * n_features), 1, n_features)

    This matches your requested PCA_features(x%) definition.
    """
    def __init__(self, fraction: float = 0.25, whiten: bool = True, random_state: int = 42):
        self.fraction = float(fraction)
        self.whiten = bool(whiten)
        self.random_state = int(random_state)
        self._pca: Optional[PCA] = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        # Fraction of features, but PCA is constrained by the training fold size:
        # n_components must be <= min(n_samples, n_features).
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
    scaler: str
    baseline: str
    simple: str
    pca: str


def build_transformers(cfg: SearchConfig, seed: int) -> List[Tuple[str, Any]]:
    """Instantiate sklearn-compatible transformers based on SearchConfig."""
    steps: List[Tuple[str, Any]] = []

    # 1) scaler
    if cfg.scaler == "StandardScaler":
        steps.append(("scaler", StandardScaler()))

    # 2) baseline (ASLS)
    if cfg.baseline == "ASLSBaseline":
        steps.append(("asls", ASLSBaseline()))

    # 3) simple spectral preprocessing
    if cfg.simple == "SavitzkyGolay":
        steps.append(("savgol", pp.SavitzkyGolay()))
    elif cfg.simple == "Baseline":
        steps.append(("baseline", pp.Baseline()))
    elif cfg.simple == "StandardNormalVariate":
        steps.append(("snv", pp.StandardNormalVariate()))
    elif cfg.simple == "Gaussian":
        steps.append(("gaussian", pp.Gaussian(order=2, sigma=1)))
    elif cfg.simple == "Normalize":
        steps.append(("normalize", pp.Normalize()))
    elif cfg.simple == "None":
        pass
    else:
        raise ValueError(f"Unknown simple preprocessing: {cfg.simple}")

    # 4) PCA
    if cfg.pca == "PCA_features_0.25":
        steps.append(("pca", PCAFeaturesFraction(fraction=0.25, whiten=True, random_state=seed)))

    return steps


def enumerate_search_space() -> List[SearchConfig]:
    """Enumerate the full cartesian product deterministically."""
    scalers = ["None", "StandardScaler"]
    baselines = ["None", "ASLSBaseline"]
    simples = ["None", "SavitzkyGolay", "Baseline", "StandardNormalVariate", "Gaussian", "Normalize"]
    pcas = ["None", "PCA_features_0.25"]

    out: List[SearchConfig] = []
    for sc in scalers:
        for bl in baselines:
            for sp in simples:
                for pc in pcas:
                    out.append(
                        SearchConfig(
                            scaler=sc,
                            baseline=bl,
                            simple=sp,
                            pca=pc,
                        )
                    )
    return out


# ==============================
# CV evaluation
# ==============================

@dataclass
class EvalResult:
    config: Dict[str, Any]
    mean_score: float
    fold_scores: List[float]


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
        kwargs["model_path"] = str(model_path)

    return ModelCls(**kwargs)


def score_one_split(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute the metric for a single split."""
    if task_type == "regression":
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        # We minimize RMSE -> use +RMSE as "loss-like" score
        return float(rmse)
    else:
        acc = accuracy_score(y_true, y_pred)
        # We maximize ACC -> use -ACC as "loss-like" score
        return float(-acc)


def cv_evaluate_config(
    dataset_folder: Path,
    cfg: SearchConfig,
    task_type: str,
    seed: int,
    n_splits: int,
    n_estimators: int,
    model_path: Optional[str],
    tmp_root: Optional[Path] = None,
) -> EvalResult:
    """
    Evaluate a configuration with SPXYG K-fold CV on the training set.
    Uses a local temporary folder if provided (useful to avoid collisions).
    """
    set_deterministic(seed)

    X_train, y_train, _, _ = load_dataset_folder(dataset_folder)
    X = X_train.to_numpy()
    y = y_train.to_numpy()

    split = SPXYGFold(n_splits=int(n_splits), random_state=int(seed))

    # SPXYGFold is designed for NIRS spectra; we use its split indices.
    # It exposes a sklearn-like API in nirs4all; we rely on split.split(X, y).

    folds = list(split.split(X, y))

    fold_scores: List[float] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        steps = build_transformers(cfg, seed=seed)

        model = make_model(task_type=task_type, seed=seed, n_estimators=n_estimators, model_path=model_path)
        pipe = Pipeline(steps + [("model", model)])

        # Fit and predict
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xva)

        fold_scores.append(score_one_split(task_type, yva, yhat))

    mean_score = float(np.mean(fold_scores))

    return EvalResult(
        config=asdict(cfg),
        mean_score=mean_score,
        fold_scores=fold_scores,
    )


# ==============================
# Final refit + predict
# ==============================

def refit_and_predict_best(
    dataset_folder: Path,
    best_cfg: SearchConfig,
    task_type: str,
    seed: int,
    n_estimators: int,
    model_path: Optional[str],
    outdir: Path,
) -> None:
    """Refit the best pipeline on full train and predict on Xtest; save artifacts."""
    set_deterministic(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_dataset_folder(dataset_folder)

    steps = build_transformers(best_cfg, seed=seed)
    model = make_model(task_type=task_type, seed=seed, n_estimators=n_estimators, model_path=model_path)
    pipe = Pipeline(steps + [("model", model)])

    pipe.fit(X_train.to_numpy(), y_train.to_numpy())
    y_pred = pipe.predict(X_test.to_numpy())

    pred_df = pd.DataFrame({"y_pred": np.asarray(y_pred).reshape(-1)})
    if y_test is not None and len(y_test) == len(pred_df):
        pred_df["y_true"] = np.asarray(y_test).reshape(-1)

    pred_path = outdir / f"{dataset_folder.name}__final_predictions.csv"
    pred_df.to_csv(pred_path, sep=";", decimal=".", index=False)

    # Optional: save pipeline
    try:
        import joblib
        joblib.dump(pipe, outdir / f"{dataset_folder.name}__final_pipeline.joblib")
    except Exception:
        # If joblib fails (e.g., due to TabPFN internals), skip saving the object.
        pass

def worker_eval_one_config(payload: Dict[str, Any]) -> EvalResult:
    """
    Top-level worker function (must be picklable for multiprocessing).
    It receives a serializable payload dict and returns an EvalResult.
    """
    dataset_folder = Path(payload["dataset_folder"])
    cfg_dict = payload["cfg"]
    cfg = SearchConfig(**cfg_dict)

    task_type = payload["task_type"]
    seed = int(payload["seed"])
    n_splits = int(payload["n_splits"])
    n_estimators = int(payload["n_estimators_search"])
    model_path = payload["model_path"]
    use_tmp_dir = bool(payload["use_tmp_dir"])
    dataset_name = payload["dataset_name"]

    set_deterministic(seed)

    tmp_root = None
    if use_tmp_dir:
        tmp_root = Path(tempfile.mkdtemp(prefix=f"tabpfn_search_{dataset_name}_"))

    try:
        return cv_evaluate_config(
            dataset_folder=dataset_folder,
            cfg=cfg,
            task_type=task_type,
            seed=seed,
            n_splits=n_splits,
            n_estimators=n_estimators,
            model_path=model_path,
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

    p.add_argument("--output_dir", type=str, default="Results/tabpfn_cartesian_search",
                   help="Output directory for CSV/JSON/predictions.")

    p.add_argument("--task_type", type=str, default="regression", choices=["regression", "classification"],
                   help="Task type for TabPFN model.")

    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--n_splits", type=int, default=3, help="Number of SPXYG folds for search (default: 3).")

    p.add_argument("--n_estimators_search", type=int, default=4,
                   help="TabPFN n_estimators during CV search.")
    p.add_argument("--n_estimators_final", type=int, default=8,
                   help="TabPFN n_estimators for the final refit.")

    p.add_argument("--model_path", type=str, default="",
                   help="Optional TabPFN checkpoint path (e.g., tabpfn-v2.5-regressor-v2.5_real.ckpt).")

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

    model_path = args.model_path.strip() if isinstance(args.model_path, str) else ""
    model_path = model_path if model_path else None

    space = enumerate_search_space()

    for ds in args.datasets:
        dataset_folder = Path(ds)
        if not dataset_folder.exists():
            raise FileNotFoundError(str(dataset_folder))

        print("=" * 100)
        print(f"Dataset: {dataset_folder.name}")
        print("=" * 100)
        print(f"Search configs: {len(space)} | SPXYG folds: {args.n_splits} | parallel={args.parallel} n_jobs={args.n_jobs}")
        print(f"TabPFN device: {TABPFN_DEVICE}")

        results: List[EvalResult] = []

        # Build serializable payloads (required for multiprocessing).
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
                "model_path": model_path,
                "use_tmp_dir": bool(args.use_tmp_dir),
            })

        if args.parallel and int(args.n_jobs) > 1:
            # Parallel evaluation: submit only top-level picklable function + pure-data payload.
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.n_jobs)) as ex:
                futs = [ex.submit(worker_eval_one_config, pl) for pl in payloads]

                print(f"Submitted {len(futs)} jobs to the process pool.", flush=True)

                with tqdm(
                    total=len(futs),
                    desc="Evaluating configs",
                    unit="cfg",
                    file=sys.stderr,          # tqdm renders reliably in logs
                    dynamic_ncols=True,
                    mininterval=0.5,
                    disable=False,          # Always show progress bar in this script
                ) as pbar:
                    for fut in concurrent.futures.as_completed(futs):
                        results.append(fut.result())
                        pbar.update(1)

        else:
            # Sequential evaluation
            for pl in payloads:
                results.append(worker_eval_one_config(pl))

        # Convert to DataFrame and select best
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
            scaler=str(best_row["scaler"]),
            baseline=str(best_row["baseline"]),
            simple=str(best_row["simple"]),
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
                    "tabpfn_device": TABPFN_DEVICE,
                    "n_estimators_search": int(args.n_estimators_search),
                    "n_estimators_final": int(args.n_estimators_final),
                    "model_path": model_path,
                    "best_config": asdict(best_cfg),
                    "best_mean_score": float(best_row["mean_score"]),
                    "best_fold_scores": json.loads(best_row["fold_scores"]),
                },
                f,
                indent=2,
            )

        print(f"✅ Best config (CV): {asdict(best_cfg)} | mean_score={float(best_row['mean_score']):.6f}")

        # Final refit + predict on Xtest
        refit_and_predict_best(
            dataset_folder=dataset_folder,
            best_cfg=best_cfg,
            task_type=args.task_type,
            seed=int(args.seed),
            n_estimators=int(args.n_estimators_final),
            model_path=model_path,
            outdir=outdir,
        )

        print(f"✅ Saved: {res_csv.name}, {best_json.name}, {dataset_folder.name}__final_predictions.csv")

    print("\nDONE.")


if __name__ == "__main__":
    main()
