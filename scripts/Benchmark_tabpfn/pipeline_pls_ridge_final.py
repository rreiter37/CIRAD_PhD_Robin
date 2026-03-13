#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_pls_ridge_final.py

Cartesian preprocessing search for PLS and Ridge on NIRS datasets,
with grouped-fold hyperparameter finetuning inspired by the practical
behavior of NIRS4ALL ``finetune_params``.

What this script does (per dataset folder):
- Cartesian search over preprocessing configurations
- For each preprocessing configuration, run Optuna finetuning inside grouped CV
- Aggregate fold scores with the mean RMSE, like a grouped finetuning strategy
- Select the best preprocessing + best model hyperparameters
- Refit on the full training set with the best configuration and best hyperparameters
- Predict on Xtest and save final predictions + fitted pipeline

Artifacts (relative to --output_dir):
- <dataset>__search_results.parquet
- <dataset>__best_config.json
- <dataset>__final_predictions.csv
- <dataset>__final_pipeline.joblib
- preds/<dataset>/predictions.parquet
"""

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
from tqdm.auto import tqdm
import sys
import re

import numpy as np
import pandas as pd
import uuid
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import optuna
from optuna.samplers import TPESampler, RandomSampler

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

from nirs4all.operators.splitters import SPXYGFold
import nirs4all.operators.transforms as pp
from nirs4all.operators.transforms import ASLSBaseline




def set_deterministic(seed: int) -> None:
    """Set as much determinism as possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_csv_strict(path: Path) -> pd.DataFrame:
    """Read CSV with the project convention: ';' separator and '.' decimal."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, sep=";", decimal=".")


def load_y_series(path: Path) -> pd.Series:
    """Load a one-column target file."""
    df = read_csv_strict(path)
    return df.iloc[:, 0]


def load_dataset_folder(folder: Path):
    """Load Xtrain, Ytrain, Xtest and optional Ytest from one dataset folder."""
    xtr = read_csv_strict(folder / "Xtrain.csv")
    ytr = load_y_series(folder / "Ytrain.csv")
    xte = read_csv_strict(folder / "Xtest.csv")

    yte_path = folder / "Ytest.csv"
    yte = load_y_series(yte_path) if yte_path.exists() else None
    return xtr, ytr, xte, yte


def _sanitize_token(token: str) -> str:
    """Sanitize a token for safe use in filenames."""
    token = str(token)
    token = token.strip().replace(" ", "_")
    token = re.sub(r"[^A-Za-z0-9_.+-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "NA"


@dataclass(frozen=True)
class SearchConfig:
    shape: str
    scatter: str
    final_repr: str


def build_preproc_tag(cfg: "SearchConfig") -> str:
    """Build a compact preprocessing tag."""
    used = []
    if cfg.shape != "None":
        used.append(cfg.shape)
    if cfg.scatter != "None":
        used.append(cfg.scatter)
    if cfg.final_repr != "None":
        used.append(cfg.final_repr)

    tag = "+".join(used) if used else "None"
    return _sanitize_token(tag)


def append_predictions_to_parquet(
    base_outdir: Path,
    dataset_name: str,
    split_name: str,
    preproc_tag: str,
    fold_id: int,
    model_name: str,
    best_params: Dict[str, Any],
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
):
    """
    Write predictions to a worker-safe shard parquet file.

    Important:
    - We do NOT append to a shared parquet file here.
    - Each call writes one independent shard file.
    - All shard files are merged later into a single predictions.parquet file.
    """
    split_name = "valid" if str(split_name).lower().startswith("v") else "test"

    shard_dir = Path(base_outdir) / "preds" / str(dataset_name) / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true is not None:
        y_true = np.asarray(y_true).reshape(-1)
    else:
        y_true = np.full(len(y_pred), np.nan)

    df = pd.DataFrame({
        "dataset": dataset_name,
        "split": split_name,
        "fold": int(fold_id),
        "preproc_tag": preproc_tag,
        "model": str(model_name),
        "best_params": json.dumps(best_params, sort_keys=True),
        "row_id": np.arange(len(y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
    })

    # Unique shard file per call: no concurrent overwrite risk.
    shard_name = (
        f"{_sanitize_token(model_name)}__"
        f"{_sanitize_token(preproc_tag)}__"
        f"{split_name}__fold_{int(fold_id)}__"
        f"{os.getpid()}__{uuid.uuid4().hex}.parquet"
    )
    shard_path = shard_dir / shard_name

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, shard_path, compression="zstd")

    return shard_path


def _make_savgol_from_name(name: str):
    """Build a Savitzky-Golay transformer from its encoded name."""
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


def build_transformers(cfg: SearchConfig) -> List[Tuple[str, Any]]:
    """Instantiate sklearn-compatible transformers based on SearchConfig."""
    steps: List[Tuple[str, Any]] = []

    # Shape correction block.
    if cfg.shape == "ASLSBaseline":
        steps.append(("asls", ASLSBaseline()))
    elif cfg.shape in {"SG_11_2_1", "SG_15_2_1", "SG_21_2_1", "SG_15_3_2", "SG_21_3_2"}:
        steps.append(("savgol", _make_savgol_from_name(cfg.shape)))
    elif cfg.shape == "Gaussian_1_2":
        steps.append(("gaussian", pp.Gaussian(order=1, sigma=2)))
    elif cfg.shape == "None":
        pass
    else:
        raise ValueError(f"Unknown shape preprocessing: {cfg.shape}")

    # Scatter correction / normalization block.
    if cfg.scatter == "SNV":
        steps.append(("snv", pp.StandardNormalVariate()))
    elif cfg.scatter == "EMSC":
        steps.append(("emsc", pp.nirs.ExtendedMultiplicativeScatterCorrection()))
    elif cfg.scatter == "None":
        pass
    else:
        raise ValueError(f"Unknown scatter preprocessing: {cfg.scatter}")

    # Final representation block.
    if cfg.final_repr == "Haar":
        steps.append(("haar", pp.Haar()))
    elif cfg.final_repr == "AreaNormalization":
        # Keep the original project convention used in the existing script.
        steps.append(("area_norm", pp.nirs.AreaNormalization()))
    elif cfg.final_repr == "None":
        pass
    else:
        raise ValueError(f"Unknown final representation preprocessing: {cfg.final_repr}")

    return steps


def enumerate_search_space() -> List[SearchConfig]:
    """Enumerate the full preprocessing cartesian product deterministically."""
    shapes = [
        "None",
        "ASLSBaseline",
        "SG_11_2_1",
        "SG_15_2_1",
        "SG_21_2_1",
        "SG_15_3_2",
        "SG_21_3_2",
        "Gaussian_1_2",
    ]
    scatters = ["None", "SNV", "EMSC"]
    final_reprs = ["None", "Haar", "AreaNormalization"]

    out: List[SearchConfig] = []
    for shape in shapes:
        for scatter in scatters:
            for final_repr in final_reprs:
                out.append(
                    SearchConfig(
                        shape=shape,
                        scatter=scatter,
                        final_repr=final_repr,
                    )
                )
    return out


def make_model(model_name: str, seed: int, model_params: Optional[Dict[str, Any]] = None) -> Any:
    """Build the regression model with explicit hyperparameters."""
    model_name = str(model_name).lower()
    model_params = model_params or {}

    if model_name == "pls":
        n_components = int(model_params.get("n_components", 2))
        return PLSRegression(n_components=n_components, scale=False)

    if model_name == "ridge":
        alpha = float(model_params.get("alpha", 1.0))
        return Ridge(alpha=alpha, random_state=int(seed))

    raise ValueError(f"Unknown model: {model_name}")


def score_one_split(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE for one split."""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return float(rmse)


@dataclass
class EvalResult:
    config: Dict[str, Any]
    model_name: str
    best_params: Dict[str, Any]
    mean_score: float
    fold_scores: List[float]
    trial_values: List[float]


def build_grouped_folds(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int):
    """Create deterministic SPXYG folds once and reuse them for grouped finetuning."""
    split = SPXYGFold(n_splits=int(n_splits), random_state=int(seed))
    return list(split.split(X, y))


def compute_safe_pls_upper_bound(
    X: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    pls_max_components: int,
) -> int:
    """Compute one global upper bound valid for every fold."""
    upper_bounds: List[int] = []
    for tr_idx, _ in folds:
        Xtr = X[tr_idx]
        upper_bounds.append(max(1, min(int(pls_max_components), Xtr.shape[0] - 1, Xtr.shape[1])))
    return int(min(upper_bounds))


def suggest_model_params(
    trial: optuna.trial.Trial,
    model_name: str,
    args_map: Dict[str, Any],
    safe_pls_upper: Optional[int] = None,
) -> Dict[str, Any]:
    """Suggest model hyperparameters for one Optuna trial."""
    model_name = str(model_name).lower()

    if model_name == "pls":
        if safe_pls_upper is None:
            raise ValueError("safe_pls_upper must be provided for PLS finetuning.")
        return {
            "n_components": trial.suggest_int("n_components", 1, int(safe_pls_upper)),
        }

    if model_name == "ridge":
        return {
            "alpha": trial.suggest_float(
                "alpha",
                float(args_map["ridge_alpha_min"]),
                float(args_map["ridge_alpha_max"]),
                log=True,
            ),
        }

    raise ValueError(f"Unknown model: {model_name}")


def _append_predictions(
    base_outdir: Path,
    dataset_name: str,
    model_name: str,
    cfg: SearchConfig,
    best_params: Dict[str, Any],
    fold_id: int,
    split_name: str,
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
) -> None:
    """Append predictions to the in-memory list and to the dataset Parquet file."""
    preproc_tag = build_preproc_tag(cfg)

    append_predictions_to_parquet(
        base_outdir=base_outdir,
        dataset_name=dataset_name,
        split_name=split_name,
        preproc_tag=preproc_tag,
        fold_id=fold_id,
        model_name=model_name,
        best_params=best_params,
        y_true=y_true,
        y_pred=y_pred,
    )


def evaluate_fixed_params_on_folds(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cfg: SearchConfig,
    model_name: str,
    model_params: Dict[str, Any],
    seed: int,
    dataset_name: str,
    outdir: Path,
) -> Tuple[float, List[float]]:
    """Evaluate one fixed preprocessing + parameter set on all folds."""
    fold_scores: List[float] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        steps = build_transformers(cfg)
        model = make_model(model_name=model_name, seed=seed, model_params=model_params)
        pipe = Pipeline(steps + [("model", model)])

        pipe.fit(Xtr, ytr)
        yhat = np.asarray(pipe.predict(Xva)).reshape(-1)
        fold_scores.append(score_one_split(yva, yhat))

        _append_predictions(
            base_outdir=outdir,
            dataset_name=dataset_name,
            model_name=model_name,
            cfg=cfg,
            best_params=model_params,
            fold_id=fold_id,
            split_name="valid",
            y_true=yva,
            y_pred=yhat,
        )

        yhat_test = np.asarray(pipe.predict(X_test)).reshape(-1)
        _append_predictions(
            base_outdir=outdir,
            dataset_name=dataset_name,
            model_name=model_name,
            cfg=cfg,
            best_params=model_params,
            fold_id=fold_id,
            split_name="test",
            y_true=y_test,
            y_pred=yhat_test,
        )

    mean_score = float(np.mean(fold_scores))
    return mean_score, fold_scores


def create_sampler(sampler_name: str, seed: int) -> optuna.samplers.BaseSampler:
    """Create the Optuna sampler requested by the user."""
    sampler_name = str(sampler_name).lower()
    if sampler_name == "tpe":
        return TPESampler(seed=int(seed))
    if sampler_name == "random":
        return RandomSampler(seed=int(seed))
    raise ValueError(f"Unknown sampler: {sampler_name}")


def finetune_one_config(
    dataset_folder: Path,
    cfg: SearchConfig,
    model_name: str,
    seed: int,
    n_splits: int,
    args_map: Dict[str, Any],
    dataset_name: str,
    outdir: Path,
) -> EvalResult:
    """Finetune model hyperparameters for one preprocessing configuration."""
    set_deterministic(seed)

    X_train, y_train, X_test, y_test = load_dataset_folder(dataset_folder)
    X = X_train.to_numpy()
    y = y_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_test_np = None if y_test is None else y_test.to_numpy()

    folds = build_grouped_folds(X=X, y=y, seed=seed, n_splits=n_splits)

    safe_pls_upper: Optional[int] = None
    if str(model_name).lower() == "pls":
        safe_pls_upper = compute_safe_pls_upper_bound(
            X=X,
            folds=folds,
            pls_max_components=int(args_map["pls_max_components"]),
        )

    trial_values: List[float] = []

    def objective(trial: optuna.trial.Trial) -> float:
        """Grouped objective: the same trial is scored on every fold."""
        set_deterministic(seed)
        model_params = suggest_model_params(
            trial=trial,
            model_name=model_name,
            args_map=args_map,
            safe_pls_upper=safe_pls_upper,
        )

        fold_scores: List[float] = []
        for tr_idx, va_idx in folds:
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            steps = build_transformers(cfg)
            model = make_model(model_name=model_name, seed=seed, model_params=model_params)
            pipe = Pipeline(steps + [("model", model)])

            pipe.fit(Xtr, ytr)
            yhat = np.asarray(pipe.predict(Xva)).reshape(-1)
            fold_scores.append(score_one_split(yva, yhat))

        objective_value = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", [float(v) for v in fold_scores])
        trial_values.append(objective_value)
        return objective_value

    study = optuna.create_study(
        direction="minimize",
        sampler=create_sampler(args_map["finetune_sampler"], seed=seed),
    )
    study.optimize(objective, n_trials=int(args_map["n_trials"]))

    best_params = {k: (int(v) if k == "n_components" else float(v)) for k, v in study.best_params.items()}
    mean_score, fold_scores = evaluate_fixed_params_on_folds(
        X=X,
        y=y,
        X_test=X_test_np,
        y_test=y_test_np,
        folds=folds,
        cfg=cfg,
        model_name=model_name,
        model_params=best_params,
        seed=seed,
        dataset_name=dataset_name,
        outdir=outdir,
    )

    return EvalResult(
        config=asdict(cfg),
        model_name=str(model_name),
        best_params=best_params,
        mean_score=mean_score,
        fold_scores=fold_scores,
        trial_values=[float(v) for v in trial_values],
    )


def refit_and_predict_best(
    dataset_folder: Path,
    best_cfg: SearchConfig,
    model_name: str,
    best_params: Dict[str, Any],
    seed: int,
    outdir: Path,
) -> None:
    """Refit the best pipeline on full train and predict on Xtest."""
    set_deterministic(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_dataset_folder(dataset_folder)

    steps = build_transformers(best_cfg)
    model = make_model(model_name=model_name, seed=seed, model_params=best_params)
    pipe = Pipeline(steps + [("model", model)])

    pipe.fit(X_train.to_numpy(), y_train.to_numpy())
    y_pred = np.asarray(pipe.predict(X_test.to_numpy())).reshape(-1)

    pred_df = pd.DataFrame({"y_pred": y_pred})
    if y_test is not None and len(y_test) == len(pred_df):
        pred_df.insert(0, "y_true", np.asarray(y_test).reshape(-1))

    pred_path = outdir / f"{dataset_folder.name}__final_predictions.csv"
    pred_df.to_csv(pred_path, sep=";", decimal=".", index=False)

    try:
        import joblib
        joblib.dump(pipe, outdir / f"{dataset_folder.name}__final_pipeline.joblib")
    except Exception:
        pass


def worker_finetune_one_config(payload: Dict[str, Any]) -> EvalResult:
    """Picklable worker for multiprocessing."""
    dataset_folder = Path(payload["dataset_folder"])
    cfg = SearchConfig(**payload["cfg"])

    return finetune_one_config(
        dataset_folder=dataset_folder,
        cfg=cfg,
        model_name=str(payload["model"]),
        seed=int(payload["seed"]),
        n_splits=int(payload["n_splits"]),
        args_map=dict(payload["args_map"]),
        dataset_name=str(payload["dataset_name"]),
        outdir=Path(payload["outdir"]),
    )


def consolidate_prediction_shards(base_outdir: Path, dataset_name: str) -> Optional[Path]:
    """
    Merge all worker-written shard parquet files into a single predictions.parquet.

    Final output:
        preds/<dataset>/predictions.parquet
    """
    dataset_pred_dir = Path(base_outdir) / "preds" / str(dataset_name)
    shard_dir = dataset_pred_dir / "_shards"
    final_path = dataset_pred_dir / "predictions.parquet"

    if not shard_dir.exists():
        return None

    shard_paths = sorted(shard_dir.glob("*.parquet"))
    if not shard_paths:
        return None

    tables = []
    for shard in shard_paths:
        tables.append(pq.read_table(shard))

    merged = pa.concat_tables(tables, promote_options="default")
    pq.write_table(merged, final_path, compression="zstd")

    return final_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset folder paths. Each folder must contain Xtrain.csv, Ytrain.csv and Xtest.csv.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="Results/pls_ridge_final",
        help="Output directory for CSV, JSON, Parquet predictions and fitted pipeline artifacts.",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["pls", "ridge"],
        help="Regression model to evaluate.",
    )

    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument("--n_splits", type=int, default=3, help="Number of SPXYG folds for grouped CV.")

    p.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel evaluation across preprocessing configurations.",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of worker processes used when --parallel is enabled.",
    )

    p.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of Optuna trials for each preprocessing configuration.",
    )
    p.add_argument(
        "--finetune_sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random"],
        help="Sampler used for grouped model hyperparameter optimization.",
    )
    p.add_argument(
        "--finetune_metric",
        type=str,
        default="rmse",
        choices=["rmse"],
        help="Optimization metric used during grouped cross-validation.",
    )
    p.add_argument(
        "--finetune_eval_mode",
        type=str,
        default="mean",
        choices=["mean"],
        help="How fold scores are aggregated across grouped CV.",
    )
    p.add_argument(
        "--pls_max_components",
        type=int,
        default=30,
        help="Hard upper bound for PLS n_components before fold-wise safety clipping.",
    )
    p.add_argument(
        "--ridge_alpha_min",
        type=float,
        default=1e-3,
        help="Lower bound of the Ridge alpha search space.",
    )
    p.add_argument(
        "--ridge_alpha_max",
        type=float,
        default=100.0,
        help="Upper bound of the Ridge alpha search space.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(int(args.seed))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    space = enumerate_search_space()

    args_map = {
        "n_trials": int(args.n_trials),
        "finetune_sampler": str(args.finetune_sampler),
        "finetune_metric": str(args.finetune_metric),
        "finetune_eval_mode": str(args.finetune_eval_mode),
        "pls_max_components": int(args.pls_max_components),
        "ridge_alpha_min": float(args.ridge_alpha_min),
        "ridge_alpha_max": float(args.ridge_alpha_max),
    }

    for ds in args.datasets:
        dataset_folder = Path(ds)
        if not dataset_folder.exists():
            raise FileNotFoundError(str(dataset_folder))

        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_folder.name}")
        print(f"MODEL:   {args.model}")
        print(f"TRIALS:  {args.n_trials}")
        print("=" * 100)

        parquet_path = outdir / "preds" / dataset_folder.name / "predictions.parquet"
        if parquet_path.exists():
            parquet_path.unlink()

        results: List[EvalResult] = []

        payloads: List[Dict[str, Any]] = []
        for cfg in space:
            payloads.append(
                {
                    "dataset_folder": str(dataset_folder),
                    "dataset_name": dataset_folder.name,
                    "cfg": asdict(cfg),
                    "model": str(args.model),
                    "seed": int(args.seed),
                    "n_splits": int(args.n_splits),
                    "args_map": args_map,
                    "outdir": str(outdir),
                }
            )

        if args.parallel and int(args.n_jobs) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(args.n_jobs)) as ex:
                futs = [ex.submit(worker_finetune_one_config, pl) for pl in payloads]
                print(f"Submitted {len(futs)} jobs to the process pool.", flush=True)

                with tqdm(
                    total=len(futs),
                    desc="Finetuning configs",
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
            for pl in tqdm(payloads, desc="Finetuning configs", unit="cfg"):
                results.append(worker_finetune_one_config(pl))

        df = pd.DataFrame(
            [
                {
                    **r.config,
                    "model": r.model_name,
                    "best_params": json.dumps(r.best_params, sort_keys=True),
                    "mean_score": r.mean_score,
                    "fold_scores": json.dumps(r.fold_scores),
                    "trial_values": json.dumps(r.trial_values),
                }
                for r in results
            ]
        )

        df = df.sort_values("mean_score", ascending=True).reset_index(drop=True)

        df.to_parquet(outdir / f"{dataset_folder.name}__search_results.parquet", index=False)

        best_row = df.iloc[0].to_dict()
        best_cfg = SearchConfig(
            shape=str(best_row["shape"]),
            scatter=str(best_row["scatter"]),
            final_repr=str(best_row["final_repr"]),
        )
        best_params = json.loads(best_row["best_params"])

        best_json = outdir / f"{dataset_folder.name}__best_config.json"
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": dataset_folder.name,
                    "model": str(args.model),
                    "seed": int(args.seed),
                    "n_splits": int(args.n_splits),
                    "n_trials": int(args.n_trials),
                    "finetune_strategy": {
                        "approach": "grouped",
                        "eval_mode": str(args.finetune_eval_mode),
                        "sampler": str(args.finetune_sampler),
                        "metric": str(args.finetune_metric),
                    },
                    "best_config": asdict(best_cfg),
                    "best_model_params": best_params,
                    "best_mean_score": float(best_row["mean_score"]),
                    "best_fold_scores": json.loads(best_row["fold_scores"]),
                    "trial_values": json.loads(best_row["trial_values"]),
                },
                f,
                indent=2,
            )

        refit_and_predict_best(
            dataset_folder=dataset_folder,
            best_cfg=best_cfg,
            model_name=str(args.model),
            best_params=best_params,
            seed=int(args.seed),
            outdir=outdir,
        )

        final_parquet = consolidate_prediction_shards(outdir, dataset_folder.name)
        if final_parquet is not None:
            print(f"Saved merged parquet predictions to: {final_parquet}")

        print(f"Saved best config to:       {best_json}")
        print(f"Saved predictions parquet:  {parquet_path}")


if __name__ == "__main__":
    main()
