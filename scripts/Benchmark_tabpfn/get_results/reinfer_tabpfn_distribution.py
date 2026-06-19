#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reinfer_tabpfn_distribution.py

Re-inference of TabPFN on the regression test sets to extract the *native
predictive distribution* (not only the point prediction), for the
TabPFN-Raw vs TabPFN-opt uncertainty analysis (manuscript_ACA, "Influence of
preprocessing" subsection).

Rationale
---------
The benchmark pipelines (``pipeline_tabpfn_final.py``) only persisted point
predictions (``output_type="mean"``). The uncertainty analysis needs the full
posterior predictive distribution that TabPFN returns. Because the best
preprocessing configuration per dataset is already fixed (stored in the master
results table), this script does *not* redo any preprocessing/hyperparameter
search: it simply refits the fixed pipeline on the full training set and runs a
single forward pass on the test set, from which it derives, per test sample:

- the predictive mean and median (point predictions),
- a dense grid of predictive quantiles (via ``criterion.icdf`` on the
  bar-distribution logits, i.e. a single forward pass), used downstream for
  prediction-interval coverage (reliability diagram), sharpness (NMPIW), and
  CRPS,
- the per-sample negative log-likelihood (via ``criterion.forward``),
- the per-sample CRPS (pinball-loss integral over the quantile grid).

The dataset-level aggregation (reliability diagram, NMPIW, coverage, CRPS/NLL
summaries) is performed by a separate post-processing script.

Notes
-----
- TabPFN is run on CPU by default, consistent with the hardware section of the
  manuscript (the GPU-enabled TabPFN stack was unavailable for this study).
- Determinism: seeds are fixed (Python/NumPy/PyTorch) for reproducibility.
- The bundled default TabPFN checkpoint is used unless ``--model-path`` is
  given. A sanity check reports the RMSE between the re-inferred mean prediction
  and the point prediction saved during the original benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Reuse the exact preprocessing/model construction used in the benchmark.
_THIS_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _THIS_DIR.parent  # scripts/Benchmark_tabpfn
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from pipeline_tabpfn_final import (  # noqa: E402
    SearchConfig,
    build_transformers,
    load_dataset_folder,
    make_model,
    set_deterministic,
)

from sklearn.pipeline import Pipeline  # noqa: E402


# ==============================
# Defaults / constants
# ==============================

#: Default quantile grid. Symmetric around the median and chosen so that the
#: standard central prediction intervals are exactly available:
#:   95% -> [0.025, 0.975], 90% -> [0.05, 0.95], 80% -> [0.10, 0.90],
#:   50% -> [0.25, 0.75].
DEFAULT_QUANTILE_GRID: List[float] = [
    round(float(t), 3) for t in np.arange(0.025, 0.9751, 0.025)
]

#: Central prediction-interval levels for the immediate per-dataset sanity log.
_HEADLINE_LEVEL = 0.90

MODEL_LABELS = ("TabPFN-Raw", "TabPFN-opt")


# ==============================
# Helpers
# ==============================

def resolve_data_folder(database: str, dataset: str, data_root: Path) -> Optional[Path]:
    """Locate the dataset folder under the regression data tree.

    Parameters
    ----------
    database : str
        Database name (e.g. ``"ALPINE"``).
    dataset : str
        Dataset folder name (e.g. ``"ALPINE_P_291_KS"``).
    data_root : Path
        Root of the data directory (expects ``regression/{DATABASE}/{dataset}``).

    Returns
    -------
    Optional[Path]
        The resolved folder if it exists and contains ``Xtest.csv``/``Ytest.csv``,
        otherwise ``None``.
    """
    candidates = [
        data_root / "regression" / str(database) / str(dataset),
    ]
    # Fall back to a glob if the database name does not match the folder layout.
    candidates += list((data_root / "regression").glob(f"*/{dataset}"))

    for folder in candidates:
        if (folder / "Xtest.csv").exists() and (folder / "Ytest.csv").exists():
            return folder
    return None


def searchconfig_from_best_config(best_config_json: str) -> SearchConfig:
    """Build a :class:`SearchConfig` from a master-results ``best_config_json``.

    Handles both key conventions encountered in the benchmark:
    ``{"shape", "scatter", "phase2"}`` (TabPFN-opt) and
    ``{"shape", "scatter", "pca"}`` (TabPFN-Raw, all ``"None"``).
    """
    cfg = json.loads(best_config_json) if isinstance(best_config_json, str) else {}
    shape = str(cfg.get("shape", "None"))
    scatter = str(cfg.get("scatter", "None"))
    phase2 = str(cfg.get("phase2", cfg.get("pca", "None")))
    return SearchConfig(shape=shape, scatter=scatter, phase2=phase2)


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Per-sample pinball (quantile) loss at level ``tau``."""
    diff = y - q
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def crps_from_quantiles(
    y_true: np.ndarray, quantiles: np.ndarray, taus: np.ndarray
) -> np.ndarray:
    """Approximate CRPS per sample as twice the mean pinball loss over a grid.

    CRPS = 2 * integral_0^1 pinball_tau(y, q_tau) dtau, approximated by the mean
    over the quantile grid. With a dense, regular grid this is an accurate and
    model-agnostic estimator of the CRPS.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Observed targets.
    quantiles : np.ndarray, shape (n, n_tau)
        Predictive quantiles at the levels in ``taus``.
    taus : np.ndarray, shape (n_tau,)
        Quantile levels.

    Returns
    -------
    np.ndarray, shape (n,)
        Per-sample CRPS estimate.
    """
    losses = np.empty_like(quantiles)
    for j, tau in enumerate(taus):
        losses[:, j] = pinball_loss(y_true, quantiles[:, j], float(tau))
    return 2.0 * losses.mean(axis=1)


# ==============================
# Core inference
# ==============================

def infer_distribution_for_dataset(
    data_folder: Path,
    cfg: SearchConfig,
    seed: int,
    n_estimators: int,
    model_path: Optional[str],
    device: str,
    taus: List[float],
) -> Dict[str, np.ndarray]:
    """Fit the fixed pipeline and extract the test predictive distribution.

    Returns a dict of per-sample arrays plus the training-target scale used for
    downstream normalization (NMPIW).
    """
    set_deterministic(seed)

    X_train, y_train, X_test, y_test = load_dataset_folder(data_folder)
    Xtr = X_train.to_numpy()
    ytr = y_train.to_numpy().astype(np.float64)
    Xte = X_test.to_numpy()
    yte = y_test.to_numpy().astype(np.float64) if y_test is not None else None

    # Build preprocessing (may be empty for TabPFN-Raw) and the TabPFN model
    # exactly as in the benchmark pipeline.
    steps = build_transformers(cfg, seed=seed)
    model = make_model(
        task_type="regression",
        seed=seed,
        n_estimators=n_estimators,
        model_path=model_path,
        device=device,
    )

    if steps:
        preproc = Pipeline(steps)
        Xtr_p = preproc.fit_transform(Xtr, ytr)
        Xte_p = preproc.transform(Xte)
    else:
        Xtr_p, Xte_p = Xtr, Xte

    model.fit(Xtr_p, ytr)

    # Single forward pass: full output gives the bar-distribution criterion and
    # logits, from which we derive every quantile (icdf) and the NLL (forward).
    full = model.predict(Xte_p, output_type="full")
    mean = np.asarray(full["mean"]).reshape(-1)
    median = np.asarray(full["median"]).reshape(-1)
    criterion = full["criterion"]
    logits = full["logits"]

    quantiles = np.stack(
        [np.asarray(criterion.icdf(logits, float(t)).detach().cpu()).reshape(-1)
         for t in taus],
        axis=1,
    )  # (n, n_tau)

    taus_arr = np.asarray(taus, dtype=np.float64)
    if yte is not None:
        y_t = torch.as_tensor(yte, dtype=logits.dtype)
        nll = np.asarray(criterion.forward(logits, y_t).detach().cpu()).reshape(-1)
        crps = crps_from_quantiles(yte, quantiles, taus_arr)
    else:
        nll = np.full(mean.shape[0], np.nan)
        crps = np.full(mean.shape[0], np.nan)

    return {
        "y_true": yte if yte is not None else np.full(mean.shape[0], np.nan),
        "y_pred": mean,
        "y_median": median,
        "nll": nll,
        "crps": crps,
        "quantiles": quantiles,
        "y_train_std": float(np.std(ytr)),
        "y_train_range": float(np.ptp(ytr)),
        "n_features": int(Xtr.shape[1]),
    }


def per_sample_dataframe(
    out: Dict[str, np.ndarray], taus: List[float], model: str, database: str,
    dataset: str,
) -> pd.DataFrame:
    """Assemble the per-sample long-ish table (one row per test sample)."""
    n = out["y_pred"].shape[0]
    df = pd.DataFrame({
        "model": model,
        "database": database,
        "dataset": dataset,
        "row_id": np.arange(n),
        "y_true": out["y_true"],
        "y_pred": out["y_pred"],
        "y_median": out["y_median"],
        "nll": out["nll"],
        "crps": out["crps"],
        # Training-target scale embedded so each parquet is self-contained for
        # downstream normalization (NMPIW, normalized CRPS), independently of the
        # meta table written at the end of the run.
        "y_train_std": out["y_train_std"],
        "y_train_range": out["y_train_range"],
    })
    for j, t in enumerate(taus):
        df[f"q{t:.3f}"] = out["quantiles"][:, j]
    return df


def headline_coverage_width(
    out: Dict[str, np.ndarray], taus: List[float], level: float = _HEADLINE_LEVEL
) -> Dict[str, float]:
    """Quick central-interval coverage and normalized width for the sanity log."""
    lo_t = round((1.0 - level) / 2.0, 3)
    hi_t = round((1.0 + level) / 2.0, 3)
    if lo_t not in taus or hi_t not in taus:
        return {"picp": np.nan, "nmpiw": np.nan}
    lo = out["quantiles"][:, taus.index(lo_t)]
    hi = out["quantiles"][:, taus.index(hi_t)]
    y = out["y_true"]
    scale = out["y_train_std"] if out["y_train_std"] > 0 else 1.0
    picp = float(np.mean((y >= lo) & (y <= hi))) if np.isfinite(y).all() else np.nan
    nmpiw = float(np.mean(hi - lo) / scale)
    return {"picp": picp, "nmpiw": nmpiw}


# ==============================
# CLI / main
# ==============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--master",
        type=str,
        default="Results/analysis/master/master_results.parquet",
        help="Path to the master results table (parquet).",
    )
    p.add_argument(
        "--data-root", type=str, default="Data",
        help="Root of the data directory (expects regression/{DB}/{dataset}).",
    )
    p.add_argument(
        "--output-dir", type=str, default="Results/tabpfn_uncertainty",
        help="Output directory for per-dataset predictive-distribution parquet files.",
    )
    p.add_argument(
        "--models", nargs="+", default=list(MODEL_LABELS),
        choices=list(MODEL_LABELS),
        help="Which TabPFN variants to re-infer.",
    )
    p.add_argument(
        "--datasets", nargs="*", default=None,
        help="Optional subset of dataset folder names to restrict to.",
    )
    p.add_argument(
        "--quantile-grid", nargs="*", type=float, default=None,
        help="Override the default quantile grid (levels in (0, 1)).",
    )
    p.add_argument(
        "--n-estimators", type=int, default=8,
        help="TabPFN n_estimators for the re-inference (default: 8).",
    )
    p.add_argument(
        "--model-path", type=str, default=None,
        help="Optional TabPFN checkpoint path; default uses the bundled checkpoint.",
    )
    p.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for TabPFN (default: cpu, consistent with the manuscript).",
    )
    p.add_argument(
        "--default-seed", type=int, default=42,
        help="Seed to use when the master row does not specify one.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many (model, dataset) pairs (smoke test).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Recompute datasets whose output parquet already exists.",
    )
    return p.parse_args()


def select_rows(master: pd.DataFrame, models: List[str],
                datasets: Optional[List[str]]) -> pd.DataFrame:
    """Select one regression row per (model, dataset) from the master table."""
    df = master[(master["task"] == "regression") & (master["model"].isin(models))].copy()
    if datasets:
        df = df[df["dataset"].isin(set(datasets))]
    # Keep a single row per (model, dataset): prefer rows with a usable config.
    df = df[df["best_config_json"].notna()]
    df = df.drop_duplicates(subset=["model", "dataset"], keep="first")
    return df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    taus = sorted(args.quantile_grid) if args.quantile_grid else list(DEFAULT_QUANTILE_GRID)
    taus = [round(float(t), 3) for t in taus]

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(args.master)
    rows = select_rows(master, args.models, args.datasets)
    if args.limit is not None:
        rows = rows.head(int(args.limit))

    print(f"TabPFN distribution re-inference | device={args.device} "
          f"| n_estimators={args.n_estimators} | quantile levels={len(taus)}")
    print(f"Selected (model, dataset) pairs: {len(rows)}")
    if args.model_path is None:
        print("Checkpoint: bundled TabPFN default (no --model-path given).")

    meta_records: List[Dict] = []
    failures: List[Dict] = []

    for i, r in rows.iterrows():
        model = str(r["model"])
        database = str(r["database_name"])
        dataset = str(r["dataset"])
        seed = int(r["seed"]) if pd.notna(r.get("seed")) else int(args.default_seed)
        tag = f"{model}__{dataset}"
        out_path = out_dir / "preds" / f"{tag}.parquet"

        if out_path.exists() and not args.overwrite:
            print(f"[{i + 1}/{len(rows)}] skip (exists): {tag}")
            continue

        folder = resolve_data_folder(database, dataset, data_root)
        if folder is None:
            print(f"[{i + 1}/{len(rows)}] MISSING data folder: {database}/{dataset}")
            failures.append({"model": model, "dataset": dataset, "error": "no_data_folder"})
            continue

        cfg = searchconfig_from_best_config(r["best_config_json"])

        try:
            out = infer_distribution_for_dataset(
                data_folder=folder, cfg=cfg, seed=seed,
                n_estimators=int(args.n_estimators), model_path=args.model_path,
                device=args.device, taus=taus,
            )
        except Exception as exc:  # noqa: BLE001 - log and continue the batch
            print(f"[{i + 1}/{len(rows)}] FAILED {tag}: {exc!r}")
            failures.append({"model": model, "dataset": dataset, "error": repr(exc)})
            continue

        df = per_sample_dataframe(out, taus, model, database, dataset)
        df.to_parquet(out_path, compression="zstd", index=False)

        # Sanity: compare re-inferred mean with the originally saved point preds.
        sanity_rmse = np.nan
        saved_path = r.get("final_predictions_path")
        if isinstance(saved_path, str) and Path(saved_path).exists():
            try:
                saved = pd.read_csv(saved_path, sep=";", decimal=".")
                if "y_pred" in saved and len(saved) == len(df):
                    sanity_rmse = float(np.sqrt(np.mean(
                        (saved["y_pred"].to_numpy() - out["y_pred"]) ** 2)))
            except Exception:  # noqa: BLE001
                pass

        head = headline_coverage_width(out, taus)
        meta_records.append({
            "model": model, "database": database, "dataset": dataset,
            "n_test": int(out["y_pred"].shape[0]), "n_features": out["n_features"],
            "seed": seed, "n_estimators": int(args.n_estimators),
            "preproc": "+".join(x for x in [cfg.shape, cfg.scatter, cfg.phase2]
                                 if x != "None") or "None",
            "y_train_std": out["y_train_std"], "y_train_range": out["y_train_range"],
            "mean_nll": float(np.nanmean(out["nll"])),
            "mean_crps": float(np.nanmean(out["crps"])),
            "picp@90": head["picp"], "nmpiw@90": head["nmpiw"],
            "sanity_rmse_vs_saved": sanity_rmse,
            "pred_path": str(out_path),
        })
        print(f"[{i + 1}/{len(rows)}] OK {tag} | n_test={out['y_pred'].shape[0]} "
              f"| cov@90={head['picp']:.3f} nmpiw@90={head['nmpiw']:.3f} "
              f"| CRPS={np.nanmean(out['crps']):.4f} "
              f"| sanityRMSE={sanity_rmse:.4g}")

    # Persist / update the meta table.
    if meta_records:
        meta_path = out_dir / "uncertainty_meta.csv"
        new_meta = pd.DataFrame(meta_records)
        if meta_path.exists() and not args.overwrite:
            old = pd.read_csv(meta_path)
            combined = pd.concat([old, new_meta], ignore_index=True)
            combined = combined.drop_duplicates(subset=["model", "dataset"], keep="last")
        else:
            combined = new_meta
        combined.to_csv(meta_path, index=False)
        print(f"\nMeta written: {meta_path} ({len(combined)} rows total)")

    if failures:
        fail_path = out_dir / "uncertainty_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"Failures: {len(failures)} (see {fail_path})")

    print("DONE.")


if __name__ == "__main__":
    main()
