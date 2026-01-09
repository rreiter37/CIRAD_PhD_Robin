#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PLS-based preprocessing selection (cartesian Optuna) + final TabPFN training.

PHASE 1 (PLS, NO SAVING):
- Optuna selects preprocessing combinations (cartesian product)
- For each trial:
    - PLSRegression with n_components in [1..25]
    - n_components calibration is PARALLELIZED with joblib
- SPXY 3-fold CV on calibration
- Selection criterion: mean RMSE over folds
- NO artifacts, NO parquet

PHASE 2 (TabPFN, WITH SAVING):
- Same preprocessing as selected in Phase 1
- Final TabPFN pipeline
- Save only final metrics:
    * tabpfn_val_rmse_mean
    * tabpfn_val_rmse_best
    * tabpfn_test_rmse
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
from pathlib import Path
import numpy as np
import polars as pl
import torch
from dotenv import load_dotenv

from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import (
    ASLSBaseline,
    SavitzkyGolay,
    StandardNormalVariate as SNV,
    Detrend,
    Gaussian,
    Haar,
    Wavelet,
)
from nirs4all.operators.transforms.nirs import (
    AreaNormalization,
    ExtendedMultiplicativeScatterCorrection as EMSC,
)

from tabpfn import TabPFNRegressor

# =============================================================================
# Environment & TabPFN safety
# =============================================================================

def get_safe_tabpfn_device():
    if not torch.cuda.is_available():
        return "cpu"
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 9:
        print("⚠️ TabPFN CUDA unsupported → CPU fallback")
        return "cpu"
    return "cuda"

TABPFN_DEVICE = get_safe_tabpfn_device()

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--pls_n_jobs", type=int, default=-1,
                        help="Number of parallel jobs for PLS n_components calibration")
    return parser.parse_args()

args = parse_args()

# =============================================================================
# Dataset configuration
# =============================================================================

dataset_config = DatasetConfigs(
    args.datasets,
    task_type="regression"
)

# =============================================================================
# Utilities
# =============================================================================

def pp_fingerprint(pp):
    return " | ".join(type(op).__name__ for op in pp)

def evaluate_pls_n_components_parallel(
    X_train, y_train, X_val, y_val,
    n_components_range, n_jobs
):
    """
    Evaluate all PLS n_components in parallel and return the best RMSE.
    """

    def _eval_one(n_comp):
        model = PLSRegression(n_components=n_comp)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return mean_squared_error(y_val, y_pred, squared=False)

    rmses = Parallel(n_jobs=n_jobs)(
        delayed(_eval_one)(n) for n in n_components_range
    )
    return float(np.min(rmses))

# =============================================================================
# PHASE 1 — PLS preprocessing selection (NO SAVING)
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 1 – PLS cartesian preprocessing selection (joblib enabled)")
print("=" * 80)

pls_cartesian_pp = {
    "_cartesian_": [
        {"_or_": [None, SNV(), EMSC(), Detrend()]},
        {"_or_": [None, EMSC(), SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2)]},
        {"_or_": [None, SavitzkyGolay(window_length=15, deriv=1), SavitzkyGolay(window_length=15, deriv=2)]},
        {"_or_": [None, Haar(), Detrend(), AreaNormalization(), Wavelet("coif3")]},
    ]
}

pipeline_pls = [
    {"feature_augmentation": {"_or_": [None, ASLSBaseline()]}},
    {"feature_augmentation": {"_or_": [None, StandardScaler(), MinMaxScaler()]}},
    {"y_processing": MinMaxScaler()},
    {"split": SPXYGFold(n_splits=3, random_state=42)},
    {"feature_augmentation": pls_cartesian_pp},
    {
        "model": PLSRegression,
        "name": "PLS_selection",
        "finetune_params": {
            "n_trials": 25,
            "approach": "grouped",
            "eval_mode": "avg",
            "model_params": {
                "n_components": list(range(1, 26)),  # classical range
            },
        },
    },
]

runner_pls = PipelineRunner(
    workspace_path=args.workspace,
    verbose=0,
    save_artifacts=False,
    save_charts=False,
    enable_tab_reports=False,
    keep_datasets=False,
)

pls_cfg = PipelineConfigs(pipeline_pls, name="PLS_PP_SELECTION_NO_SAVE")
preds_pls, _ = runner_pls.run(pls_cfg, dataset_config)

best_pred = preds_pls.top(
    n=1,
    rank_metric="rmse",
    rank_partition="val"
)[0]

best_pp = runner_pls.manifest_manager.extract_generator_choice(
    best_pred,
    choice_index=0,
    instantiate=True
)

if not isinstance(best_pp, list):
    best_pp = [best_pp]

print("\n🏆 Selected preprocessing:")
print(pp_fingerprint(best_pp))

# =============================================================================
# PHASE 2 — Final TabPFN training (WITH SAVING)
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 2 – Final TabPFN calibration + test (WITH saving)")
print("=" * 80)

tabpfn_real_path = "tabpfn-v2.5-regressor-v2.5_real.ckpt"

pipeline_tabpfn = []
pipeline_tabpfn.extend(best_pp)

pipeline_tabpfn += [
    ASLSBaseline(),
    {"y_processing": StandardScaler()},
    StandardScaler(),
    {"split": SPXYGFold(n_splits=3, random_state=42)},
    {"_or_": [None, SavitzkyGolay()]},
    PCA(n_components=0.99, whiten=True, random_state=42),
    StandardScaler(),
    {
        "model": TabPFNRegressor(
            n_estimators=4,
            device=TABPFN_DEVICE,
            random_state=42,
            model_path=tabpfn_real_path,
            ignore_pretraining_limits=True,
        ),
        "name": "TabPFN-final",
    },
]

tabpfn_cfg = PipelineConfigs(
    pipeline_tabpfn,
    name="SOTA_TabPFN_SelectedPP"
)

runner_tabpfn = PipelineRunner(
    workspace_path=args.workspace,
    verbose=args.verbose,
    save_artifacts=True,
    save_charts=True,
    enable_tab_reports=True,
    keep_datasets=True,
)

preds_tabpfn, preds_per_ds = runner_tabpfn.run(
    tabpfn_cfg,
    dataset_config
)

# =============================================================================
# Save final metrics
# =============================================================================

for ds_name, ds_pred in preds_per_ds.items():
    run_preds = ds_pred["run_predictions"]

    val_scores = run_preds.get_scores(metric="rmse", partition="val")
    test_scores = run_preds.get_scores(metric="rmse", partition="test")

    val_mean = float(np.mean(val_scores))
    val_best = float(np.min(val_scores))
    test_rmse = float(np.mean(test_scores))

    print(f"\nDataset: {ds_name}")
    print(f"VAL mean RMSE: {val_mean:.5f}")
    print(f"VAL best RMSE: {val_best:.5f}")
    print(f"TEST RMSE:     {test_rmse:.5f}")

    df = run_preds._storage._df.with_columns([
        pl.lit(val_mean).alias("tabpfn_val_rmse_mean"),
        pl.lit(val_best).alias("tabpfn_val_rmse_best"),
        pl.lit(test_rmse).alias("tabpfn_test_rmse"),
        pl.lit(pp_fingerprint(best_pp)).alias("selected_preprocessing"),
    ])

    run_preds._storage._df = df
    run_preds.save()

print("\n✅ Finished successfully.")
