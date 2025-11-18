#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual stacking based on (model, preprocessing) pairs selected by different pipelines.

For each requested pipeline:
  - Build base estimators as (preprocessing -> model) pipelines, following the logic of
    your association script (including EnsureDataFrame for LGBM models).
  - Perform manual stacking:
      * Generate out-of-fold predictions from base models on Xcal.
      * Train a meta-model on these stacked predictions.
      * Retrain base models on full Xcal and predict on Xval.
  - Save:
      * Global metrics (train/test) in CSV.
      * Per-individual predictions (train/test) in another CSV.

Author: ChatGPT (all comments in English).
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from itertools import combinations

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone

import nirs4all.operators.transformations as pp

from scripts.utils.ensure_dataframe import EnsureDataFrame
from scripts.utils.utils_bdd import split_data

from scripts.Models.PLS.PLS_opti import AutoPLSRegression
from scripts.Models.PLS.PLS_opti_classif import AutoPLSDAClassifier
from scripts.Models.Ridge.Ridge_opti import RidgeCVRegressor
from scripts.Models.Ridge.Ridge_opti_classif import RidgeCVClassifier
from scripts.Models.LGBM.LGBM_optuna import LGBMOptuna
from scripts.Models.LGBM.LGBM_optuna_classif import LGBMOptunaClassifier
from scripts.Models.DeepLearning.Train_predict.nicon_optuna import NiconOptunaRegressor
from scripts.Models.DeepLearning.Train_predict.nicon_optuna_classif import NiconOptunaClassifier


# ======================================================================
# Configuration for locating selected-pairs CSVs per pipeline
# ======================================================================

PIPELINE_FILE_CONFIG = {
    # logical_name: (dir_name, file_stub)
    "gatekeeping": ("gatekeeping", "gatekeeping"),
    "graph_pruning": ("graph", "graph_pruning"),
    "weakness_coverage": ("weakness_coverage", "weakness_coverage"),
}


# ======================================================================
# Build preprocessing dictionary (same style as association script)
# ======================================================================

def build_preprocessing_dict(random_state: int = 42):
    """Build a mapping from preprocessing_name to transformer."""
    simple_preprocs = [
        ('id', pp.IdentityTransformer()),
        ('baseline', pp.Baseline()),
        ('derivate', pp.Derivate()),
        ('detrend', pp.Detrend()),
        ('MSC', pp.MultiplicativeScatterCorrection()),
        ('normalize', pp.Normalize()),
        ('RNV', pp.RobustStandardNormalVariate()),
        ('savgol', pp.SavitzkyGolay()),
        ('simplescale', pp.SimpleScale()),
        ('SNV', pp.StandardNormalVariate()),
        ('haar', pp.Wavelet('haar')),
        ('gaussian', pp.Gaussian(order=2, sigma=1)),
    ]

    preprocessings = dict(simple_preprocs)

    # Add 2-combinations of simple preprocessing methods (excluding "id")
    for (name1, trans1), (name2, trans2) in combinations(simple_preprocs[1:], 2):
        combo_name = f"{name1}_{name2}"
        combo_pipeline = Pipeline([
            (name1, trans1),
            (name2, trans2)
        ])
        preprocessings[combo_name] = combo_pipeline

    # Add PCA
    preprocessings["PCA"] = PCA(random_state=random_state)

    return preprocessings


# ======================================================================
# Build models dict for Regression / Classification
# ======================================================================

def build_models_dict(mode: str, num_classes: int, random_state: int = 42):
    """Create a dictionary of base models keyed by full name (e.g. 'Ridge_reg')."""
    if mode == "Regression":
        models = {
            "Ridge_reg": RidgeCVRegressor(alphas=np.logspace(-4, 2, 50), cv=5, random_state=random_state),
            "PLS_reg": AutoPLSRegression(cv=3, seed=random_state),
            "LGBM_reg": LGBMOptuna(cv=5, n_trials=20, random_state=random_state,
                                   verbose=0, verbose_optuna=False),
            "CNN_reg": NiconOptunaRegressor(
                n_trials=30,
                epochs=200,
                patience=20,
                cyclic_learning=True,
                lr_min=1e-6,
                lr_max=1e-3,
                epochs_optuna=10,
                random_state=random_state,
                device="cuda" if False else "cpu",
                verbose_optuna=False,
            ),
        }
    else:
        models = {
            "Ridge_classif": RidgeCVClassifier(alphas=np.logspace(-4, 2, 50), cv=5, random_state=random_state),
            "PLS_classif": AutoPLSDAClassifier(cv=5, seed=random_state),
            "LGBM_classif": LGBMOptunaClassifier(cv=5, n_trials=30, random_state=random_state,
                                                 verbose=0, verbose_optuna=False),
            "CNN_classif": NiconOptunaClassifier(
                num_classes=num_classes,
                n_trials=30,
                epochs=200,
                patience=20,
                epochs_optuna=30,
                random_state=random_state,
                verbose_optuna=False,
            ),
        }
    return models


# ======================================================================
# Pipeline selection CSV loading
# ======================================================================

def load_selected_pairs_for_pipeline(pipeline_name: str) -> pd.DataFrame:
    """Load the CSV of selected (model, preprocessing) pairs for a given pipeline."""
    if pipeline_name not in PIPELINE_FILE_CONFIG:
        raise ValueError(f"Unknown pipeline_name: {pipeline_name}")

    dir_name, file_stub = PIPELINE_FILE_CONFIG[pipeline_name]
    base_dir = os.path.join("Results", "assoc_pp_model", "All_datasets", f"Pipeline_{dir_name}")
    csv_name = f"pipeline_{file_stub}_selected.csv"
    csv_path = os.path.join(base_dir, csv_name)

    if not os.path.exists(csv_path):
        print(f"[WARN] Selection file not found for '{pipeline_name}': {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if "selected" in df.columns:
        df = df[df["selected"] == True]

    return df


# ======================================================================
# Build base estimators list from selection
# ======================================================================

def build_base_estimators_from_selection(
        mode: str,
        df_sel: pd.DataFrame,
        preprocessings: dict,
        models_dict: dict,
):
    """
    Build a list of (name, estimator) for stacking, using selected pairs.

    mode: 'Regression' or 'Classification'
    df_sel: DataFrame with ['model_name', 'preprocessing_name', ...]
    preprocessings: dict {prep_name -> transformer}
    models_dict: dict {'Ridge_reg' -> model_instance, ...}
    """
    estimators = []
    suffix = "_reg" if mode == "Regression" else "_classif"

    if df_sel.empty:
        return estimators

    # Keep only models matching the current mode
    df_sel = df_sel[df_sel["model_name"].str.endswith(suffix)].copy()
    if df_sel.empty:
        return estimators

    df_sel = df_sel.drop_duplicates(subset=["model_name", "preprocessing_name"])

    for _, row in df_sel.iterrows():
        full_model_name = str(row["model_name"])
        prep_name = str(row["preprocessing_name"])

        if full_model_name not in models_dict:
            print(f"[WARN] Model '{full_model_name}' not in models_dict. Skipping.")
            continue
        if prep_name not in preprocessings:
            print(f"[WARN] Preprocessing '{prep_name}' not in preprocessing dict. Skipping.")
            continue

        base_model = clone(models_dict[full_model_name])
        prep_obj = preprocessings[prep_name]

        if full_model_name.startswith("LGBM"):
            pipe = Pipeline([
                ("prep", prep_obj),
                ("ensure_df", EnsureDataFrame()),
                ("model", base_model),
            ])
        else:
            pipe = Pipeline([
                ("prep", prep_obj),
                ("model", base_model),
            ])

        est_name = f"{full_model_name}_{prep_name}"
        estimators.append((est_name, pipe))

    return estimators


# ======================================================================
# Manual stacking (regression & classification)
# ======================================================================

def manual_stacking_regression(estimators, X_train, y_train, X_test, n_splits=5, random_state=42):
    """
    Manual stacking for regression:
      - Generate out-of-fold predictions (Z_train).
      - Fit RidgeCV on Z_train, y_train.
      - Retrain base models on full X_train and generate Z_train_full, Z_test.
    Returns:
      meta_model, Z_train_full, Z_test
    """
    n_samples = X_train.shape[0]
    n_estimators = len(estimators)

    Z_train = np.zeros((n_samples, n_estimators), dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Generate out-of-fold predictions
    for est_idx, (name, est) in enumerate(tqdm(estimators, desc="OF preds (reg)", unit="est")):
        Z_train_col = np.zeros(n_samples, dtype=float)
        for train_idx, val_idx in tqdm(kf.split(X_train, y_train),
                               total=kf.get_n_splits(),
                               desc=f"Folds for {name}",
                               leave=False):
            est_fold = clone(est)
            est_fold.fit(X_train[train_idx], y_train[train_idx])
            pred_val = est_fold.predict(X_train[val_idx])
            Z_train_col[val_idx] = pred_val
        Z_train[:, est_idx] = Z_train_col

    # Fit meta-model on stacked features
    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
    meta_model.fit(Z_train, y_train)

    # Retrain each estimator on full train and predict on train+test
    Z_train_full = np.zeros((n_samples, n_estimators), dtype=float)
    Z_test = np.zeros((X_test.shape[0], n_estimators), dtype=float)

    for est_idx, (name, est) in enumerate(tqdm(estimators, desc="Full fit (reg)", unit="est")):
        est_full = clone(est)
        est_full.fit(X_train, y_train)
        Z_train_full[:, est_idx] = est_full.predict(X_train)
        Z_test[:, est_idx] = est_full.predict(X_test)

    return meta_model, Z_train_full, Z_test


def manual_stacking_classification(estimators, X_train, y_train, X_test, n_splits=5, random_state=42):
    """
    Manual stacking for classification:
      - Generate out-of-fold predictions (discrete labels) from base estimators.
      - Fit LogisticRegression on these stacked predictions.
      - Retrain base models on full X_train and generate Z_train_full, Z_test.
    Returns:
      meta_model, Z_train_full, Z_test
    """
    n_samples = X_train.shape[0]
    n_estimators = len(estimators)

    Z_train = np.zeros((n_samples, n_estimators), dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for est_idx, (name, est) in enumerate(tqdm(estimators, desc="OF preds (classif)", unit="est")):
        Z_train_col = np.zeros(n_samples, dtype=float)
        for train_idx, val_idx in tqdm(skf.split(X_train, y_train),
                               total=skf.get_n_splits(),
                               desc=f"Folds for {name}",
                               leave=False):
            est_fold = clone(est)
            est_fold.fit(X_train[train_idx], y_train[train_idx])
            pred_val = est_fold.predict(X_train[val_idx])
            Z_train_col[val_idx] = pred_val
        Z_train[:, est_idx] = Z_train_col

    meta_model = LogisticRegression(max_iter=2000)
    meta_model.fit(Z_train, y_train)

    Z_train_full = np.zeros((n_samples, n_estimators), dtype=float)
    Z_test = np.zeros((X_test.shape[0], n_estimators), dtype=float)
    for est_idx, (name, est) in enumerate(tqdm(estimators, desc="Full fit (classif)", unit="est")):
        est_full = clone(est)
        est_full.fit(X_train, y_train)
        Z_train_full[:, est_idx] = est_full.predict(X_train)
        Z_test[:, est_idx] = est_full.predict(X_test)

    return meta_model, Z_train_full, Z_test


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Manual stacking using selected (model, preprocessing) pairs.")
    parser.add_argument("--mode", type=str, choices=["Regression", "Classification"], required=True,
                        help="Task type: 'Regression' or 'Classification'.")
    parser.add_argument("--data_source", type=str, required=True,
                        help="Dataset name (e.g., 'YamProtein').")
    parser.add_argument("--subset_pipelines", nargs="+", default=None,
                        help="Subset of pipelines to run among: 'all', 'gatekeeping', 'graph_pruning', 'weakness_coverage'. "
                             "'all' means baseline using all (model, preprocessing) combinations.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for out-of-fold stacking (default: 5).")

    args = parser.parse_args()
    mode = args.mode
    data_source = args.data_source
    subset = args.subset_pipelines
    rd_seed = args.random_seed
    n_folds = args.n_folds

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------
    Xcal, Ycal, Xval, Yval = split_data(mode, data_source, verbose=True)
    Xcal = np.asarray(Xcal)
    Xval = np.asarray(Xval)
    Ycal = np.asarray(Ycal).ravel()
    Yval = np.asarray(Yval).ravel()

    num_classes = len(np.unique(Ycal))

    # MinMax scaling for regression (same idea as association script)
    scaler_Y = None
    if mode == "Regression":
        scaler_Y = MinMaxScaler()
        Ycal = scaler_Y.fit_transform(Ycal.reshape(-1, 1)).ravel()
        Yval = scaler_Y.transform(Yval.reshape(-1, 1)).ravel()

    # -------------------------------------------------------------
    # Prepare preprocessings and models
    # -------------------------------------------------------------
    preprocessings = build_preprocessing_dict(random_state=rd_seed)
    models_dict = build_models_dict(mode=mode, num_classes=num_classes, random_state=rd_seed)

    # -------------------------------------------------------------
    # Pipelines to consider
    # -------------------------------------------------------------
    all_logical_pipelines = ["gatekeeping", "graph_pruning", "weakness_coverage"]

    if subset is None:
        pipelines_to_run = all_logical_pipelines
    else:
        if "all" in subset:
            pipelines_to_run = ["all"]
        else:
            pipelines_to_run = [p for p in subset if p in all_logical_pipelines]
            if not pipelines_to_run:
                print("[WARN] No valid pipelines in subset_pipelines. Falling back to all three.")
                pipelines_to_run = all_logical_pipelines

    print(f"[INFO] Pipelines considered: {pipelines_to_run}")

    # Output dir
    out_dir = os.path.join("Results", "assoc_pp_model", "per_dataset", data_source)
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Loop over pipelines
    # -------------------------------------------------------------
    for pipeline_name in tqdm(pipelines_to_run, desc="Pipelines", unit="pipeline"):
        print(f"\n[INFO] ===== Processing pipeline: {pipeline_name} =====")

        if pipeline_name == "all":
            # Baseline: all combinations of (model, preprocessing) consistent with mode
            estimators = []
            suffix = "_reg" if mode == "Regression" else "_classif"
            for model_key, base_model in models_dict.items():
                if not model_key.endswith(suffix):
                    continue
                for prep_name, prep_obj in preprocessings.items():
                    mdl = clone(base_model)
                    if model_key.startswith("LGBM"):
                        pipe = Pipeline([
                            ("prep", prep_obj),
                            ("ensure_df", EnsureDataFrame()),
                            ("model", mdl),
                        ])
                    else:
                        pipe = Pipeline([
                            ("prep", prep_obj),
                            ("model", mdl),
                        ])
                    est_name = f"{model_key}_{prep_name}"
                    estimators.append((est_name, pipe))
        else:
            # Restricted to selected pairs from pipeline_<name>_selected.csv
            df_sel = load_selected_pairs_for_pipeline(pipeline_name)
            if df_sel.empty:
                print(f"[WARN] No selected rows for pipeline '{pipeline_name}'. Skipping.")
                continue
            estimators = build_base_estimators_from_selection(
                mode=mode,
                df_sel=df_sel,
                preprocessings=preprocessings,
                models_dict=models_dict,
            )

        if not estimators:
            print(f"[WARN] No base estimators built for pipeline '{pipeline_name}'. Skipping.")
            continue

        print(f"[INFO] Number of base estimators for '{pipeline_name}': {len(estimators)}")

        # ---------------------------------------------------------
        # Manual stacking
        # ---------------------------------------------------------
        if mode == "Regression":
            meta_model, Z_train_full, Z_test = manual_stacking_regression(
                estimators=estimators,
                X_train=Xcal,
                y_train=Ycal,
                X_test=Xval,
                n_splits=n_folds,
                random_state=rd_seed,
            )

            y_pred_train_meta = meta_model.predict(Z_train_full)
            y_pred_test_meta = meta_model.predict(Z_test)

            # Back to original Y scale
            Ycal_orig = scaler_Y.inverse_transform(Ycal.reshape(-1, 1)).ravel()
            Yval_orig = scaler_Y.inverse_transform(Yval.reshape(-1, 1)).ravel()
            y_pred_train_orig = scaler_Y.inverse_transform(y_pred_train_meta.reshape(-1, 1)).ravel()
            y_pred_test_orig = scaler_Y.inverse_transform(y_pred_test_meta.reshape(-1, 1)).ravel()

            range_y = np.max(Ycal_orig) - np.min(Ycal_orig)
            rmse_train = np.sqrt(mean_squared_error(Ycal_orig, y_pred_train_orig))
            rmse_test = np.sqrt(mean_squared_error(Yval_orig, y_pred_test_orig))
            rrmse_train = rmse_train / range_y if range_y != 0 else np.nan
            rrmse_test = rmse_test / range_y if range_y != 0 else np.nan

            metrics = {
                "RRMSE_train": rrmse_train,
                "RRMSE_test": rrmse_test,
                "RMSE_train": rmse_train,
                "RMSE_test": rmse_test,
                "range_y": range_y,
            }

            y_true_train_to_save = Ycal_orig
            y_true_test_to_save = Yval_orig
            y_pred_train_to_save = y_pred_train_orig
            y_pred_test_to_save = y_pred_test_orig

        else:
            meta_model, Z_train_full, Z_test = manual_stacking_classification(
                estimators=estimators,
                X_train=Xcal,
                y_train=Ycal,
                X_test=Xval,
                n_splits=n_folds,
                random_state=rd_seed,
            )

            y_pred_train_meta = meta_model.predict(Z_train_full)
            y_pred_test_meta = meta_model.predict(Z_test)

            acc_train = accuracy_score(Ycal, y_pred_train_meta)
            acc_test = accuracy_score(Yval, y_pred_test_meta)
            f1_train = f1_score(Ycal, y_pred_train_meta, average="weighted")
            f1_test = f1_score(Yval, y_pred_test_meta, average="weighted")

            metrics = {
                "Accuracy_train": acc_train,
                "Accuracy_test": acc_test,
                "F1_train": f1_train,
                "F1_test": f1_test,
            }

            y_true_train_to_save = Ycal
            y_true_test_to_save = Yval
            y_pred_train_to_save = y_pred_train_meta
            y_pred_test_to_save = y_pred_test_meta

        print(f"[RESULTS] Pipeline '{pipeline_name}' metrics: {metrics}")

        # ---------------------------------------------------------
        # Save global metrics
        # ---------------------------------------------------------
        df_global = pd.DataFrame([{
            "data_source": data_source,
            "pipeline": pipeline_name,
            "mode": mode,
            "n_base_estimators": len(estimators),
            **metrics,
        }])

        global_path = os.path.join(out_dir, f"stacking_global_{pipeline_name}.csv")
        df_global.to_csv(global_path, index=False)
        print(f"[INFO] Global metrics saved to {global_path}")

        # ---------------------------------------------------------
        # Save per-individual predictions
        # ---------------------------------------------------------
        n_train = len(y_true_train_to_save)
        n_test = len(y_true_test_to_save)

        df_individuals = pd.DataFrame({
            "data_source": [data_source] * (n_train + n_test),
            "pipeline": [pipeline_name] * (n_train + n_test),
            "mode": [mode] * (n_train + n_test),
            "split": (["train"] * n_train) + (["test"] * n_test),
            "y_true": np.concatenate([y_true_train_to_save, y_true_test_to_save]),
            "y_pred": np.concatenate([y_pred_train_to_save, y_pred_test_to_save]),
        })

        indiv_path = os.path.join(out_dir, f"stacking_individuals_{pipeline_name}.csv")
        df_individuals.to_csv(indiv_path, index=False)
        print(f"[INFO] Per-individual predictions saved to {indiv_path}")


if __name__ == "__main__":
    main()