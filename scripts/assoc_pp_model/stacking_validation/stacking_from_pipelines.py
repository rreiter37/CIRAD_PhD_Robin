#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive stacking model based on selected (model, preprocessing) pairs from pipelines.
Automatically chooses between regression or classification stacking according to --mode.

For each selected pipeline:
  - Loads the (model, preprocessing) pairs marked as selected=True.
  - Builds a stacking ensemble using those base estimators.
  - Evaluates performance (RRMSE for regression, Accuracy for classification).
  - Saves global metrics and per-individual predictions (train/test) to CSV.
"""

import os
import argparse
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

# Import local preprocessing and model classes if needed
import nirs4all.operators.transformations as pp
from scripts.Models.PLS.PLS_opti import AutoPLSRegression
from scripts.Models.PLS.PLS_opti_classif import AutoPLSDAClassifier
from scripts.Models.Ridge.Ridge_opti import RidgeCVRegressor
from scripts.Models.Ridge.Ridge_opti_classif import RidgeCVClassifier
from scripts.Models.LGBM.LGBM_optuna import LGBMOptuna
from scripts.Models.LGBM.LGBM_optuna_classif import LGBMOptunaClassifier
from scripts.Models.DeepLearning.Train_predict.nicon_optuna import NiconOptunaRegressor
from scripts.Models.DeepLearning.Train_predict.nicon_optuna_classif import NiconOptunaClassifier
from scripts.utils.utils_bdd import split_data

# -----------------------------
# Helper functions
# -----------------------------

def load_selected_pairs(pipeline_name):
    """Load selected (model, preprocessing) pairs for a given pipeline."""
    base_dir = f"Results/assoc_pp_model/All_datasets/Pipeline_{pipeline_name}"
    candidates = [f for f in os.listdir(base_dir) if f.startswith("pipeline_") and f.endswith("_selected.csv")]
    if not candidates:
        print(f"[WARN] No selection CSV found for {pipeline_name}")
        return pd.DataFrame()
    df = pd.read_csv(os.path.join(base_dir, candidates[0]))
    return df[df["selected"] == True]


def get_preprocessing_object(prep_name, rd_seed=42):
    """Return the preprocessing object corresponding to a given name."""
    mapping = {
        "id": pp.IdentityTransformer(),
        "baseline": pp.Baseline(),
        "derivate": pp.Derivate(),
        "detrend": pp.Detrend(),
        "MSC": pp.MultiplicativeScatterCorrection(),
        "normalize": pp.Normalize(),
        "RNV": pp.RobustStandardNormalVariate(),
        "savgol": pp.SavitzkyGolay(),
        "simplescale": pp.SimpleScale(),
        "SNV": pp.StandardNormalVariate(),
        "haar": pp.Wavelet('haar'),
        "gaussian": pp.Gaussian(order=2, sigma=1),
        "PCA": PCA(random_state = rd_seed),
    }
    if prep_name in mapping:
        return mapping[prep_name]
    # For combined preprocessings like "MSC_SNV"
    if "_" in prep_name:
        steps = []
        for sub in prep_name.split("_"):
            if sub in mapping:
                steps.append((sub, mapping[sub]))
        return Pipeline(steps)
    return pp.IdentityTransformer()  # fallback


def get_model_object(model_name, mode, num_classes, random_state=42):
    """Return an instantiated model object based on its name."""
    if mode == "Regression":
        models = {
            "Ridge_reg": RidgeCVRegressor(random_state=random_state),
            "PLS_reg": AutoPLSRegression(seed=random_state),
            "LGBM_reg": LGBMOptuna(random_state=random_state, n_trials=10, verbose=0),
            "CNN_reg": NiconOptunaRegressor(random_state=random_state, n_trials=10, epochs_optuna=5)
        }
    else:
        models = {
            "Ridge_classif": RidgeCVClassifier(random_state=random_state),
            "PLS_classif": AutoPLSDAClassifier(seed=random_state),
            "LGBM_classif": LGBMOptunaClassifier(random_state=random_state, n_trials=10, verbose=0),
            "CNN_classif": NiconOptunaClassifier(num_classes=num_classes, random_state=random_state, n_trials=10, epochs_optuna=5)
        }
    return models.get(model_name, None)


# -----------------------------
# Main script
# -----------------------------

parser = argparse.ArgumentParser(description="Adaptive stacking using selected (model, preprocessing) pairs from pipelines.")
parser.add_argument("--mode", choices=["Regression", "Classification"], required=True, help="Task type.")
parser.add_argument("--data_source", required=True, help="Dataset name.")
parser.add_argument("--subset_pipelines", nargs="+", default=None,
                    help="List of pipelines to include (gatekeeping, graph_pruning, weakness_coverage, or 'all'). If None, all are used.")
parser.add_argument("--random_seed", type=int, default=42,
                    help="Global random seed (Default: 42).")
args = parser.parse_args()

mode = args.mode
data_source = args.data_source
subset = args.subset_pipelines
rd_seed = args.random_seed

# Load dataset (consistent with your other scripts)
Xcal, Ycal, Xval, Yval = split_data(mode, data_source, verbose=True)
num_classes = len(np.unique(Ycal))

pipelines_available = ["gatekeeping", "graph_pruning", "weakness_coverage"]
if subset is None or "all" in subset:
    subset = pipelines_available

print(f"[INFO] Pipelines considered: {subset}")

for pipeline_name in subset:
    df_sel = load_selected_pairs(pipeline_name)
    if df_sel.empty:
        print(f"[WARN] No selected pairs found for {pipeline_name}. Skipping.")
        continue

    print(f"[INFO] Building stacking model for pipeline: {pipeline_name} with {len(df_sel)} selected pairs")

    estimators = []
    for _, row in df_sel.iterrows():
        model_name = row["model_name"]
        prep_name = row["preprocessing_name"]
        base_model = get_model_object(model_name, mode, num_classes, random_state=rd_seed)
        prep_obj = get_preprocessing_object(prep_name, rd_seed=rd_seed)
        if base_model is not None:
            estimators.append((f"{model_name}_{prep_name}", Pipeline([
                ("prep", prep_obj),
                ("model", base_model)
            ])))
    # Build the meta-model (stacking)
    if mode == "Regression":
        meta_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        stack_model = StackingRegressor(estimators=estimators, final_estimator=meta_model, n_jobs=-1)
    else:
        meta_model = LogisticRegression(max_iter=2000)
        stack_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, n_jobs=-1)

    # Train stacking model
    stack_model.fit(Xcal, Ycal)

    # Predict on train/test
    y_pred_train = stack_model.predict(Xcal)
    y_pred_test = stack_model.predict(Xval)

    # Compute metrics
    if mode == "Regression":
        range_y = np.max(Ycal) - np.min(Ycal)
        rrms_train = np.sqrt(mean_squared_error(Ycal, y_pred_train)) / range_y
        rrms_test = np.sqrt(mean_squared_error(Yval, y_pred_test)) / range_y
        metrics = {"RRMSE_train": rrms_train, "RRMSE_test": rrms_test}
    else:
        acc_train = accuracy_score(Ycal, y_pred_train)
        acc_test = accuracy_score(Yval, y_pred_test)
        metrics = {"Accuracy_train": acc_train, "Accuracy_test": acc_test}

    print(f"[RESULTS] {pipeline_name} → {metrics}")

    # Save results
    output_dir = os.path.join("Results", "assoc_pp_model", "per_dataset", data_source)
    os.makedirs(output_dir, exist_ok=True)

    # Global metrics
    df_global = pd.DataFrame([{"pipeline": pipeline_name, **metrics}])
    df_global.to_csv(os.path.join(output_dir, f"stacking_global_{pipeline_name}.csv"), index=False)

    # Per-individual predictions
    df_individuals = pd.DataFrame({
        "pipeline": pipeline_name,
        "y_true_train": np.concatenate([Ycal, np.full_like(Yval, np.nan)]),
        "y_pred_train": np.concatenate([y_pred_train, np.full_like(Yval, np.nan)]),
        "y_true_test": np.concatenate([np.full_like(Ycal, np.nan), Yval]),
        "y_pred_test": np.concatenate([np.full_like(Ycal, np.nan), y_pred_test])
    })
    df_individuals.to_csv(os.path.join(output_dir, f"stacking_individuals_{pipeline_name}.csv"), index=False)

    # Save model
    dump(stack_model, os.path.join(output_dir, f"stacking_model_{pipeline_name}.joblib"))
    print(f"[INFO] Saved stacking model and results for pipeline '{pipeline_name}'.")
