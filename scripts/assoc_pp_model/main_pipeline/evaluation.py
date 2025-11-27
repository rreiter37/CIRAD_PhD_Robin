"""
Evaluation module for preprocessing/model combinations.

This creates a clean, modular, testable version of the original
evaluate_combination() logic from association_pp_model.py.

Responsibilities:
- Apply preprocessing
- Train model (PLS, LGBM, CNN, Ridge)
- Handle progressive optimization (best_trials propagation)
- Compute metrics (Regression or Classification)
- Manage batch size logging for CNN models
- Catch and return errors cleanly
"""

import torch
import time
import numpy as np
import warnings
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    root_mean_squared_error,
    accuracy_score,
    f1_score,
    confusion_matrix,
    minmax_scale,
)

from scripts.utils.ensure_dataframe import EnsureDataFrame
from scripts.utils.correct_class_unbalances import correct_class_unbalances
from scripts.Model_optim.pls_components_hybrid import get_pls_component_candidates


# ================================================================
# INTERNAL UTILS
# ================================================================

def _save_trained_model(pipeline, cfg, mdl_name, pp_name):
    """
    Save the trained pipeline model to:
    outputs/models/{data_source}/{mdl_name}/{mdl_name}_{pp_name}.joblib
    """
    data_source = cfg["data_source"]

    out_dir = os.path.join(
        "outputs", "models", data_source, mdl_name
    )
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{mdl_name}_{pp_name}.joblib"
    path = os.path.join(out_dir, filename)

    joblib.dump(pipeline, path)
    print(f"[INFO] Saved model → {path}")


def _compute_regression_metric(Y_true_original, Y_pred_original):
    """Compute normalized RMSE (as in original script)."""
    range_y = np.max(Y_true_original) - np.min(Y_true_original)
    rmse = root_mean_squared_error(Y_true_original, Y_pred_original)
    normalized = rmse / range_y if range_y != 0 else np.nan
    return normalized


def _compute_classification_metrics(Y_true, Y_pred):
    """Return accuracy, F1, and FPR."""
    acc = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred, average="weighted")

    cm = confusion_matrix(Y_true, Y_pred, labels=np.unique(Y_true))
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fpr = np.mean(fp / (fp + tn))
    return acc, f1, fpr


# ================================================================
# MAIN LOGIC
# ================================================================

def evaluate_combination(
    pp_name,
    pp_method,
    mdl_name,
    mdl,
    cfg,
    Xcal,
    Ycal,
    Xval,
    Yval
):
    """
    Evaluate one (preprocessing, model) combination.

    This function encapsulates:
    - building pipelines
    - progressive optimization (best_trials propagation)
    - model training
    - prediction
    - metrics computation
    - error catching
    - timing

    Returns a tuple standardized so the caller can aggregate results cleanly.
    """

    mode = cfg["mode"]
    progressive = cfg["progressive_optim"]
    adaptive_bs = cfg["adaptive_batch_size"]
    rd_seed = cfg["random_seed"]

    combo_start = time.time()

    # Copy data to avoid side effects
    X_train, X_test = np.asarray(Xcal), np.asarray(Xval)
    Y_train, Y_test = np.asarray(Ycal).ravel(), np.asarray(Yval).ravel()

    metrics_storage = []  # used for anomaly filtering as in original script

    # Determine if dataset is "large" to adjust PLS / LGBM configuration
    big_dataset = Xcal.shape[0] > 1e3

    try:
        # ------------------------------------------------------------
        # MODEL-SPECIFIC PARAMETER ADJUSTMENTS
        # ------------------------------------------------------------
        # --------------------- PLS ---------------------
        if mdl_name.startswith("PLS"):
            n_wl = Xcal.shape[1]
            total_eval = max(100, n_wl // 10)
            parallelism = big_dataset

            max_evals = total_eval // 5 if progressive else total_eval

            # Only use candidate components when progressive optimization is activated
            candidates = None
            if progressive:
                candidates = get_pls_component_candidates(
                    n_spectra=Xcal.shape[0],
                    n_wavelengths=n_wl,
                    prior_components=[],
                    max_evals=max_evals,
                    cv=5,
                    big_dataset=big_dataset,
                    rd_seed=rd_seed
                )

            if mode == "Regression":
                from scripts.Models.PLS.PLS_opti import AutoPLSRegression
                mdl = AutoPLSRegression(
                    cv=5,
                    scale=True,
                    seed=rd_seed,
                    candidate_components=candidates
                )
            else:
                from scripts.Models.PLS.PLS_opti_classif import AutoPLSDAClassifier
                mdl = AutoPLSDAClassifier(
                    cv=5,
                    scale=True,
                    seed=rd_seed,
                    candidate_components=candidates,
                    parallelism=parallelism
                )

        # --------------------- LGBM ---------------------
        elif mdl_name.startswith("LGBM"):
            if progressive:
                # progressive => first combination deep search, next reduced
                if mdl.best_trials is None:
                    n_trials = 100
                else:
                    n_trials = 20
            else:
                n_trials = 20

            cv = 3 if big_dataset else 5
            subsampling_rate = 0.3 if big_dataset else None

            if mode == "Regression":
                from scripts.Models.LGBM.LGBM_optuna import LGBMOptuna
                mdl = LGBMOptuna(
                    cv=cv,
                    n_trials=n_trials,
                    random_state=rd_seed,
                    verbose=0,
                    verbose_optuna=True,
                    scoring="neg_mean_squared_error",
                    best_trials=mdl.best_trials if progressive else None,
                    name_pp=pp_name,
                    subsampling_rate=subsampling_rate
                )
            else:
                from scripts.Models.LGBM.LGBM_optuna_classif import LGBMOptunaClassifier
                mdl = LGBMOptunaClassifier(
                    cv=cv,
                    n_trials=n_trials,
                    random_state=rd_seed,
                    verbose=0,
                    verbose_optuna=True,
                    scoring="log_loss",
                    best_trials=mdl.best_trials if progressive else None,
                    name_pp=pp_name,
                    subsampling_rate=subsampling_rate
                )

        # --------------------- CNN ---------------------
        elif mdl_name.startswith("CNN"):
            # Extract progressive Optuna settings from cfg
            if progressive:
                if mdl.best_trials is None:
                    default_trials = 500 if mode == "Regression" else 100
                    default_epochs_opt = 30
                else:
                    default_trials = 90 if mode == "Regression" else 20
                    default_epochs_opt = 10
            else:
                default_trials = 90
                default_epochs_opt = 10

            if mode == "Regression":
                from scripts.Models.DeepLearning.Train_predict.nicon_optuna import NiconOptunaRegressor
                mdl = NiconOptunaRegressor(
                    n_trials=default_trials,
                    epochs=10000,
                    patience=1000,
                    cyclic_learning=True,
                    lr_min=1e-6,
                    lr_max=1e-3,
                    epochs_optuna=default_epochs_opt,
                    random_state=rd_seed,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    verbose_optuna=True,
                    best_trials=mdl.best_trials if progressive else None,
                    name_pp=pp_name,
                    adaptive_batch_size=adaptive_bs
                )
            else:
                from scripts.Models.DeepLearning.Train_predict.nicon_optuna_classif import NiconOptunaClassifier
                mdl = NiconOptunaClassifier(
                    num_classes=len(np.unique(Y_train)),
                    n_trials=default_trials,
                    epochs=10000,
                    patience=10,
                    epochs_optuna=default_epochs_opt,
                    cyclic_learning=True,
                    lr_min=1e-6,
                    lr_max=1e-3,
                    random_state=rd_seed,
                    verbose_optuna=True,
                    best_trials=mdl.best_trials if progressive else None,
                    name_pp=pp_name
                )

        # ------------------------------------------------------------
        # BUILD SKLEARN PIPELINE
        # ------------------------------------------------------------
        if mdl_name.startswith("LGBM"):
            pipe = Pipeline(
                [
                    ("prep", pp_method),
                    ("ensure_df", EnsureDataFrame()),
                    ("model", clone(mdl)),
                ]
            )
        else:
            pipe = Pipeline(
                [
                    ("prep", pp_method),
                    ("model", clone(mdl)),
                ]
            )

        # ------------------------------------------------------------
        # FIT + PREDICT WITH WARNINGS CAPTURE
        # ------------------------------------------------------------
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pipe.fit(X_train, Y_train)

        Y_pred = pipe.predict(X_test)
        trained_model = pipe.named_steps["model"]

        # ------------------------------------------------------------
        # SAVE TRAINED MODEL PIPELINE
        # ------------------------------------------------------------
        try:
            _save_trained_model(pipe, cfg, mdl_name, pp_name)
        except Exception as e:
            print(f"[WARNING] Failed to save model {mdl_name} + {pp_name}: {e}")

        # ------------------------------------------------------------
        # METRIC COMPUTATION
        # ------------------------------------------------------------
        if mode == "Regression":
            # Inverse MinMax scaling is applied outside this module
            # => We assume upstream code already scaled/unscaled appropriately
            metric = _compute_regression_metric(Y_test, Y_pred)
        else:
            acc, f1, fpr = _compute_classification_metrics(Y_test, Y_pred)
            metric = acc  # the main metric for the heatmap

        metrics_storage.append(metric)

        # Filter unrealistic metrics as in original script
        mean_metric = np.nanmean(metrics_storage) if len(metrics_storage) > 0 else None
        if mode == "Regression" and mean_metric is not None and abs(metric) > 10 * mean_metric:
            metric = np.nan
        if mode == "Classification" and mean_metric is not None and metric < 0.4 * mean_metric:
            metric = np.nan

        # ------------------------------------------------------------
        # UPDATE PROGRESSIVE OPTIM best_trials
        # ------------------------------------------------------------
        updated_best = None

        if hasattr(trained_model, "best_trials"):
            updated_best = trained_model.best_trials

        # ------------------------------------------------------------
        # CNN batch size logging (static or dynamic)
        # ------------------------------------------------------------
        batch_info = None
        if mdl_name.startswith("CNN"):
            if adaptive_bs == "dynamic" and hasattr(trained_model, "batch_size_history"):
                batch_info = trained_model.batch_size_history
            elif hasattr(trained_model, "batch_size"):
                batch_info = trained_model.batch_size

        # ------------------------------------------------------------
        # TIME MEASUREMENT
        # ------------------------------------------------------------
        combo_time = time.time() - combo_start

        # ------------------------------------------------------------
        # RETURN FORMAT
        # ------------------------------------------------------------
        if mode == "Regression":
            return (
                pp_name,           # name of preprocessing
                mdl_name,          # model name
                metric,            # main regression metric (normalized RMSE)
                updated_best,      # best_trials (optional)
                combo_time,        # execution time
                batch_info         # CNN batch sizes (optional)
            )
        else:
            return (
                pp_name, mdl_name,
                metric,      # accuracy
                f1,          # F1-score
                fpr,         # FPR
                updated_best,
                combo_time,
                batch_info
            )

    # ============================================================
    # ERROR HANDLING
    # ============================================================
    except Exception as e:
        combo_time = time.time() - combo_start
        print(f"[ERROR] {pp_name} + {mdl_name}: {e}")

        if mode == "Regression":
            return (
                pp_name, mdl_name, np.nan, None, combo_time, None
            )
        else:
            return (
                pp_name, mdl_name,
                np.nan, np.nan, np.nan,
                None, combo_time, None
            )
