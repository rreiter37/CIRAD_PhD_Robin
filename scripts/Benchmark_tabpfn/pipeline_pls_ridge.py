"""
Run NIRS4ALL pipelines for Ridge and PLS on multiple regression datasets, with:
- Cartesian preprocessing search
- Fine-tuning (n_trials) for each model
- Dataset discovery from an Excel file (DatabaseDetail.xlsx)
- Final refit on full Xtrain/Ytrain using the best preprocessing + hyperparams (from validation),
  and prediction on Xtest (outside of NIRS4ALL, because it is not directly handled as requested).

Expected dataset folder structure:
Data/Regression/{Database}/{Dataset}/Xtrain.csv
Data/Regression/{Database}/{Dataset}/Ytrain.csv
Data/Regression/{Database}/{Dataset}/Xtest.csv
Data/Regression/{Database}/{Dataset}/Ytest.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYGFold

# Transforms from nirs4all
from nirs4all.operators.transforms import (
    ASLSBaseline,
    Detrend,
    StandardNormalVariate,
    SavitzkyGolay,
    Gaussian,
    Haar,
)

from nirs4all.operators.transforms.nirs import (
    ExtendedMultiplicativeScatterCorrection as EMSC,
    AreaNormalization,
)


# ============================== CLI ==============================

def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()

    # Dataset discovery
    parser.add_argument(
        "--database_detail_xlsx",
        type=str,
        default="Data/DatabaseDetail.xlsx",
        help="Path to DatabaseDetail.xlsx used to discover datasets.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="Data/regression",
        help="Root folder containing regression datasets organized as Data/Regression/{Database}/{Dataset}/",
    )
    parser.add_argument(
        "--databases",
        nargs="*",
        default=None,
        help="Optional list of Database names to include (if omitted, include all from the Excel).",
    )
    parser.add_argument(
        "--datasets_include",
        nargs="*",
        default=None,
        help="Optional list of Dataset names to include (if omitted, include all from the Excel).",
    )

    # Pipeline runner
    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace_pls_ridge",
        help="Workspace folder where predictions and logs will be stored.",
    )
    parser.add_argument(
        "--group_key",
        type=str,
        default="ID",
        help="Grouping key for grouped CV. Use 'None' to disable grouping.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=3,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )

    # Finetune controls
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of optimization trials for each model.",
    )

    # Final refit controls
    parser.add_argument(
        "--do_final_refit",
        action="store_true",
        help="If set, refit best config on full train and predict final test.",
    )
    parser.add_argument(
        "--final_refit_rank_metric",
        type=str,
        default="rmse",
        help="Metric used to select the best run from validation. Usually 'rmse' for regression.",
    )

    return parser.parse_args()


# ============================== Dataset discovery ==============================

def discover_datasets_from_excel(
    xlsx_path: Path,
    data_root: Path,
    databases_filter: Optional[List[str]] = None,
    datasets_filter: Optional[List[str]] = None,
) -> List[str]:
    """
    Build a list of dataset folder paths from DatabaseDetail.xlsx.

    The Excel is expected to include columns:
      - Database
      - Dataset

    Each dataset folder is expected to contain: Xtrain.csv, Ytrain.csv, Xtest.csv, Ytest.csv.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Database detail file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    required_cols = {"Database", "Dataset"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {xlsx_path}: {sorted(missing)}")

    if databases_filter:
        df = df[df["Database"].astype(str).isin(set(databases_filter))]

    if datasets_filter:
        df = df[df["Dataset"].astype(str).isin(set(datasets_filter))]

    dataset_paths: List[str] = []
    seen = set()

    for _, row in df.iterrows():
        db = str(row["Database"]).strip()
        ds = str(row["Dataset"]).strip()

        folder = data_root / db / ds
        if str(folder) in seen:
            continue

        # Validate expected CSV files
        expected = ["Xtrain.csv", "Ytrain.csv", "Xtest.csv", "Ytest.csv"]
        missing_files = [f for f in expected if not (folder / f).exists()]

        if missing_files:
            # We skip silently but print a warning (common when some datasets are not exported yet)
            print(f"⚠️ Skipping {folder} (missing files: {missing_files})")
            continue

        dataset_paths.append(str(folder))
        seen.add(str(folder))

    if not dataset_paths:
        raise RuntimeError(
            "No valid datasets found after filtering and file checks. "
            "Please verify data_root and DatabaseDetail.xlsx."
        )

    return dataset_paths


# ============================== Pipeline definition ==============================

def build_pipeline(
    group_key: Optional[str],
    n_splits: int,
    random_state: int,
    n_trials: int,
) -> List[Dict[str, Any]]:
    """
    Build a NIRS4ALL pipeline with a cartesian preprocessing search and two models:
    - PLS with finetune over n_components in [1, 30]
    - Ridge with finetune over alpha in [1e-3, 100]
    """
    if group_key == "None":
        group_key = None

    # Cartesian preprocessing search space
    preprocessing_cartesian = {
        "_cartesian_": [
            {"_or_": [None, StandardScaler(), MinMaxScaler()]},
            {"_or_": [None, ASLSBaseline(), Detrend()]},
            {"_or_": [None, StandardNormalVariate(), EMSC()]},
            {"_or_": [None, SavitzkyGolay(window_length=15), Gaussian(order=1, sigma=2)]},
            {"_or_": [None, Haar(), AreaNormalization()]},
        ]
    }

    # IMPORTANT:
    # If grouping is disabled, DO NOT pass "group": None to NIRS4ALL splitters,
    # otherwise it raises: "Group column must be a string, got NoneType".
    split_step: Dict[str, Any] = {"split": SPXYGFold(n_splits=n_splits, random_state=random_state)}
    if group_key is not None:
        split_step["group"] = group_key

    pipeline = [
        # CV split (group-aware if enabled)
        split_step,

        # Expand all preprocessing combinations
        preprocessing_cartesian,

        # -------------------- PLS --------------------
        {
            "model": PLSRegression(),
            "name": "PLS",
            "finetune_params": {
                "n_trials": n_trials,
                "verbose": 0,
                "approach": "grouped",
                "eval_mode": "avg",
                "sample": "tpe",
                "model_params": {
                    "n_components": ("int", 1, 30),
                },
            },
        },

        # -------------------- Ridge --------------------
        {
            "model": Ridge(random_state=random_state),
            "name": "Ridge",
            "finetune_params": {
                "n_trials": n_trials,
                "approach": "grouped",
                "eval_mode": "avg",
                "model_params": {
                    "alpha": ("log_float", 1e-3, 100.0),
                },
            },
        },
    ]

    return pipeline


# ============================== Final refit utilities ==============================

def _safe_make_sklearn_pipeline(
    preprocessings: Any,
    model: Any,
) -> Optional[Pipeline]:
    """
    Best-effort conversion of NIRS4ALL 'preprocessings' to a sklearn Pipeline.

    We assume that `preprocessings` is either:
      - a list/tuple of sklearn-compatible transformers (or None entries),
      - or something already close to a pipeline.

    If it is a string representation, we cannot reliably reconstruct objects → return None.
    """
    if preprocessings is None:
        steps = []
    elif isinstance(preprocessings, (list, tuple)):
        steps = []
        for i, tr in enumerate(preprocessings):
            if tr is None:
                continue
            steps.append((f"pp_{i}", tr))
    elif isinstance(preprocessings, Pipeline):
        # If NIRS4ALL already returned a sklearn pipeline, just append the model
        steps = list(preprocessings.steps)
    else:
        # Often a string; cannot safely reconstruct
        return None

    steps.append(("model", model))
    return Pipeline(steps)


def _extract_best_run(
    dataset_prediction: Dict[str, Any],
    metric: str,
) -> Dict[str, Any]:
    """
    Extract the best run dict from predictions_per_dataset entry using validation ranking.

    This relies on NIRS4ALL Predictions API:
    run_predictions.top(1, metric=..., rank_partition="val", ...)
    """
    run_preds = dataset_prediction["run_predictions"]
    best_list = run_preds.top(
        1,
        metric=metric,
        rank_partition="val",
        display_metrics=[metric, "r2", "mape", "nrmse"],
    )
    if not best_list:
        raise RuntimeError("No runs found to select best configuration.")
    return best_list[0]


def _load_train_test_from_folder(dataset_folder: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:
    """
    Load Xtrain/Ytrain/Xtest/Ytest from a dataset folder.
    """
    xtr = pd.read_csv(dataset_folder / "Xtrain.csv")
    ytr = pd.read_csv(dataset_folder / "Ytrain.csv")

    # Y can be a single column; keep as Series
    if ytr.shape[1] == 1:
        ytr_s = ytr.iloc[:, 0]
    else:
        # If multiple columns, keep the first by default
        ytr_s = ytr.iloc[:, 0]

    xte = pd.read_csv(dataset_folder / "Xtest.csv")
    yte_path = dataset_folder / "Ytest.csv"
    yte_s: Optional[pd.Series] = None
    if yte_path.exists():
        yte = pd.read_csv(yte_path)
        yte_s = yte.iloc[:, 0] if yte.shape[1] >= 1 else None

    return xtr, ytr_s, xte, yte_s


def final_refit_and_predict(
    dataset_name: str,
    dataset_folder: Path,
    best_run: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Refit the best config (preprocessings + model hyperparams) on the full train set,
    then predict on Xtest and save artifacts to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    X_train, y_train, X_test, y_test = _load_train_test_from_folder(dataset_folder)

    # Try to extract model object and hyperparams from best run
    # We do a best-effort approach because NIRS4ALL run dict schemas can vary.
    model_obj = None
    for key in ["model", "estimator", "fitted_model"]:
        if key in best_run:
            model_obj = best_run[key]
            break

    if model_obj is None:
        # As a fallback, try to infer from name
        if "Ridge" in str(best_run.get("name", "")):
            model_obj = Ridge()
        elif "PLS" in str(best_run.get("name", "")):
            model_obj = PLSRegression()
        else:
            raise RuntimeError("Could not extract model object from best run.")

    # If we have params, apply them
    params_dict = None
    for key in ["model_params", "params", "best_params", "hyperparams"]:
        if key in best_run and isinstance(best_run[key], dict):
            params_dict = best_run[key]
            break

    model_to_fit = clone(model_obj)
    if params_dict:
        try:
            model_to_fit.set_params(**params_dict)
        except Exception:
            # We do not fail hard if keys do not match sklearn estimator
            pass

    preprocessings = best_run.get("preprocessings", None)
    sk_pipe = _safe_make_sklearn_pipeline(preprocessings, model_to_fit)

    if sk_pipe is None:
        print(f"⚠️ [{dataset_name}] Cannot reconstruct sklearn pipeline from preprocessings (likely a string).")
        print("    Skipping final refit for this dataset.")
        return

    # Fit on full train, predict on test
    sk_pipe.fit(X_train, y_train)
    y_pred = sk_pipe.predict(X_test)

    # Compute optional test metrics if y_test exists
    metrics_out: Dict[str, Any] = {}
    if y_test is not None and len(y_test) == len(y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        metrics_out = {"test_rmse": rmse, "test_r2": r2}

    # Save predictions
    pred_df = pd.DataFrame({"y_pred": np.asarray(y_pred).reshape(-1)})
    if y_test is not None and len(y_test) == len(y_pred):
        pred_df["y_true"] = np.asarray(y_test).reshape(-1)

    pred_path = output_dir / f"{dataset_name}_final_refit_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save fitted pipeline
    model_path = output_dir / f"{dataset_name}_final_refit_pipeline.joblib"
    joblib.dump(sk_pipe, model_path)

    # Save config snapshot
    cfg = {
        "dataset_name": dataset_name,
        "dataset_folder": str(dataset_folder),
        "selected_best_run_name": best_run.get("name", None),
        "selected_best_run_preprocessings": str(best_run.get("preprocessings", None)),
        "selected_best_run_params": params_dict,
        "final_refit_metrics": metrics_out,
    }
    cfg_path = output_dir / f"{dataset_name}_final_refit_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    msg = f"✅ [{dataset_name}] Final refit saved: {pred_path.name}, {model_path.name}, {cfg_path.name}"
    if metrics_out:
        msg += f" | test_rmse={metrics_out['test_rmse']:.6f}, test_r2={metrics_out['test_r2']:.6f}"
    print(msg)


# ============================== Main ==============================

def main():
    args = parse_args()

    xlsx_path = Path(args.database_detail_xlsx)
    data_root = Path(args.data_root)

    # Discover datasets
    datasets = discover_datasets_from_excel(
        xlsx_path=xlsx_path,
        data_root=data_root,
        databases_filter=args.databases,
        datasets_filter=args.datasets_include,
    )

    print(f"Selected datasets (valid folders): {len(datasets)}")
    for p in datasets[:10]:
        print(f"  • {p}")
    if len(datasets) > 10:
        print("  • ...")

    # Build pipeline config
    dataset_config = DatasetConfigs(datasets, task_type="regression")

    pipeline = build_pipeline(
        group_key=args.group_key,
        n_splits=args.n_splits,
        random_state=args.random_state,
        n_trials=args.n_trials,
    )
    pipeline_config = PipelineConfigs(pipeline, name="PLS_Ridge_Cartesian_Finetune")

    # Run NIRS4ALL pipeline
    runner = PipelineRunner(verbose=0, workspace_path=args.workspace)
    predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

    # Summary + optional final refit
    print("\n" + "=" * 120)
    print("DONE. Summary per dataset (top by val metric):")
    print("=" * 120)

    final_dir = Path(args.workspace) / "final_refit"
    for dataset_name, dataset_prediction in predictions_per_dataset.items():
        run_preds = dataset_prediction["run_predictions"]
        best = run_preds.top(
            1,
            metric=args.final_refit_rank_metric,
            rank_partition="val",
            display_metrics=[args.final_refit_rank_metric, "r2", "mape", "nrmse"],
        )[0]

        # Print a compact summary
        try:
            val_rmse = best["metrics"]["val"]["rmse"]
            print(f"- {dataset_name}: {best['name']} | val_rmse={val_rmse:.6f} | preprocess={best['preprocessings']}")
        except Exception:
            print(f"- {dataset_name}: {best.get('name')} | preprocess={best.get('preprocessings')}")

        # Final refit
        if args.do_final_refit:
            # DatasetConfigs uses dataset folder paths; we can reconstruct it from dataset_name if needed,
            # but NIRS4ALL typically keys by dataset name. We instead find the folder path by matching end part.
            dataset_folder = None
            for d in datasets:
                if Path(d).name == dataset_name:
                    dataset_folder = Path(d)
                    break
            if dataset_folder is None:
                # Fallback: match suffix
                for d in datasets:
                    if str(d).endswith(dataset_name):
                        dataset_folder = Path(d)
                        break

            if dataset_folder is None:
                print(f"⚠️ [{dataset_name}] Could not resolve dataset folder path → skipping final refit.")
                continue

            best_run = _extract_best_run(dataset_prediction, metric=args.final_refit_rank_metric)
            final_refit_and_predict(
                dataset_name=dataset_name,
                dataset_folder=dataset_folder,
                best_run=best_run,
                output_dir=final_dir,
            )


if __name__ == "__main__":
    main()
