"""
Run NIRS4ALL pipelines for Ridge and PLS on multiple regression datasets, with:
- Cartesian preprocessing search
- Fine-tuning (n_trials) for each model
- Dataset discovery from an Excel file (DatabaseDetail.xlsx)
- Automatic Y header handling:
    * X*.csv always has a header row (wavelengths)
    * Y*.csv may or may not have a header row
  The script normalizes Y files (train/test) to a consistent format before passing datasets to NIRS4ALL.
- Final refit on full Xtrain/Ytrain using the best preprocessing + hyperparams (from validation),
  and prediction on Xtest (outside of NIRS4ALL, because it is not directly handled as requested).

Expected dataset folder structure:
Data/regression/{Database}/{Dataset}/Xtrain.csv
Data/regression/{Database}/{Dataset}/Ytrain.csv
Data/regression/{Database}/{Dataset}/Xtest.csv
Data/regression/{Database}/{Dataset}/Ytest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
        help="Root folder containing regression datasets organized as Data/regression/{Database}/{Dataset}/",
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


# ============================== CSV header detection & normalization ==============================

def count_total_lines_fast(csv_path: Path) -> int:
    """Count total lines in a CSV quickly (including header if present)."""
    n = 0
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def first_row_is_all_numeric(csv_path: Path, delimiter: str = ",") -> bool:
    """Return True if every cell in the first row can be parsed as float."""
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            row = next(reader)
        except StopIteration:
            return False

    if not row:
        return False

    for cell in row:
        cell = str(cell).strip()
        if cell == "":
            return False
        try:
            float(cell)
        except Exception:
            return False
    return True


def detect_y_has_header_by_matching_x(
    x_path: Path,
    y_path: Path,
    x_always_has_header: bool = True,
    delimiter: str = ",",
) -> bool:
    """
    Detect whether Y has a header row.

    Core idea:
    - X always has a header row (wavelengths) -> true sample count = total_lines(X) - 1
    - Y may or may not have a header -> sample count is either total_lines(Y) - 1 (if header)
      or total_lines(Y) (if no header)

    We pick the option that matches X sample count.
    If both match (rare) or neither match, we use a numeric test on the first row of Y:
      - non-numeric first row => header
      - numeric first row     => no header
    """
    x_lines = count_total_lines_fast(x_path)
    y_lines = count_total_lines_fast(y_path)

    if x_lines <= 1 or y_lines <= 0:
        # Degenerate files. Default to "no header" to avoid dropping data.
        return False

    x_samples = x_lines - 1 if x_always_has_header else x_lines
    y_samples_if_header = max(0, y_lines - 1)
    y_samples_if_no_header = y_lines

    match_header = (y_samples_if_header == x_samples)
    match_no_header = (y_samples_if_no_header == x_samples)

    if match_header and not match_no_header:
        return True
    if match_no_header and not match_header:
        return False

    # Tie-breaker fallback
    y_first_row_numeric = first_row_is_all_numeric(y_path, delimiter=delimiter)
    return not y_first_row_numeric


def normalize_dataset_folder(
    src_folder: Path,
    dst_folder: Path,
) -> Tuple[bool, str]:
    """
    Create a normalized copy of a dataset folder under dst_folder.

    Why?
    NIRS4ALL's loader can crash if X/Y row counts mismatch. Here, X always has a header
    while Y sometimes has a header and sometimes not. We normalize Y to a consistent format.

    Strategy:
    - Copy Xtrain/Xtest as-is (keep header wavelengths).
    - Rewrite Ytrain/Ytest to **NO HEADER** (pure numeric rows), always as a single-column CSV.

    Returns (ok, reason_if_failed).
    """
    dst_folder.mkdir(parents=True, exist_ok=True)

    required = ["Xtrain.csv", "Ytrain.csv", "Xtest.csv", "Ytest.csv"]
    missing = [f for f in required if not (src_folder / f).exists()]
    if missing:
        return False, f"Missing files: {missing}"

    # Copy X files as-is
    shutil.copyfile(src_folder / "Xtrain.csv", dst_folder / "Xtrain.csv")
    shutil.copyfile(src_folder / "Xtest.csv", dst_folder / "Xtest.csv")

    # Normalize Y files to "no header"
    for split in ["train", "test"]:
        x_path = src_folder / f"X{split}.csv"
        y_path = src_folder / f"Y{split}.csv"
        y_has_header = detect_y_has_header_by_matching_x(x_path, y_path)

        if y_has_header:
            y_df = pd.read_csv(y_path, header=0)
        else:
            y_df = pd.read_csv(y_path, header=None)

        if y_df.shape[1] < 1:
            return False, f"Empty Y{split}.csv"

        y_series = y_df.iloc[:, 0]
        # Force a clean numeric series when possible (keeps NaNs if any)
        y_series = pd.to_numeric(y_series, errors="coerce")

        # Save as single-column, no header
        y_out = dst_folder / f"Y{split}.csv"
        y_series.to_frame().to_csv(y_out, index=False, header=False)

    # Final sanity check on row counts: X has 1 header row.
    xtr_lines = count_total_lines_fast(dst_folder / "Xtrain.csv")
    ytr_lines = count_total_lines_fast(dst_folder / "Ytrain.csv")
    xtr_samples = max(0, xtr_lines - 1)
    ytr_samples = ytr_lines  # no header

    if xtr_samples != ytr_samples:
        return False, f"Train mismatch after normalization: X({xtr_samples}) Y({ytr_samples})"

    xte_lines = count_total_lines_fast(dst_folder / "Xtest.csv")
    yte_lines = count_total_lines_fast(dst_folder / "Ytest.csv")
    xte_samples = max(0, xte_lines - 1)
    yte_samples = yte_lines  # no header

    if xte_samples != yte_samples:
        return False, f"Test mismatch after normalization: X({xte_samples}) Y({yte_samples})"

    return True, ""


# ============================== Dataset discovery ==============================

def discover_datasets_from_excel(
    xlsx_path: Path,
    data_root: Path,
    normalized_root: Path,
    databases_filter: Optional[List[str]] = None,
    datasets_filter: Optional[List[str]] = None,
) -> List[str]:
    """
    Build a list of normalized dataset folder paths from DatabaseDetail.xlsx.

    The Excel is expected to include columns:
      - Database
      - Dataset

    Each source dataset folder is expected to contain: Xtrain.csv, Ytrain.csv, Xtest.csv, Ytest.csv.
    The script will create normalized copies under:
      {normalized_root}/{Database}/{Dataset}/...

    The returned list contains the normalized dataset folder paths.
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

        src_folder = data_root / db / ds
        if str(src_folder) in seen:
            continue

        # Validate expected CSV files
        expected = ["Xtrain.csv", "Ytrain.csv", "Xtest.csv", "Ytest.csv"]
        missing_files = [f for f in expected if not (src_folder / f).exists()]
        if missing_files:
            print(f"⚠️ Skipping {src_folder} (missing files: {missing_files})")
            continue

        # Normalize the dataset into workspace to avoid loader crashes
        dst_folder = normalized_root / db / ds
        ok, reason = normalize_dataset_folder(src_folder, dst_folder)
        if not ok:
            print(f"⚠️ Skipping {src_folder} ({reason})")
            continue

        dataset_paths.append(str(dst_folder))
        seen.add(str(src_folder))

    if not dataset_paths:
        raise RuntimeError(
            "No valid datasets found after filtering and normalization. "
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

    If it is a string representation, we cannot reliably reconstruct objects -> return None.
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
        steps = list(preprocessings.steps)
    else:
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

    Important:
    - X files have a header row (wavelengths) -> pandas default is fine.
    - Y files were normalized by this script to have **no header**.
    """
    xtr = pd.read_csv(dataset_folder / "Xtrain.csv")
    ytr = pd.read_csv(dataset_folder / "Ytrain.csv", header=None)

    ytr_s = ytr.iloc[:, 0] if ytr.shape[1] >= 1 else pd.Series(dtype=float)

    xte = pd.read_csv(dataset_folder / "Xtest.csv")
    yte_path = dataset_folder / "Ytest.csv"
    yte_s: Optional[pd.Series] = None
    if yte_path.exists():
        yte = pd.read_csv(yte_path, header=None)
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

    # Load raw (normalized) data
    X_train, y_train, X_test, y_test = _load_train_test_from_folder(dataset_folder)

    # Try to extract model object and hyperparams from best run
    model_obj = None
    for key in ["model", "estimator", "fitted_model"]:
        if key in best_run:
            model_obj = best_run[key]
            break

    if model_obj is None:
        # Fallback based on run name
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

    # We store normalized datasets under the workspace to avoid touching the original data.
    normalized_root = Path(args.workspace) / "normalized_data"

    # Discover + normalize datasets
    datasets = discover_datasets_from_excel(
        xlsx_path=xlsx_path,
        data_root=data_root,
        normalized_root=normalized_root,
        databases_filter=args.databases,
        datasets_filter=args.datasets_include,
    )

    print(f"Selected datasets (valid folders after normalization): {len(datasets)}")
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
            # DatasetConfigs keys are usually dataset folder names; we resolve a folder path by matching name.
            dataset_folder = None
            for d in datasets:
                if Path(d).name == dataset_name:
                    dataset_folder = Path(d)
                    break
            if dataset_folder is None:
                for d in datasets:
                    if str(d).endswith(dataset_name):
                        dataset_folder = Path(d)
                        break

            if dataset_folder is None:
                print(f"⚠️ [{dataset_name}] Could not resolve dataset folder path -> skipping final refit.")
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
