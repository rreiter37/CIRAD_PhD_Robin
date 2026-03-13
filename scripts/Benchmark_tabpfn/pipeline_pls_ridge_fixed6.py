# -*- coding: utf-8 -*-
"""
Run NIRS4ALL pipelines for Ridge and PLS on multiple regression datasets, with:
- Cartesian preprocessing search using the final NIRS search space discussed for PLS and Ridge- Fine-tuning (n_trials) for each model
- Shape correction block: None / ASLSBaseline / selected Savitzky-Golay variants / Gaussian(1,2)
- Scatter correction block: None / SNV / EMSC
- Final representation block: None / Haar / AreaNormalization
- Dataset discovery from an Excel file (DatabaseDetail.xlsx)
- Dataset files are used *as-is* from the provided directory (no normalization/copy of CSVs).
- Optional final refit on full Xtrain/Ytrain using the best preprocessing + hyperparams (from validation),
  and prediction on Xtest (outside of NIRS4ALL).

Expected dataset folder structure:
Data/regression/{Database}/{Dataset}/Xtrain.csv
Data/regression/{Database}/{Dataset}/Ytrain.csv
Data/regression/{Database}/{Dataset}/Xtest.csv
Data/regression/{Database}/{Dataset}/Ytest.csv

Optional metadata (if available):
Data/regression/{Database}/{Dataset}/Mtrain.csv
Data/regression/{Database}/{Dataset}/Mtest.csv

Important note about CSV formats:
- X*.csv always uses separator ';', decimals '.', and has a header row (wavelengths) with samples in rows.
- Y*.csv uses separator ';', decimals '.', and may or may not have a header row.
  Since NIRS4ALL typically reads CSVs with header=0, a header-less Y may drop a sample if not handled.
  This script does *not* rewrite any CSVs; datasets are consumed exactly as stored on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
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

def parse_args() -> argparse.Namespace:
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
        default="wk_pls_ridge",
        help="Workspace folder where predictions and logs will be stored.",
    )
    parser.add_argument(
        "--group_key",
        type=str,
        default="None",
        help=(
            "Grouping key for grouped CV. Use 'None' to disable grouping. "
            "If a dataset has no metadata or the column is missing, grouping is disabled for that dataset."
        ),
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
        default=10,
        help="Number of Optuna trials for each model and preprocessing configuration.",
    )

    # Model selection
    parser.add_argument(
        "--only_model",
        type=str,
        default="Both",
        choices=["Both", "PLS", "Ridge"],
        help=(
            "If set, run only a single model. "
            "Use 'PLS' to run only PLSRegression, 'Ridge' to run only Ridge regression, "
            "or 'Both' to run both models (default)."
        ),
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


# ============================== CSV helpers ==============================

def count_total_lines_fast(csv_path: Path) -> int:
    """Count total lines in a CSV quickly (including header if present)."""
    n = 0
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


def read_csv_fixed_sep(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV using the project's fixed conventions.

    Conventions enforced by this script:
    - Field separator is always ';'
    - Decimal separator is always '.'
    """
    return pd.read_csv(path, sep=";", decimal=".", **kwargs)


def detect_csv_delimiter(csv_path: Path, default: str = ";") -> str:
    """Lightweight delimiter detection using the first non-empty line.

    This is mainly defensive for edge-cases (e.g., metadata files). The project convention is ';'.
    """
    candidates = [";", ",", "\t", "|"]
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counts = {d: line.count(d) for d in candidates}
                best = max(counts, key=counts.get)
                return best if counts[best] > 0 else default
    except Exception:
        return default
    return default


def first_row_is_all_numeric(csv_path: Path, delimiter: str = ";") -> bool:
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
    delimiter: str = ";",
) -> bool:
    """Detect whether Y has a header row."""
    delimiter = detect_csv_delimiter(y_path, default=delimiter)

    x_lines = count_total_lines_fast(x_path)
    y_lines = count_total_lines_fast(y_path)

    if x_lines <= 1 or y_lines <= 0:
        # Default to "no header" to avoid dropping data.
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

    # Tie-breaker
    return not first_row_is_all_numeric(y_path, delimiter=delimiter)


def load_y_as_series_auto(y_path: Path, x_path: Path) -> pd.Series:
    """Load Y as a numeric Series, auto-detecting whether it contains a header."""
    y_has_header = detect_y_has_header_by_matching_x(x_path, y_path)
    if y_has_header:
        y_df = read_csv_fixed_sep(y_path, header=0)
        y = y_df.iloc[:, 0]
    else:
        y_df = read_csv_fixed_sep(y_path, header=None)
        y = y_df.iloc[:, 0]

    # Ensure numeric dtype. Also handle decimal-comma exports defensively.
    y = (
        y.astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    y = pd.to_numeric(y, errors="coerce")
    return y


def get_x_shape_fast(x_path: Path) -> Tuple[int, int]:
    """Return (n_samples, n_features) for Xtrain.csv quickly."""
    header_df = read_csv_fixed_sep(x_path, nrows=0)
    n_features = int(header_df.shape[1])

    n_lines = count_total_lines_fast(x_path)
    n_samples = max(0, n_lines - 1)  # subtract header
    return n_samples, n_features


def dataset_has_group_column(dataset_folder: Path, group_key: str) -> bool:
    """Return True if Mtrain.csv exists and contains the group column."""
    mtrain = dataset_folder / "Mtrain.csv"
    if not mtrain.exists():
        return False
    try:
        df = read_csv_fixed_sep(mtrain, header=0)
        return group_key in df.columns
    except Exception:
        return False


# ============================== Dataset discovery ==============================

def discover_datasets_from_excel(
    xlsx_path: Path,
    data_root: Path,
    databases_filter: Optional[List[str]] = None,
    datasets_filter: Optional[List[str]] = None,
) -> List[str]:
    """Discover datasets using DatabaseDetail.xlsx without rewriting any files."""
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
        folder_s = str(folder)
        if folder_s in seen:
            continue

        expected = ["Xtrain.csv", "Ytrain.csv", "Xtest.csv", "Ytest.csv"]
        missing_files = [f for f in expected if not (folder / f).exists()]
        if missing_files:
            print(f"⚠️ Skipping {folder} (missing files: {missing_files})")
            seen.add(folder_s)
            continue

        dataset_paths.append(folder_s)
        seen.add(folder_s)

    if not dataset_paths:
        raise RuntimeError(
            "No valid datasets found after filtering. Please verify data_root and DatabaseDetail.xlsx."
        )

    return dataset_paths


def compute_safe_pls_max_components(dataset_folder: Path, n_splits: int, hard_cap: int = 30) -> int:
    """Compute a safe upper bound for PLS n_components for a given dataset and CV scheme."""
    xtr = dataset_folder / "Xtrain.csv"
    n_samples, n_features = get_x_shape_fast(xtr)

    if n_samples < 3 or n_features < 1:
        return 0

    conservative_min_train = min(8, max(1, int(n_samples * 0.3)))
    if conservative_min_train < 2:
        return 0

    safe_upper = min(n_features, max(1, conservative_min_train - 1), hard_cap)
    return int(max(0, safe_upper))


# ============================== Pipeline definition ==============================

def build_pipeline(
    group_key: Optional[str],
    n_splits: int,
    random_state: int,
    n_trials: int,
    pls_max_components: int,
    only_model: str = "Both",
) -> List[Dict[str, Any]]:
    """Build a NIRS4ALL pipeline with the final cartesian preprocessing search space for PLS and Ridge."""
    preprocessing_cartesian = {
        "_cartesian_": [{"_or_": [None,
                                  ASLSBaseline(),
                                  SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
                                  SavitzkyGolay(window_length=15, polyorder=2, deriv=1),
                                  SavitzkyGolay(window_length=21, polyorder=2, deriv=1),
                                  SavitzkyGolay(window_length=15, polyorder=3, deriv=2),
                                  SavitzkyGolay(window_length=21, polyorder=3, deriv=2),
                                  Gaussian(order=1, sigma=2),
                                  ]
            },
            {"_or_": [None,
                      StandardNormalVariate(),
                      EMSC(),
                      ]
            },
            {"_or_": [None,
                      Haar(),
                      AreaNormalization(),
                      ]
            },
        ]
    }

    pls_max_components = int(pls_max_components)
    if pls_max_components < 1:
        print("⚠️ PLS skipped because pls_max_components < 1 (dataset too small for reliable PLS CV).")

    if only_model == "PLS" and pls_max_components < 1:
        raise ValueError("Cannot run only_model='PLS' because pls_max_components < 1 for this dataset.")

    split_step: Dict[str, Any] = {"split": SPXYGFold(n_splits=n_splits, random_state=random_state)}
    if group_key is not None:
        split_step["group"] = group_key

    pipeline: List[Dict[str, Any]] = [
        split_step,
        preprocessing_cartesian,
    ]

    # PLS
    if only_model in ("Both", "PLS") and pls_max_components >= 1:
        pipeline.append(
            {
                "model": PLSRegression(),
                "name": "PLS",
                "finetune_params": {
                    "n_trials": int(n_trials),
                    "verbose": 0,
                    "approach": "grouped",
                    "eval_mode": "avg",
                    "sample": "tpe",
                    "eval_metric": "rmse",
                    "metric": "rmse",
                    "model_params": {
                        "n_components": ("int", 1, int(pls_max_components)),
                    },
                },
            }
        )

    # Ridge
    if only_model in ("Both", "Ridge"):
        pipeline.append(
            {
                "model": Ridge(random_state=random_state),
                "name": "Ridge",
                "finetune_params": {
                    "n_trials": int(n_trials),
                    "approach": "grouped",
                    "eval_mode": "avg",
                    "sample": "tpe",
                    "eval_metric": "rmse",
                    "metric": "rmse",
                    "model_params": {
                        "alpha": ("log_float", 1e-3, 100.0),
                    },
                },
            }
        )

    return pipeline


# ============================== Final refit utilities ==============================

def _parse_preprocessings_string(s: str) -> List[Any]:
    """Parse a NIRS4ALL preprocessing string into instantiated transformer objects.

    NIRS4ALL may store preprocessings as a human-readable string (e.g., 'Detr', 'ASLS+SNV+SG').
    This parser is deliberately conservative and supports the operators used in this script's
    cartesian search space.
    """
    if s is None:
        return []
    s_norm = str(s).strip()
    if not s_norm or s_norm.lower() in ("none", "null", "nan"):
        return []

    # Normalize separators to '+'
    s_norm = (
        s_norm.replace("|", "+")
              .replace(",", "+")
              .replace(";", "+")
              .replace("/", "+")
              .replace("__", "_")
    )

    # Tokenize on '+' and whitespace
    raw_tokens: List[str] = []
    for part in s_norm.split("+"):
        part = part.strip()
        if not part:
            continue
        raw_tokens.extend([t for t in part.split() if t.strip()])

    tokens = [t.strip() for t in raw_tokens if t.strip()]
    tokens_l = [t.lower() for t in tokens]

    # Helper: detect token presence by substring matching (robust to abbreviated labels)
    def has_any(subs: List[str]) -> bool:
        return any(any(sub in t for sub in subs) for t in tokens_l)

    out: List[Any] = []

    # Baseline / detrend
    if has_any(["asls", "baseline"]):
        out.append(ASLSBaseline())
    if has_any(["detr", "detrend"]):
        out.append(Detrend())

    # Scatter correction
    if has_any(["snv", "standardnormalvariate", "standard_normal_variate"]):
        out.append(StandardNormalVariate())
    if has_any(["emsc", "extendedmultiplicativescattercorrection"]):
        out.append(EMSC())

    # Smoothing / derivatives
    # Savitzky-Golay default used in search space: window_length=15
    if has_any(["savgol", "savitzky", "savitzkygolay", "sg"]):
        out.append(SavitzkyGolay(window_length=15))
    # Gaussian default: order=1, sigma=2
    if has_any(["gauss", "gaussian"]):
        out.append(Gaussian(order=1, sigma=2))

    # Wavelets / area norm
    if has_any(["haar", "wavelet"]):
        out.append(Haar())
    if has_any(["area", "areanorm", "areanormalization"]):
        out.append(AreaNormalization())

    return out


def _safe_make_sklearn_pipeline(preprocessings: Any, model: Any) -> Optional[Pipeline]:
    """Best-effort conversion of NIRS4ALL 'preprocessings' to a sklearn Pipeline.

    Supports:
    - None
    - list/tuple of transformer objects
    - sklearn Pipeline
    - string labels produced by NIRS4ALL (limited to this script's operators)
    """
    if preprocessings is None:
        steps: List[Tuple[str, Any]] = []
    elif isinstance(preprocessings, (list, tuple)):
        steps = []
        for i, tr in enumerate(preprocessings):
            if tr is None:
                continue
            steps.append((f"pp_{i}", tr))
    elif isinstance(preprocessings, Pipeline):
        steps = list(preprocessings.steps)
    elif isinstance(preprocessings, str):
        pp_list = _parse_preprocessings_string(preprocessings)
        steps = [(f"pp_{i}", tr) for i, tr in enumerate(pp_list)]
    else:
        return None

    steps.append(("model", model))
    return Pipeline(steps)


def _extract_best_run(dataset_prediction: Dict[str, Any], metric: str) -> Dict[str, Any]:
    """Extract the best run dict using validation ranking."""
    run_preds = dataset_prediction["run_predictions"]
    best_list = run_preds.top(
        1,
        rank_metric=metric,
        rank_partition="val",
        display_metrics=[metric, "r2", "mape", "nrmse"],
    )
    if not best_list:
        raise RuntimeError("No runs found to select best configuration.")
    return best_list[0]


def _load_train_test_from_folder(dataset_folder: Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:
    """Load Xtrain/Ytrain/Xtest/Ytest from dataset folder using fixed CSV conventions."""
    xtr_path = dataset_folder / "Xtrain.csv"
    ytr_path = dataset_folder / "Ytrain.csv"
    xte_path = dataset_folder / "Xtest.csv"
    yte_path = dataset_folder / "Ytest.csv"

    X_train = read_csv_fixed_sep(xtr_path, header=0)
    y_train = load_y_as_series_auto(ytr_path, x_path=xtr_path)

    X_test = read_csv_fixed_sep(xte_path, header=0)

    y_test: Optional[pd.Series] = None
    if yte_path.exists():
        y_test = load_y_as_series_auto(yte_path, x_path=xte_path)

    return X_train, y_train, X_test, y_test


def final_refit_and_predict(
    dataset_name: str,
    dataset_folder: Path,
    best_run: Dict[str, Any],
    output_dir: Path,
    random_state: int = 42,
) -> None:
    """Refit the best config on full train then predict on Xtest; save predictions + model + config."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = _load_train_test_from_folder(dataset_folder)

    # Extract estimator (NIRS4ALL run dicts can differ across versions)
    model_obj = None
    for key in ("model", "estimator", "fitted_model"):
        if key in best_run and best_run[key] is not None:
            model_obj = best_run[key]
            break

    # Robust fallback: infer from the run label (works even if `name` is missing)
    if model_obj is None:
        label = _best_name_safe(best_run).lower()
        if "pls" in label:
            model_obj = PLSRegression()
        elif "ridge" in label:
            model_obj = Ridge(random_state=random_state)
        else:
            # Last resort: try a few alternative keys used by some NIRS4ALL versions
            for k in ("model_name", "estimator_name", "tag"):
                v = best_run.get(k, "")
                if isinstance(v, str) and v.strip():
                    vv = v.strip().lower()
                    if "pls" in vv:
                        model_obj = PLSRegression()
                        break
                    if "ridge" in vv:
                        model_obj = Ridge(random_state=random_state)
                        break

    if model_obj is None:
        raise RuntimeError("Could not extract model object from best run (no model instance and no recognizable label).")

    # Extract best hyperparams
    params_dict = None
    for key in ("model_params", "params", "best_params", "hyperparams"):
        if isinstance(best_run.get(key, None), dict):
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

    sk_pipe.fit(X_train, y_train)
    y_pred = sk_pipe.predict(X_test)

    metrics_out: Dict[str, Any] = {}
    if y_test is not None and len(y_test) == len(y_pred):
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        metrics_out = {"test_rmse": rmse, "test_r2": r2}

    pred_df = pd.DataFrame({"y_pred": np.asarray(y_pred).reshape(-1)})
    if y_test is not None and len(y_test) == len(y_pred):
        pred_df["y_true"] = np.asarray(y_test).reshape(-1)

    pred_path = output_dir / f"{dataset_name}_final_refit_predictions.csv"
    pred_df.to_csv(pred_path, sep=";", decimal=".", index=False)

    model_path = output_dir / f"{dataset_name}_final_refit_pipeline.joblib"
    joblib.dump(sk_pipe, model_path)

    cfg = {
        "dataset_name": dataset_name,
        "dataset_folder": str(dataset_folder),
        "selected_best_run_name": best_run.get("name", None),
        "selected_best_run_preprocessings": str(best_run.get("preprocessings", None)),
        "selected_best_run_params": params_dict,
        "final_refit_metrics": metrics_out,
    }
    cfg_path = output_dir / f"{dataset_name}_final_refit_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    msg = f"✅ [{dataset_name}] Final refit saved: {pred_path.name}, {model_path.name}, {cfg_path.name}"
    if metrics_out:
        msg += f" | test_rmse={metrics_out['test_rmse']:.6f}, test_r2={metrics_out['test_r2']:.6f}"
    print(msg)


# ============================== Main ==============================

def _best_name_safe(b: Dict[str, Any]) -> str:
    """Get a stable name for the best run across NIRS4ALL versions."""
    for k in ("name", "model_name", "estimator_name", "tag"):
        v = b.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for k in ("model", "estimator", "fitted_model"):
        est = b.get(k, None)
        if est is not None:
            return est.__class__.__name__
    return "UnknownModel"


def _resolve_dataset_folder(datasets: List[str], dataset_name: str) -> Optional[Path]:
    """Resolve dataset folder path from a possibly normalized dataset_name (case-insensitive)."""
    target = str(dataset_name).strip().lower()

    for d in datasets:
        p = Path(d)
        if p.name.strip().lower() == target:
            return p

    for d in datasets:
        if str(d).strip().lower().endswith(target):
            return Path(d)

    return None


def main() -> None:
    args = parse_args()

    xlsx_path = Path(args.database_detail_xlsx)
    data_root = Path(args.data_root)

    group_key = None if args.group_key == "None" else args.group_key

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

    runner = PipelineRunner(verbose=0, workspace_path=args.workspace)

    all_predictions_per_dataset: Dict[str, Any] = {}

    for d in datasets:
        ds_folder = Path(d)

        use_group_key = group_key if (group_key is not None and dataset_has_group_column(ds_folder, group_key)) else None
        if group_key is not None and use_group_key is None:
            print(f"⚠️ [{ds_folder.name}] Grouping disabled (missing metadata or group column '{group_key}').")

        pls_max = compute_safe_pls_max_components(ds_folder, n_splits=args.n_splits, hard_cap=30)
        if pls_max < 1:
            print(f"⚠️ [{ds_folder.name}] PLS will be skipped (safe pls_max_components={pls_max}).")
        else:
            print(f"ℹ️ [{ds_folder.name}] Using safe PLS upper bound: n_components <= {pls_max}")

        # Typical NIRS4ALL style: a single dataset path (not a list)
        dataset_config_single = DatasetConfigs(str(ds_folder), task_type="regression")

        pipeline_single = build_pipeline(
            group_key=use_group_key,
            n_splits=args.n_splits,
            random_state=args.random_state,
            n_trials=args.n_trials,
            pls_max_components=pls_max,
            only_model=args.only_model,
        )
        pipeline_name = "PLS_Ridge_Cartesian_Finetune_grouped" if use_group_key is not None else "PLS_Ridge_Cartesian_Finetune"
        pipeline_config_single = PipelineConfigs(pipeline_single, name=pipeline_name)

        try:
            _, preds_per_ds = runner.run(pipeline_config_single, dataset_config_single)
            all_predictions_per_dataset.update(preds_per_ds)
        except Exception as e:
            import traceback
            print(f"[X] Pipeline failed on dataset {ds_folder.name}: {e}")
            print(traceback.format_exc())
            continue

    print("\n" + "=" * 120)
    print("DONE. Summary per dataset (top by val metric):")
    print("=" * 120)

    final_dir = Path(args.workspace) / "final_refit"

    for dataset_name, dataset_prediction in all_predictions_per_dataset.items():
        run_preds = dataset_prediction["run_predictions"]
        best_list = run_preds.top(
            1,
            rank_metric=args.final_refit_rank_metric,
            rank_partition="val",
            display_metrics=[args.final_refit_rank_metric, "r2", "mape", "nrmse"],
        )
        if not best_list:
            print(f"- {dataset_name}: ⚠️ no completed runs (all trials failed)")
            continue

        best = best_list[0]

        try:
            val_rmse = best["metrics"]["val"]["rmse"]
            print(f"- {dataset_name}: {_best_name_safe(best)} | val_rmse={val_rmse:.6f} | preprocess={best.get('preprocessings')}")
        except Exception:
            print(f"- {dataset_name}: {_best_name_safe(best)} | preprocess={best.get('preprocessings')}")

        if args.do_final_refit:
            dataset_folder = _resolve_dataset_folder(datasets, dataset_name)
            if dataset_folder is None:
                print(f"⚠️ [{dataset_name}] Could not resolve dataset folder path → skipping final refit.")
                continue

            best_run = _extract_best_run(dataset_prediction, metric=args.final_refit_rank_metric)
            final_refit_and_predict(
                dataset_name=dataset_name,
                dataset_folder=dataset_folder,
                best_run=best_run,
                output_dir=final_dir,
                random_state=args.random_state,
            )


if __name__ == "__main__":
    main()
