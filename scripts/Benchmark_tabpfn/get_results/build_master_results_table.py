#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_master_results_table.py

Build a single master results table from heterogeneous result folders.

Design goals
------------
- One row per dataset x model x result directory "best final result".
- Flexible CLI: the number of models is not fixed and some results may be missing.
- Robust to partial/missing artifacts.
- Compatible with the saving conventions used by the provided pipelines:
  * TabPFN:     <dataset>__search_results.csv, <dataset>__best_config.json,
                <dataset>__final_predictions.csv, preds/<dataset>/predictions.parquet
  * PLS/Ridge:  <dataset>__search_results.parquet, <dataset>__best_config.json,
                <dataset>__final_predictions.csv, preds/<dataset>/predictions.parquet
  * CatBoost:   <dataset>__search_results.csv, <dataset>__best_config.json,
                <dataset>__final_predictions.csv, all_predictions.parquet

Main outputs
------------
- master_results.parquet
- master_results.csv
- master_results_errors.csv

Notes
-----
- By user request, there is no "variant" column.
- Task inference rule: if "classif" appears in the result directory path, task is
  set to "classification". Otherwise, task defaults to "regression".
- Regression metrics are computed when y_true is available in final_predictions.csv.
- RMSECV: mean validation RMSE across folds (taken from best_config/search results).
- RMSE-MF: mean RMSE on the external test set across fold-specific models, computed
  from the stored per-fold test predictions for the best preprocessing configuration.
- RMSEP: RMSE of the final refitted best model on the external test set.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CLI parsing helpers
# =============================================================================


def parse_kv_spec(spec: str) -> Dict[str, str]:
    """
    Parse a CLI specification of the form:
        dir=/path/to/results,model=TabPFN

    The parser is intentionally simple and assumes commas are separators between
    top-level key=value pairs.
    """
    out: Dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid --input specification fragment '{part}'. Expected key=value."
            )
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    if "dir" not in out:
        raise ValueError(f"Invalid --input specification '{spec}': missing 'dir='."
                         )
    if "model" not in out:
        raise ValueError(f"Invalid --input specification '{spec}': missing 'model='."
                         )
    return out


@dataclass(frozen=True)
class InputSource:
    """Description of one result source provided via CLI."""

    directory: Path
    model: str
    label: str


# =============================================================================
# Generic utility helpers
# =============================================================================


def infer_task_from_path(path: Path) -> str:
    """Infer task type from the directory path according to the user rule."""
    lower = str(path).lower()
    return "classification" if "classif" in lower else "regression"


def sanitize_jsonable(value: Any) -> Any:
    """Convert NumPy / Path / nested objects into JSON-serializable Python values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): sanitize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_jsonable(v) for v in value]
    return str(value)


def to_json_string(value: Any) -> Optional[str]:
    """Serialize a Python object to a stable JSON string."""
    if value is None:
        return None
    return json.dumps(sanitize_jsonable(value), ensure_ascii=False, sort_keys=True)


def read_table_auto(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet depending on suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=";", decimal=".")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def read_predictions_csv(path: Path) -> pd.DataFrame:
    """Read final predictions CSV using the project convention."""
    return pd.read_csv(path, sep=";", decimal=".")


def safe_float(value: Any) -> Optional[float]:
    """Convert a value to float when possible."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    if isinstance(value, (int, np.integer, np.floating)):
        val = float(value)
        return None if math.isnan(val) else val
    try:
        val = float(value)
    except Exception:
        return None
    return None if math.isnan(val) else val


def parse_list_like(value: Any) -> Optional[List[Any]]:
    """
    Parse a list stored either as a Python list, a JSON string, or a repr string.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
    return None


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute RMSE with population convention on paired vectors."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute MAE on paired vectors."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def r2_score_manual(y_true: Sequence[float], y_pred: Sequence[float]) -> Optional[float]:
    """Compute R² without sklearn and guard against constant targets."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return None
    return float(1.0 - ss_res / ss_tot)


# =============================================================================
# Artifact discovery
# =============================================================================


def collect_dataset_stems(result_dir: Path) -> List[str]:
    """
    Collect dataset stems from the result directory.

    A dataset stem is the prefix in:
      - <dataset>__best_config.json
      - <dataset>__search_results.csv/parquet
      - <dataset>__final_predictions.csv
    """
    stems = set()
    patterns = [
        "*__best_config.json",
        "*__search_results.csv",
        "*__search_results.parquet",
        "*__final_predictions.csv",
    ]
    for pattern in patterns:
        for path in result_dir.glob(pattern):
            if "__" in path.name:
                stems.add(path.name.split("__", 1)[0])
    return sorted(stems)


@dataclass
class DatasetArtifacts:
    """Resolved file paths for one dataset in one result directory."""

    dataset: str
    best_config_json: Optional[Path]
    search_results_path: Optional[Path]
    final_predictions_csv: Optional[Path]
    predictions_parquet: Optional[Path]
    all_predictions_parquet: Optional[Path]


def find_dataset_artifacts(result_dir: Path, dataset: str) -> DatasetArtifacts:
    """Resolve the paths of the known artifacts for one dataset."""
    best_config_json = result_dir / f"{dataset}__best_config.json"
    if not best_config_json.exists():
        best_config_json = None

    search_csv = result_dir / f"{dataset}__search_results.csv"
    search_parquet = result_dir / f"{dataset}__search_results.parquet"
    if search_csv.exists():
        search_results_path: Optional[Path] = search_csv
    elif search_parquet.exists():
        search_results_path = search_parquet
    else:
        search_results_path = None

    final_predictions_csv = result_dir / f"{dataset}__final_predictions.csv"
    if not final_predictions_csv.exists():
        final_predictions_csv = None

    predictions_parquet = result_dir / "preds" / dataset / "predictions.parquet"
    if not predictions_parquet.exists():
        predictions_parquet = None

    all_predictions_parquet = result_dir / "all_predictions.parquet"
    if not all_predictions_parquet.exists():
        all_predictions_parquet = None

    return DatasetArtifacts(
        dataset=dataset,
        best_config_json=best_config_json,
        search_results_path=search_results_path,
        final_predictions_csv=final_predictions_csv,
        predictions_parquet=predictions_parquet,
        all_predictions_parquet=all_predictions_parquet,
    )


# =============================================================================
# Configuration / preprocessing extraction
# =============================================================================


def build_preprocessing_pipeline_string(best_cfg: Any) -> Optional[str]:
    """
    Build a readable preprocessing pipeline string from the best_config payload.

    This function supports the different schema shapes used by the provided
    pipelines:
      - TabPFN / CatBoost: {shape, scatter, pca}
      - PLS / Ridge:       {shape, scatter, final_repr}
    """
    if not isinstance(best_cfg, dict):
        return None

    ordered_keys = ["shape", "scatter", "pca", "final_repr"]
    parts: List[str] = []
    for key in ordered_keys:
        value = best_cfg.get(key)
        if value is None:
            continue
        text = str(value)
        if text == "None":
            continue
        parts.append(text)

    return " + ".join(parts) if parts else "None"


# =============================================================================
# RMSE-MF extraction from fold-level predictions
# =============================================================================


def choose_best_preproc_tag(best_cfg: Any) -> Optional[str]:
    """
    Build the expected fold prediction tag from best_config.

    The fold parquet produced by TabPFN / PLS-Ridge stores a preproc_tag that is
    built from non-None preprocessing blocks joined with '+', or 'None' when no
    preprocessing is active. CatBoost stores a similar string under the 'config'
    column in all_predictions.parquet.
    """
    if not isinstance(best_cfg, dict):
        return None

    values: List[str] = []
    for key in ("shape", "scatter", "pca", "final_repr"):
        value = best_cfg.get(key)
        if value is None:
            continue
        text = str(value)
        if text != "None":
            values.append(text)

    return "+".join(values) if values else "None"


def compute_rmse_mf_from_predictions(
    artifacts: DatasetArtifacts,
    expected_tag: Optional[str],
    model_name: str,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute RMSE-MF as the mean test RMSE across fold-specific models.

    Returns
    -------
    (rmse_mf, source)
        source is a small string explaining which artifact was used.
    """
    if expected_tag is None:
        return None, None

    # Case 1: dataset-specific predictions.parquet (TabPFN, PLS, Ridge)
    if artifacts.predictions_parquet is not None:
        try:
            df = pd.read_parquet(artifacts.predictions_parquet)
        except Exception:
            return None, None

        if "split" not in df.columns or "y_true" not in df.columns or "y_pred" not in df.columns:
            return None, None

        df = df[df["split"].astype(str).str.lower() == "test"].copy()
        if df.empty:
            return None, None

        if "preproc_tag" in df.columns:
            df = df[df["preproc_tag"].astype(str) == expected_tag].copy()

        if "model" in df.columns:
            df = df[df["model"].astype(str).str.lower() == model_name.lower()].copy()

        if df.empty or "fold" not in df.columns:
            return None, None

        fold_rmses: List[float] = []
        for _, fold_df in df.groupby("fold"):
            fold_df = fold_df.dropna(subset=["y_true", "y_pred"])
            if fold_df.empty:
                continue
            fold_rmses.append(rmse(fold_df["y_true"], fold_df["y_pred"]))

        if not fold_rmses:
            return None, None
        return float(np.mean(fold_rmses)), "preds/<dataset>/predictions.parquet"

    # Case 2: global all_predictions.parquet (CatBoost)
    if artifacts.all_predictions_parquet is not None:
        try:
            df = pd.read_parquet(artifacts.all_predictions_parquet)
        except Exception:
            return None, None

        required = {"dataset", "split", "fold", "y_true", "y_pred"}
        if not required.issubset(df.columns):
            return None, None

        df = df[df["dataset"].astype(str) == artifacts.dataset].copy()
        df = df[df["split"].astype(str).str.lower() == "test"].copy()
        if df.empty:
            return None, None

        # CatBoost uses the column name "config" instead of "preproc_tag".
        if "config" in df.columns:
            df = df[df["config"].astype(str) == expected_tag].copy()

        if df.empty:
            return None, None

        fold_rmses = []
        for _, fold_df in df.groupby("fold"):
            fold_df = fold_df.dropna(subset=["y_true", "y_pred"])
            if fold_df.empty:
                continue
            fold_rmses.append(rmse(fold_df["y_true"], fold_df["y_pred"]))

        if not fold_rmses:
            return None, None
        return float(np.mean(fold_rmses)), "all_predictions.parquet"

    return None, None


# =============================================================================
# Per-dataset row assembly
# =============================================================================


def build_row_for_dataset(source: InputSource, artifacts: DatasetArtifacts) -> Dict[str, Any]:
    """Build one master-results row for one dataset and one model source."""
    result_dir = source.directory
    dataset = artifacts.dataset
    task = infer_task_from_path(result_dir)

    row: Dict[str, Any] = {
        "dataset": dataset,
        "task": task,
        "model": source.model,
        "result_dir": str(result_dir),
        "result_label": source.label,
        "status": "ok",
        # Preprocessing / config info
        "preprocessing_pipeline": None,
        "best_config_json": None,
        "best_model_params_json": None,
        # CV / test metrics
        "RMSECV": None,
        "RMSE_MF": None,
        "RMSEP": None,
        "MAE_test": None,
        "r2_test": None,
        # Extra traceability
        "best_fold_scores_json": None,
        "trial_values_json": None,
        "search_mean_score": None,
        "search_results_path": str(artifacts.search_results_path) if artifacts.search_results_path else None,
        "best_config_path": str(artifacts.best_config_json) if artifacts.best_config_json else None,
        "final_predictions_path": str(artifacts.final_predictions_csv) if artifacts.final_predictions_csv else None,
        "fold_predictions_path": (
            str(artifacts.predictions_parquet)
            if artifacts.predictions_parquet is not None
            else (str(artifacts.all_predictions_parquet) if artifacts.all_predictions_parquet is not None else None)
        ),
        "rmse_mf_source": None,
        "seed": None,
        "n_splits": None,
    }

    status_flags: List[str] = []

    # -------------------------------------------------------------------------
    # Read best_config.json when available
    # -------------------------------------------------------------------------
    best_payload: Dict[str, Any] = {}
    if artifacts.best_config_json is not None:
        try:
            with open(artifacts.best_config_json, "r", encoding="utf-8") as f:
                best_payload = json.load(f)
        except Exception:
            status_flags.append("invalid_best_config")
            best_payload = {}
    else:
        status_flags.append("missing_best_config")

    best_cfg = best_payload.get("best_config")
    row["preprocessing_pipeline"] = build_preprocessing_pipeline_string(best_cfg)
    row["best_config_json"] = to_json_string(best_cfg)
    row["best_model_params_json"] = to_json_string(best_payload.get("best_model_params"))
    row["best_fold_scores_json"] = to_json_string(best_payload.get("best_fold_scores"))
    row["trial_values_json"] = to_json_string(best_payload.get("trial_values"))
    row["seed"] = safe_float(best_payload.get("seed"))
    row["n_splits"] = safe_float(best_payload.get("n_splits"))

    # User-requested convention: RMSECV is the mean RMSE across validation folds.
    rmsecv = safe_float(best_payload.get("best_mean_score"))

    # -------------------------------------------------------------------------
    # Fallback to search_results when best_config.json is missing or incomplete
    # -------------------------------------------------------------------------
    if artifacts.search_results_path is not None:
        try:
            search_df = read_table_auto(artifacts.search_results_path)
            if not search_df.empty:
                search_df = search_df.sort_values("mean_score", ascending=True).reset_index(drop=True)
                best_row = search_df.iloc[0].to_dict()

                if rmsecv is None:
                    rmsecv = safe_float(best_row.get("mean_score"))

                if row["best_fold_scores_json"] is None:
                    row["best_fold_scores_json"] = to_json_string(parse_list_like(best_row.get("fold_scores")))
                if row["trial_values_json"] is None:
                    row["trial_values_json"] = to_json_string(parse_list_like(best_row.get("trial_values")))
                if row["best_model_params_json"] is None:
                    # In PLS/Ridge search results, best_params is present directly in the table.
                    best_params = best_row.get("best_params")
                    if isinstance(best_params, str):
                        try:
                            best_params = json.loads(best_params)
                        except Exception:
                            pass
                    row["best_model_params_json"] = to_json_string(best_params)

                row["search_mean_score"] = safe_float(best_row.get("mean_score"))

                if row["preprocessing_pipeline"] is None:
                    cfg_candidate = {
                        key: best_row.get(key)
                        for key in ("shape", "scatter", "pca", "final_repr")
                        if key in best_row
                    }
                    row["preprocessing_pipeline"] = build_preprocessing_pipeline_string(cfg_candidate)
                    row["best_config_json"] = to_json_string(cfg_candidate)
                    if best_cfg is None:
                        best_cfg = cfg_candidate
        except Exception:
            status_flags.append("invalid_search_results")
    else:
        status_flags.append("missing_search_results")

    row["RMSECV"] = rmsecv

    # -------------------------------------------------------------------------
    # Compute RMSE-MF from stored per-fold test predictions
    # -------------------------------------------------------------------------
    expected_tag = choose_best_preproc_tag(best_cfg)
    rmse_mf, rmse_mf_source = compute_rmse_mf_from_predictions(
        artifacts=artifacts,
        expected_tag=expected_tag,
        model_name=source.model,
    )
    row["RMSE_MF"] = rmse_mf
    row["rmse_mf_source"] = rmse_mf_source
    if rmse_mf is None:
        status_flags.append("missing_fold_test_predictions")

    # -------------------------------------------------------------------------
    # Compute final-test metrics from <dataset>__final_predictions.csv
    # -------------------------------------------------------------------------
    if artifacts.final_predictions_csv is not None:
        try:
            pred_df = read_predictions_csv(artifacts.final_predictions_csv)
            required_cols = {"y_true", "y_pred"}
            if required_cols.issubset(pred_df.columns):
                pred_df = pred_df.dropna(subset=["y_true", "y_pred"])
                if not pred_df.empty:
                    row["RMSEP"] = rmse(pred_df["y_true"], pred_df["y_pred"])
                    row["MAE_test"] = mae(pred_df["y_true"], pred_df["y_pred"])
                    row["r2_test"] = r2_score_manual(pred_df["y_true"], pred_df["y_pred"])
                else:
                    status_flags.append("empty_final_predictions")
            else:
                status_flags.append("missing_ytrue_in_final_predictions")
        except Exception:
            status_flags.append("invalid_final_predictions")
    else:
        status_flags.append("missing_final_predictions")

    # -------------------------------------------------------------------------
    # Final status consolidation
    # -------------------------------------------------------------------------
    hard_missing = {
        "missing_best_config",
        "missing_search_results",
        "missing_final_predictions",
        "invalid_best_config",
        "invalid_search_results",
        "invalid_final_predictions",
    }

    if not status_flags:
        row["status"] = "ok"
    elif any(flag in hard_missing for flag in status_flags):
        row["status"] = "partial"
    else:
        row["status"] = "partial"

    row["status_details"] = ";".join(status_flags) if status_flags else None
    return row


# =============================================================================
# Main entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified master results table from heterogeneous model result folders."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Input result source. Format: dir=/path/to/results,model=TabPFN[,label=my_alias]. "
            "Repeat --input for each model family / result directory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where master_results.parquet/csv and master_results_errors.csv will be written.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Unused for artifact collection itself, but kept for future flexibility. "
            "Currently, the script expects artifacts directly inside each provided result directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_sources: List[InputSource] = []
    for idx, raw_spec in enumerate(args.input):
        parsed = parse_kv_spec(raw_spec)
        directory = Path(parsed["dir"]).expanduser().resolve()
        if not directory.exists():
            raise FileNotFoundError(f"Input directory does not exist: {directory}")
        model = parsed["model"].strip()
        label = parsed.get("label", f"input_{idx}_{model}").strip()
        input_sources.append(InputSource(directory=directory, model=model, label=label))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for source in input_sources:
        dataset_stems = collect_dataset_stems(source.directory)
        if not dataset_stems:
            # Keep one explicit error-like line so the absence is visible.
            rows.append(
                {
                    "dataset": None,
                    "task": infer_task_from_path(source.directory),
                    "model": source.model,
                    "result_dir": str(source.directory),
                    "result_label": source.label,
                    "status": "partial",
                    "status_details": "no_dataset_artifacts_found",
                }
            )
            continue

        for dataset in dataset_stems:
            artifacts = find_dataset_artifacts(source.directory, dataset)
            row = build_row_for_dataset(source, artifacts)
            rows.append(row)

    master_df = pd.DataFrame(rows)

    # Stable column ordering for downstream scripts.
    preferred_columns = [
        "dataset",
        "task",
        "model",
        "result_label",
        "result_dir",
        "status",
        "status_details",
        "preprocessing_pipeline",
        "RMSECV",
        "RMSE_MF",
        "RMSEP",
        "MAE_test",
        "r2_test",
        "search_mean_score",
        "seed",
        "n_splits",
        "best_config_json",
        "best_model_params_json",
        "best_fold_scores_json",
        "trial_values_json",
        "search_results_path",
        "best_config_path",
        "final_predictions_path",
        "fold_predictions_path",
        "rmse_mf_source",
    ]
    other_columns = [col for col in master_df.columns if col not in preferred_columns]
    master_df = master_df[[col for col in preferred_columns if col in master_df.columns] + other_columns]

    parquet_path = output_dir / "master_results.parquet"
    csv_path = output_dir / "master_results.csv"
    errors_path = output_dir / "master_results_errors.csv"

    master_df.to_parquet(parquet_path, index=False)
    master_df.to_csv(csv_path, index=False)

    errors_df = master_df[master_df["status"].astype(str) != "ok"].copy()
    errors_df.to_csv(errors_path, index=False)

    print(f"Saved master parquet: {parquet_path}")
    print(f"Saved master csv:     {csv_path}")
    print(f"Saved errors csv:     {errors_path}")
    print(f"Rows: {len(master_df)} | Errors/partials: {len(errors_df)}")


if __name__ == "__main__":
    main()
