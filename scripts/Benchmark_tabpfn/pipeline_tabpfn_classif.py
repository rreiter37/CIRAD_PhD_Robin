#!/usr/bin/env python3
"""
TabPFN raw classification pipeline (no scaling, no preprocessing).

- Loads datasets from:
  Data/classification/{dataset_type}/{dataset_name}/{Xtrain.csv,Ytrain.csv,Xtest.csv,Ytest.csv}
- Fits TabPFNClassifier only (raw).
- Saves predictions + per-dataset metrics.

Notes:
- Some datasets may contain non-standard CSV formatting (different delimiters, preamble lines, broken rows).
  The CSV reader below is intentionally robust.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

import torch
from tabpfn import TabPFNClassifier


# ===================== Device safety (avoid unsupported CUDA kernels) ===================== #

def get_safe_tabpfn_device() -> str:
    """
    Returns 'cuda' if the GPU architecture is supported by TabPFN CUDA kernels,
    otherwise falls back to 'cpu' to prevent CUDA kernel crashes.
    """
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available → Using CPU for TabPFN.")
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)

    # TabPFN kernels may not be compiled for very recent GPU arch (e.g., SM >= 90 / RTX 50xx).
    if major >= 9:
        print("⚠️ Detected GPU architecture potentially incompatible with TabPFN CUDA kernels (SM >= 90).")
        print("➡️ Switching TabPFN to CPU mode to avoid CUDA errors.")
        return "cpu"

    return "cuda"


TABPFN_DEVICE = get_safe_tabpfn_device()


# ===================== Data structures ===================== #

@dataclass(frozen=True)
class DatasetId:
    dataset_type: str
    dataset_name: str

    @property
    def tag(self) -> str:
        return f"{self.dataset_type}__{self.dataset_name}"


# ===================== Robust CSV reading ===================== #

_CANDIDATE_DELIMS: List[str] = [",", ";", "\t", "|"]


def _guess_delimiter_and_skiprows(path: Path, max_lines: int = 80) -> Tuple[str, int]:
    """
    Try to guess:
      - which delimiter is used
      - whether we need to skip a preamble (metadata lines) before the real table begins.

    Heuristic:
      - look at the first max_lines lines
      - for each candidate delimiter, compute the maximum number of "fields" (delim_count + 1)
      - choose the delimiter with the highest maximum
      - set skiprows to the first line index where that maximum is observed (i.e., table likely starts there)
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        raw = path.read_text(errors="replace").splitlines()

    raw = raw[:max_lines]
    if not raw:
        return ",", 0

    best_delim = ","
    best_max_fields = 1
    best_start_idx = 0

    for delim in _CANDIDATE_DELIMS:
        field_counts = []
        for line in raw:
            # Skip empty lines
            if not line.strip():
                field_counts.append(1)
                continue
            field_counts.append(line.count(delim) + 1)

        max_fields = max(field_counts) if field_counts else 1
        start_idx = field_counts.index(max_fields) if field_counts else 0

        if max_fields > best_max_fields:
            best_delim = delim
            best_max_fields = max_fields
            best_start_idx = start_idx

    # If we never found anything better than "1 field", keep comma and no skip.
    if best_max_fields <= 1:
        return ",", 0

    return best_delim, best_start_idx


def _read_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader for the project datasets.

    Strategy:
      1) Guess delimiter + skiprows (to bypass preamble lines).
      2) Try fast C-engine read.
      3) If it fails, fall back to python engine + on_bad_lines='skip'.
      4) Drop possible Unnamed index columns.
    """
    delim, skiprows = _guess_delimiter_and_skiprows(path)

    # First attempt: faster C engine
    try:
        df = pd.read_csv(path, sep=delim, skiprows=skiprows)
    except Exception:
        # Fallback: python engine is more tolerant; skip bad lines if any
        df = pd.read_csv(
            path,
            sep=delim,
            skiprows=skiprows,
            engine="python",
            on_bad_lines="skip",
        )

    # Common safety: drop unnamed index columns if present.
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

    # If we ended up with a single column but the file likely contains more,
    # try a last-resort delimiter inference with pandas' sep=None sniffing.
    if df.shape[1] == 1:
        try:
            df2 = pd.read_csv(path, sep=None, engine="python", skiprows=skiprows, on_bad_lines="skip")
            df2 = df2.loc[:, ~df2.columns.astype(str).str.match(r"^Unnamed")]
            if df2.shape[1] > 1:
                df = df2
        except Exception:
            pass

    return df

def _force_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Force all features to be numeric (float), because TabPFN's internal preprocessing may crash
    when encountering mixed dtype/object columns (strings + floats, etc.).

    This is not "preprocessing" in the ML sense (no scaling, no PCA): it is just sanitizing input types.
    Strategy:
      - align columns between train and test
      - coerce everything to numeric (errors='coerce')
      - replace inf/-inf by NaN
      - impute NaNs using train column means (fallback 0.0 if a column is all NaNs)
      - cast to float32 for stability/perf
    """
    # Align columns (intersection only, keep order from train)
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    if len(common_cols) == 0:
        raise ValueError("No common feature columns between X_train and X_test.")

    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    # Coerce to numeric
    for c in common_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    # Replace inf/-inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Impute using train means
    col_means = X_train.mean(axis=0, skipna=True)
    col_means = col_means.fillna(0.0)  # if a column is entirely NaN

    X_train = X_train.fillna(col_means)
    X_test = X_test.fillna(col_means)

    return X_train.astype(np.float32), X_test.astype(np.float32)


def load_classification_dataset(dataset_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load Xtrain/Ytrain/Xtest/Ytest from a single dataset directory.
    """
    xtr_path = dataset_dir / "Xtrain.csv"
    ytr_path = dataset_dir / "Ytrain.csv"
    xte_path = dataset_dir / "Xtest.csv"
    yte_path = dataset_dir / "Ytest.csv"

    missing = [p.name for p in [xtr_path, ytr_path, xte_path, yte_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {dataset_dir}: {missing}")

    X_train = _read_csv(xtr_path)
    y_train_df = _read_csv(ytr_path)
    X_test = _read_csv(xte_path)
    y_test_df = _read_csv(yte_path)

    # Expect single-column labels; if multiple columns exist, keep the first.
    y_train = y_train_df.iloc[:, 0].to_numpy()
    y_test = y_test_df.iloc[:, 0].to_numpy()
    
    # Ensure features are purely numeric to prevent TabPFN/sklearn encoder crashes on object/mixed dtypes
    X_train, X_test = _force_numeric_features(X_train, X_test)

    return X_train, y_train, X_test, y_test


def encode_labels(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Encode labels to integers (required by many classifiers, and safer for metrics).
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


# ===================== Metrics ===================== #

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_classes: int,
) -> Dict[str, float]:
    """
    Compute robust classification metrics.
    - accuracy
    - balanced_accuracy
    - f1_macro
    - roc_auc (binary) or roc_auc_ovr_macro (multiclass), if probabilities available
    """
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))

    if y_proba is not None and n_classes >= 2:
        try:
            if n_classes == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            # ROC AUC can fail if a class is missing in y_true
            pass

    return out


# ===================== Core runner ===================== #

def run_tabpfn_raw_classification(
    dataset_id: DatasetId,
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 42,
    n_estimators: int = 16,
    ignore_pretraining_limits: bool = True,
) -> Dict[str, object]:
    """
    Fit TabPFNClassifier on the dataset and write artifacts:
    - predictions CSV
    - metrics JSON
    - classification report TXT
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_classification_dataset(dataset_dir)
    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    n_classes = int(len(le.classes_))

    clf = TabPFNClassifier(
        n_estimators=n_estimators,
        device=TABPFN_DEVICE,
        random_state=seed,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )

    clf.fit(X_train, y_train_enc)
    y_pred = clf.predict(X_test)

    y_proba = None
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        y_proba = None

    metrics = compute_metrics(y_test_enc, y_pred, y_proba, n_classes=n_classes)

    # Decode predictions back to original labels for readability
    y_pred_labels = le.inverse_transform(np.asarray(y_pred, dtype=int))

    # Save predictions
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_labels})
    if y_proba is not None:
        for i, cls in enumerate(le.classes_):
            pred_df[f"proba__{cls}"] = y_proba[:, i]

    pred_path = output_dir / f"predictions__{dataset_id.tag}.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save metrics
    metrics_payload = {
        "dataset_type": dataset_id.dataset_type,
        "dataset_name": dataset_id.dataset_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": n_classes,
        "classes": [str(c) for c in le.classes_],
        "seed": seed,
        "n_estimators": n_estimators,
        "device": TABPFN_DEVICE,
        "metrics": metrics,
    }
    metrics_path = output_dir / f"metrics__{dataset_id.tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    # Save classification report
    try:
        report = classification_report(
            y_test_enc,
            y_pred,
            target_names=[str(c) for c in le.classes_],
            digits=4,
            zero_division=0,
        )
    except Exception:
        report = "classification_report failed."

    report_path = output_dir / f"report__{dataset_id.tag}.txt"
    report_path.write_text(report, encoding="utf-8")

    return metrics_payload
