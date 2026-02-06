#!/usr/bin/env python3
"""
TabPFN classification pipeline with an optional PCA search on a validation split.

- Loads datasets from:
  Data/classification/{dataset_type}/{dataset_name}/{Xtrain.csv,Ytrain.csv,Xtest.csv,Ytest.csv}
- Fits TabPFNClassifier in CPU mode.
- Optionally performs a small PCA search on the TRAIN split using validation
  (Stratified CV) to pick the best PCA option, then retrains on the full
  train set and evaluates on the test set.
- Saves predictions + per-dataset metrics.

Notes:
- Some datasets may contain non-standard CSV formatting (different delimiters, preamble lines, broken rows).
  The CSV reader below is intentionally robust.

PCA search space (as requested):
  - no_pca
  - pca_adapt_0.25n  (n_components = max(5, int(0.25 * n_train)))
  - pca_adapt_0.10n  (n_components = max(5, int(0.10 * n_train)))
  - pca_0.99         (n_components keeps 99% explained variance)
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
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

import torch
from tabpfn import TabPFNClassifier


# ===================== Device policy ===================== #

# IMPORTANT:
# The user explicitly requested to keep TabPFN on CPU for now.
# This avoids CUDA/SM120 issues on RTX 50xx until the environment is upgraded.
TABPFN_DEVICE = "cpu"



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


# ===================== PCA helpers (aligned with MOD7 search) ===================== #

class PCAAdaptive:
    """
    PCA where n_components is derived from the number of training samples.

    This mimics the "PCA_0.25n" / "PCA_0.10n" logic used in the MOD7 search script.
    """

    def __init__(
        self,
        fraction: float,
        whiten: bool = True,
        min_components: int = 5,
        random_state: int = 42,
    ):
        self.fraction = float(fraction)
        self.whiten = bool(whiten)
        self.min_components = int(min_components)
        self.random_state = int(random_state)
        self._pca: Optional[PCA] = None

    def fit(self, X: np.ndarray) -> "PCAAdaptive":
        X = np.asarray(X)
        n_samples, n_features = X.shape

        n_comp = int(self.fraction * n_samples)
        n_comp = max(self.min_components, n_comp)
        n_comp = min(n_comp, n_features)

        self._pca = PCA(
            n_components=n_comp,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        self._pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAAdaptive.transform() called before fit().")
        return self._pca.transform(np.asarray(X))

    @property
    def n_components_(self) -> Optional[int]:
        if self._pca is None:
            return None
        return int(self._pca.n_components_)


def _make_pca_candidates(seed: int) -> List[Tuple[str, Optional[object]]]:
    """Return the requested PCA candidates."""
    return [
        ("no_pca", None),
        ("pca_adapt_0.25n", PCAAdaptive(fraction=0.25, whiten=True, random_state=seed)),
        ("pca_adapt_0.10n", PCAAdaptive(fraction=0.10, whiten=True, random_state=seed)),
        ("pca_0.99", PCA(n_components=0.99, whiten=True, random_state=seed)),
    ]


def _apply_pca_fit_transform(
    pca_obj: Optional[object],
    X_fit: np.ndarray,
    X_transform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """
    Fit PCA on X_fit and transform both X_fit and X_transform.

    Returns:
        X_fit_pca, X_transform_pca, n_components_used
    """
    if pca_obj is None:
        return X_fit, X_transform, None

    # Defensive clone: some estimators store state in-place.
    # For sklearn PCA, creating a new instance is simplest.
    if isinstance(pca_obj, PCA):
        pca = PCA(
            n_components=pca_obj.n_components,
            whiten=bool(getattr(pca_obj, "whiten", True)),
            random_state=getattr(pca_obj, "random_state", None),
        )
        pca.fit(X_fit)
        return pca.transform(X_fit), pca.transform(X_transform), int(pca.n_components_)

    if isinstance(pca_obj, PCAAdaptive):
        pca = PCAAdaptive(
            fraction=pca_obj.fraction,
            whiten=pca_obj.whiten,
            min_components=pca_obj.min_components,
            random_state=pca_obj.random_state,
        )
        pca.fit(X_fit)
        return pca.transform(X_fit), pca.transform(X_transform), pca.n_components_

    raise TypeError(f"Unsupported PCA object type: {type(pca_obj)}")


def select_best_pca_via_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_estimators: int,
    ignore_pretraining_limits: bool,
    n_splits: int = 3,
) -> Dict[str, object]:
    """
    Select the best PCA option using StratifiedKFold validation on the train set.

    Selection metric:
      - primary: mean accuracy
      - tie-break: mean balanced accuracy

    Returns a dict containing:
      - best_pca_name
      - best_mean_accuracy
      - best_mean_balanced_accuracy
      - best_n_components (if applicable)
      - all_candidates (list of per-candidate scores)
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    candidates = _make_pca_candidates(seed)

    all_rows: List[Dict[str, object]] = []

    for pca_name, pca_obj in candidates:
        fold_acc: List[float] = []
        fold_bacc: List[float] = []
        fold_ncomp: List[int] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr = X_train[tr_idx]
            y_tr = y_train[tr_idx]
            X_va = X_train[va_idx]
            y_va = y_train[va_idx]

            X_tr_p, X_va_p, n_comp = _apply_pca_fit_transform(pca_obj, X_tr, X_va)
            if n_comp is not None:
                fold_ncomp.append(int(n_comp))

            clf = TabPFNClassifier(
                n_estimators=int(n_estimators),
                device=TABPFN_DEVICE,  # CPU by policy
                random_state=int(seed) + int(fold_idx),
                ignore_pretraining_limits=bool(ignore_pretraining_limits),
            )
            clf.fit(X_tr_p, y_tr)
            y_hat = clf.predict(X_va_p)

            fold_acc.append(float(accuracy_score(y_va, y_hat)))
            fold_bacc.append(float(balanced_accuracy_score(y_va, y_hat)))

        row = {
            "pca": pca_name,
            "mean_accuracy": float(np.mean(fold_acc)) if fold_acc else float("nan"),
            "mean_balanced_accuracy": float(np.mean(fold_bacc)) if fold_bacc else float("nan"),
            "n_components": int(np.median(fold_ncomp)) if fold_ncomp else None,
        }
        all_rows.append(row)

    # Deterministic selection: sort by (-acc, -bacc, pca_name)
    all_rows_sorted = sorted(
        all_rows,
        key=lambda r: (
            -float(r["mean_accuracy"]),
            -float(r["mean_balanced_accuracy"]),
            str(r["pca"]),
        ),
    )

    best = all_rows_sorted[0]
    return {
        "best_pca_name": best["pca"],
        "best_mean_accuracy": best["mean_accuracy"],
        "best_mean_balanced_accuracy": best["mean_balanced_accuracy"],
        "best_n_components": best.get("n_components", None),
        "all_candidates": all_rows_sorted,
    }


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


def run_tabpfn_classification_with_pca_search(
    dataset_id: DatasetId,
    dataset_dir: Path,
    output_dir: Path,
    seed: int = 42,
    n_estimators: int = 16,
    ignore_pretraining_limits: bool = True,
    pca_search_n_splits: int = 3,
) -> Dict[str, object]:
    """
    Run TabPFN classification with a small PCA search on the TRAIN split.

    Workflow:
      1) Load (Xtrain, Ytrain, Xtest, Ytest)
      2) Encode labels to ints
      3) Select the best PCA option using StratifiedKFold validation on Xtrain
      4) Refit on the full Xtrain with the selected PCA
      5) Evaluate on Xtest

    Artifacts:
      - predictions__<dataset>.csv           (same name as raw mode for downstream compatibility)
      - metrics__<dataset>.json              (adds pca_search and best_pca details)
      - report__<dataset>.txt
      - pca_search__<dataset>.csv            (per-candidate validation scores)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_classification_dataset(dataset_dir)
    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)
    n_classes = int(len(le.classes_))

    # --- PCA selection on train only (validation) ---
    pca_search = select_best_pca_via_validation(
        X_train=X_train,
        y_train=y_train_enc,
        seed=int(seed),
        n_estimators=int(n_estimators),
        ignore_pretraining_limits=bool(ignore_pretraining_limits),
        n_splits=int(pca_search_n_splits),
    )

    best_pca_name = str(pca_search["best_pca_name"])
    pca_candidates = {name: obj for name, obj in _make_pca_candidates(seed)}
    best_pca_obj = pca_candidates.get(best_pca_name, None)

    # Fit PCA on full train and transform train/test
    X_train_p, X_test_p, n_comp_used = _apply_pca_fit_transform(best_pca_obj, X_train, X_test)

    clf = TabPFNClassifier(
        n_estimators=int(n_estimators),
        device=TABPFN_DEVICE,  # CPU by policy
        random_state=int(seed),
        ignore_pretraining_limits=bool(ignore_pretraining_limits),
    )
    clf.fit(X_train_p, y_train_enc)
    y_pred = clf.predict(X_test_p)

    y_proba = None
    try:
        y_proba = clf.predict_proba(X_test_p)
    except Exception:
        y_proba = None

    metrics = compute_metrics(y_test_enc, y_pred, y_proba, n_classes=n_classes)

    # Decode predictions back to original labels for readability
    y_pred_labels = le.inverse_transform(np.asarray(y_pred, dtype=int))

    # Save predictions (keep same filename pattern as raw mode)
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred_labels})
    if y_proba is not None:
        for i, cls in enumerate(le.classes_):
            pred_df[f"proba__{cls}"] = y_proba[:, i]

    pred_path = output_dir / f"predictions__{dataset_id.tag}.csv"
    pred_df.to_csv(pred_path, index=False)

    # Persist PCA search table
    pca_search_path = output_dir / f"pca_search__{dataset_id.tag}.csv"
    pd.DataFrame(pca_search["all_candidates"]).to_csv(pca_search_path, index=False)

    # Save metrics
    metrics_payload: Dict[str, object] = {
        "dataset_type": dataset_id.dataset_type,
        "dataset_name": dataset_id.dataset_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": n_classes,
        "classes": [str(c) for c in le.classes_],
        "seed": int(seed),
        "n_estimators": int(n_estimators),
        "device": TABPFN_DEVICE,
        "metrics": metrics,
        "pca_search": {
            "n_splits": int(pca_search_n_splits),
            "candidates": [c[0] for c in _make_pca_candidates(seed)],
            "best_pca": best_pca_name,
            "best_n_components": int(n_comp_used) if n_comp_used is not None else None,
            "best_mean_accuracy": float(pca_search["best_mean_accuracy"]),
            "best_mean_balanced_accuracy": float(pca_search["best_mean_balanced_accuracy"]),
        },
    }

    metrics_path = output_dir / f"metrics__{dataset_id.tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    # Save classification report
    try:
        report = classification_report(
            y_test_enc,
            np.asarray(y_pred, dtype=int),
            target_names=[str(c) for c in le.classes_],
            digits=4,
            zero_division=0,
        )
    except Exception:
        report = "classification_report failed."

    report_path = output_dir / f"report__{dataset_id.tag}.txt"
    header = (
        f"PCA search: best={best_pca_name} | n_components={n_comp_used}\n"
        f"Validation mean accuracy={pca_search['best_mean_accuracy']:.6f} | "
        f"mean balanced accuracy={pca_search['best_mean_balanced_accuracy']:.6f}\n\n"
    )
    report_path.write_text(header + report, encoding="utf-8")

    return metrics_payload
