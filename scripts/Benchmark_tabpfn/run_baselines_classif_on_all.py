#!/usr/bin/env python3
"""
Baseline classification pipeline (Q20-style "pipeline" syntax) for:
- CatBoost
- PLS-DA
- CNN (PyTorch)
- MLPClassifier (sklearn)

Datasets are loaded from:
  Data/classification/{dataset_type}/{dataset_name}/
    Xtrain.csv, Ytrain.csv, Xtest.csv, Ytest.csv

Outputs (comparable to TabPFN runner):
- predictions__{dataset_type}__{dataset_name}__{model}.csv
- metrics__{dataset_type}__{dataset_name}__{model}.json
- report__{dataset_type}__{dataset_name}__{model}.txt
- summary_metrics.csv

Notes:
- Default behavior is "raw" (no preprocessing), to keep comparability with TabPFN raw pipeline.
- The pipeline syntax mirrors Q20_analysis.py: a list of {"model": ..., "name": ...}.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.neural_network import MLPClassifier

# Optional dependencies
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    # Prefer nirs4all PLSDA if available (like Q20 uses)
    from nirs4all.operators.models.sklearn import PLSDA as Nirs4AllPLSDA
except Exception:
    Nirs4AllPLSDA = None

try:
    # Fallback PLS approach
    from sklearn.cross_decomposition import PLSRegression
except Exception:
    PLSRegression = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


# ===================== Data structures ===================== #

@dataclass(frozen=True)
class DatasetId:
    dataset_type: str
    dataset_name: str

    @property
    def tag(self) -> str:
        return f"{self.dataset_type}__{self.dataset_name}"


# ===================== Robust CSV reading (aligned with TabPFN pipeline) ===================== #

_CANDIDATE_DELIMS: List[str] = [",", ";", "\t", "|"]


def _guess_delimiter_and_skiprows(path: Path, max_lines: int = 80) -> Tuple[str, int]:
    """
    Guess delimiter and possible preamble skiprows.
    This mirrors the robust logic used in pipeline_tabpfn_classif.py.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        raw = path.read_text(errors="replace").splitlines()

    raw = raw[:max_lines]
    if not raw:
        return ",", 0

    best_delim, best_max_fields, best_start_idx = ",", 1, 0

    for delim in _CANDIDATE_DELIMS:
        field_counts = []
        for line in raw:
            if not line.strip():
                field_counts.append(1)
                continue
            field_counts.append(line.count(delim) + 1)

        max_fields = max(field_counts) if field_counts else 1
        start_idx = field_counts.index(max_fields) if field_counts else 0

        if max_fields > best_max_fields:
            best_delim, best_max_fields, best_start_idx = delim, max_fields, start_idx

    return best_delim, best_start_idx


def _read_csv(path: Path) -> pd.DataFrame:
    """Robust CSV reader with delimiter + skiprows inference."""
    delim, skiprows = _guess_delimiter_and_skiprows(path)

    # Primary attempt
    try:
        df = pd.read_csv(path, sep=delim, skiprows=skiprows, engine="python")
        if df.shape[1] <= 1:
            df2 = pd.read_csv(path, sep=",", skiprows=skiprows, engine="python")
            if df2.shape[1] > df.shape[1]:
                df = df2
        return df
    except Exception:
        pass

    # Fallback attempts
    for d in _CANDIDATE_DELIMS:
        try:
            df = pd.read_csv(path, sep=d, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

    return pd.read_csv(path, engine="python")


def _force_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure both train/test have numeric features only.
    - Keep common columns
    - Coerce to numeric, replace inf, impute using train means
    """
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    for c in common_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    col_means = X_train.mean(axis=0, skipna=True).fillna(0.0)
    X_train = X_train.fillna(col_means)
    X_test = X_test.fillna(col_means)

    return X_train.to_numpy(dtype=np.float32), X_test.to_numpy(dtype=np.float32)


def load_classification_dataset(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Xtrain/Ytrain/Xtest/Ytest from a dataset folder."""
    xtr = dataset_dir / "Xtrain.csv"
    ytr = dataset_dir / "Ytrain.csv"
    xte = dataset_dir / "Xtest.csv"
    yte = dataset_dir / "Ytest.csv"

    missing = [p.name for p in [xtr, ytr, xte, yte] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {dataset_dir}: {missing}")

    X_train_df = _read_csv(xtr)
    y_train_df = _read_csv(ytr)
    X_test_df = _read_csv(xte)
    y_test_df = _read_csv(yte)

    y_train = y_train_df.iloc[:, 0].to_numpy()
    y_test = y_test_df.iloc[:, 0].to_numpy()

    X_train, X_test = _force_numeric_features(X_train_df, X_test_df)
    return X_train, y_train, X_test, y_test


def encode_labels(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Encode labels to integer classes."""
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
    """Compute accuracy / balanced_accuracy / f1_macro / roc_auc when possible."""
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
            pass

    return out


# ===================== Models ===================== #

def fit_predict_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    seed: int,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Train CatBoost and return y_pred + y_proba."""
    if CatBoostClassifier is None:
        raise RuntimeError("CatBoost is not installed. Please `pip install catboost`.")

    model = CatBoostClassifier(
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,  # important to avoid disk artifacts (like Q20)
        **params,
    )
    model.fit(X_train, y_train)

    y_pred = np.asarray(model.predict(X_test)).reshape(-1).astype(int)

    y_proba = None
    try:
        y_proba = np.asarray(model.predict_proba(X_test), dtype=np.float64)
    except Exception:
        y_proba = None

    meta = {"model": "CatBoostClassifier", "params": dict(params)}
    return y_pred, y_proba, meta


def fit_predict_plsda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    n_components: int,
    standardize: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    PLS-DA, nirs4all-first (like Q20), fallback to sklearn PLSRegression on one-hot labels.
    """
    if Nirs4AllPLSDA is not None:
        steps = []
        if standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("plsda", Nirs4AllPLSDA(n_components=n_components)))
        clf = SkPipeline(steps)

        clf.fit(X_train, y_train)
        y_pred = np.asarray(clf.predict(X_test)).reshape(-1).astype(int)

        y_proba = None
        if hasattr(clf, "predict_proba"):
            try:
                y_proba = clf.predict_proba(X_test)
            except Exception:
                y_proba = None

        meta = {"model": "PLSDA(nirs4all)", "params": {"n_components": n_components, "standardize": standardize}}
        return y_pred, y_proba, meta

    # Fallback
    if PLSRegression is None:
        raise RuntimeError("Neither nirs4all PLSDA nor sklearn PLSRegression is available.")

    Y = np.zeros((len(y_train), n_classes), dtype=np.float32)
    Y[np.arange(len(y_train)), y_train.astype(int)] = 1.0

    if standardize:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
    else:
        Xtr = X_train
        Xte = X_test

    pls = PLSRegression(n_components=n_components)
    pls.fit(Xtr, Y)
    scores = np.asarray(pls.predict(Xte), dtype=np.float64)

    # Softmax for pseudo-probabilities
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    y_proba = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-12)

    y_pred = np.argmax(y_proba, axis=1).astype(int)

    meta = {"model": "PLS-DA(fallback)", "params": {"n_components": n_components, "standardize": standardize}}
    return y_pred, y_proba, meta


def fit_predict_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    hidden_layers: Tuple[int, ...],
    max_iter: int,
    standardize: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    MLPClassifier baseline (like Q20 includes multiple MLP configs).
    """
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=seed,
        ))
    )
    clf = SkPipeline(steps)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test).astype(int)

    y_proba = None
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_test)
        except Exception:
            y_proba = None

    meta = {
        "model": "MLPClassifier(sklearn)",
        "params": {
            "hidden_layer_sizes": list(hidden_layers),
            "max_iter": int(max_iter),
            "standardize": bool(standardize),
        },
    }
    return y_pred, y_proba, meta


class SimpleCNN1D(nn.Module):
    """A minimal 1D CNN for spectral vectors (features as sequence)."""
    def __init__(self, n_features: int, n_classes: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, n_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.head(self.net(x))


def _set_torch_determinism(seed: int) -> None:
    """Best-effort CPU determinism for PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.set_num_threads(1)


def fit_predict_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_channels: int,
    dropout: float,
    standardize: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Train a small CNN on CPU and return y_pred + y_proba."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Please install torch to use CNN baseline.")

    _set_torch_determinism(seed)

    if standardize:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
    else:
        Xtr = X_train
        Xte = X_test

    device = torch.device("cpu")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1)  # (N, 1, F)
    ytr_t = torch.tensor(y_train.astype(int), dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).unsqueeze(1)

    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = SimpleCNN1D(
        n_features=Xtr.shape[1],
        n_classes=n_classes,
        hidden_channels=hidden_channels,
        dropout=dropout,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t.to(device)).cpu().numpy().astype(np.float64)

    # Softmax probabilities
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    y_proba = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-12)

    y_pred = np.argmax(y_proba, axis=1).astype(int)

    meta = {
        "model": "CNN1D(torch)",
        "params": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "hidden_channels": int(hidden_channels),
            "dropout": float(dropout),
            "standardize": bool(standardize),
            "device": str(device),
        },
    }
    return y_pred, y_proba, meta


# ===================== Outputs ===================== #

def save_outputs(
    output_dir: Path,
    dataset_id: DatasetId,
    model_name: str,
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    y_true_enc: np.ndarray,
    y_pred_enc: np.ndarray,
    le: LabelEncoder,
    y_proba: Optional[np.ndarray],
    metrics_payload: Dict[str, Any],
) -> None:
    """Save predictions CSV + metrics JSON + report TXT."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({"y_true": y_true_labels, "y_pred": y_pred_labels})
    if y_proba is not None:
        for i, cls in enumerate(le.classes_):
            pred_df[f"proba__{cls}"] = y_proba[:, i]

    pred_path = output_dir / f"predictions__{dataset_id.tag}__{model_name}.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = output_dir / f"metrics__{dataset_id.tag}__{model_name}.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        report = classification_report(
            y_true_enc,
            y_pred_enc,
            target_names=[str(c) for c in le.classes_],
            digits=4,
            zero_division=0,
        )
    except Exception:
        report = "classification_report failed."

    report_path = output_dir / f"report__{dataset_id.tag}__{model_name}.txt"
    report_path.write_text(report, encoding="utf-8")


# ===================== Dataset discovery (same philosophy as TabPFN runner) ===================== #

def discover_datasets(data_root: Path) -> List[Dict[str, Path]]:
    """Discover datasets with required files."""
    required = {"Xtrain.csv", "Ytrain.csv", "Xtest.csv", "Ytest.csv"}
    out: List[Dict[str, Path]] = []

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    for dataset_type_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        for dataset_dir in sorted([p for p in dataset_type_dir.iterdir() if p.is_dir()]):
            files = {p.name for p in dataset_dir.iterdir() if p.is_file()}
            if required.issubset(files):
                out.append(
                    {
                        "dataset_type": dataset_type_dir.name,
                        "dataset_name": dataset_dir.name,
                        "path": dataset_dir,
                    }
                )
    return out


# ===================== Q20-style pipeline builder ===================== #

def build_pipeline_from_args(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Q20-style pipeline: list of {"model": <callable or spec>, "name": str, "meta": dict}.
    We keep it simple: each step is a model spec used by the local runner.
    """
    pipeline: List[Dict[str, Any]] = []

    # PLS-DA steps (Q20 uses multiple component settings)
    if args.run_plsda:
        for nc in args.plsda_components_list:
            pipeline.append({
                "name": f"plsda_nc{nc}",
                "model": "plsda",
                "params": {"n_components": int(nc), "standardize": bool(args.standardize_plsda)},
            })

    # CatBoost step
    if args.run_catboost:
        pipeline.append({
            "name": "catboost",
            "model": "catboost",
            "params": {
                "iterations": int(args.catboost_iterations),
                "depth": int(args.catboost_depth),
                "learning_rate": float(args.catboost_lr),
            },
        })

    # MLP steps (Q20 includes multiple MLP variants)
    if args.run_mlp:
        for arch in args.mlp_arch_list:
            layers = tuple(int(x) for x in arch.split(",") if x.strip())
            pipeline.append({
                "name": f"mlp_{'_'.join(map(str, layers))}",
                "model": "mlp",
                "params": {
                    "hidden_layers": layers,
                    "max_iter": int(args.mlp_max_iter),
                    "standardize": bool(args.standardize_mlp),
                },
            })

    # CNN step
    if args.run_cnn:
        pipeline.append({
            "name": "cnn",
            "model": "cnn",
            "params": {
                "epochs": int(args.cnn_epochs),
                "batch_size": int(args.cnn_batch_size),
                "lr": float(args.cnn_lr),
                "weight_decay": float(args.cnn_weight_decay),
                "hidden_channels": int(args.cnn_hidden_channels),
                "dropout": float(args.cnn_dropout),
                "standardize": bool(args.standardize_cnn),
            },
        })

    if not pipeline:
        # Default: run everything with the full user-provided grids
        pipeline = []
        for nc in (args.plsda_components_list or [15]):
            pipeline.append({
                "name": f"plsda_nc{nc}",
                "model": "plsda",
                "params": {"n_components": int(nc), "standardize": bool(args.standardize_plsda)},
            })

        pipeline.append({
            "name": "catboost",
            "model": "catboost",
            "params": {
                "iterations": int(args.catboost_iterations),
                "depth": int(args.catboost_depth),
                "learning_rate": float(args.catboost_lr),
            },
        })

        for arch in (args.mlp_arch_list or ["128,64"]):
            layers = tuple(int(x) for x in arch.split(",") if x.strip())
            pipeline.append({
                "name": f"mlp_{'_'.join(map(str, layers))}",
                "model": "mlp",
                "params": {
                    "hidden_layers": layers,
                    "max_iter": int(args.mlp_max_iter),
                    "standardize": bool(args.standardize_mlp),
                },
            })

        pipeline.append({
            "name": "cnn",
            "model": "cnn",
            "params": {
                "epochs": int(args.cnn_epochs),
                "batch_size": int(args.cnn_batch_size),
                "lr": float(args.cnn_lr),
                "weight_decay": float(args.cnn_weight_decay),
                "hidden_channels": int(args.cnn_hidden_channels),
                "dropout": float(args.cnn_dropout),
                "standardize": bool(args.standardize_cnn),
            },
        })

    return pipeline


# ===================== CLI ===================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baselines on all classification datasets (Q20-style pipeline).")

    p.add_argument("--data_root", type=str, default="Data/classification")
    p.add_argument("--output_dir", type=str, default="Results/baselines_classif_raw")
    p.add_argument("--seed", type=int, default=42)

    # Optional filters
    p.add_argument("--only_type", type=str, default=None)
    p.add_argument("--only_dataset", type=str, default=None)

    # Which models
    p.add_argument("--run_catboost", action="store_true")
    p.add_argument("--run_plsda", action="store_true")
    p.add_argument("--run_cnn", action="store_true")
    p.add_argument("--run_mlp", action="store_true")

    # CatBoost params
    p.add_argument("--catboost_iterations", type=int, default=500)
    p.add_argument("--catboost_depth", type=int, default=8)
    p.add_argument("--catboost_lr", type=float, default=0.1)

    # PLS-DA params
    p.add_argument("--plsda_components_list", type=int, nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--standardize_plsda", action="store_true")

    # MLP params
    p.add_argument("--mlp_arch_list", type=str, nargs="+", default=["128,64", "64,32"])
    p.add_argument("--mlp_max_iter", type=int, default=500)
    p.add_argument("--standardize_mlp", action="store_true")

    # CNN params
    p.add_argument("--cnn_epochs", type=int, default=50)
    p.add_argument("--cnn_batch_size", type=int, default=64)
    p.add_argument("--cnn_lr", type=float, default=1e-3)
    p.add_argument("--cnn_weight_decay", type=float, default=0.0)
    p.add_argument("--cnn_hidden_channels", type=int, default=32)
    p.add_argument("--cnn_dropout", type=float, default=0.2)
    p.add_argument("--standardize_cnn", action="store_true")

    return p.parse_args()


# ===================== Local runner ===================== #

def run_one_model(
    model_spec: Dict[str, Any],
    X_train: np.ndarray,
    y_train_enc: np.ndarray,
    X_test: np.ndarray,
    y_test_enc: np.ndarray,
    le: LabelEncoder,
    seed: int,
) -> Tuple[str, np.ndarray, Optional[np.ndarray], Dict[str, Any], Dict[str, float]]:
    """
    Execute one model spec from the Q20-style pipeline and return:
    - model_name (string)
    - y_pred_enc
    - y_proba
    - meta
    - metrics
    """
    model_key = model_spec["model"]
    model_name = model_spec["name"]
    params = model_spec.get("params", {})

    n_classes = int(len(le.classes_))

    if model_key == "catboost":
        y_pred, y_proba, meta = fit_predict_catboost(
            X_train=X_train,
            y_train=y_train_enc,
            X_test=X_test,
            n_classes=n_classes,
            seed=seed,
            params=params,
        )
    elif model_key == "plsda":
        y_pred, y_proba, meta = fit_predict_plsda(
            X_train=X_train,
            y_train=y_train_enc,
            X_test=X_test,
            n_classes=n_classes,
            n_components=int(params["n_components"]),
            standardize=bool(params.get("standardize", False)),
        )
    elif model_key == "mlp":
        y_pred, y_proba, meta = fit_predict_mlp(
            X_train=X_train,
            y_train=y_train_enc,
            X_test=X_test,
            seed=seed,
            hidden_layers=tuple(params["hidden_layers"]),
            max_iter=int(params["max_iter"]),
            standardize=bool(params.get("standardize", False)),
        )
    elif model_key == "cnn":
        y_pred, y_proba, meta = fit_predict_cnn(
            X_train=X_train,
            y_train=y_train_enc,
            X_test=X_test,
            n_classes=n_classes,
            seed=seed,
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
            hidden_channels=int(params["hidden_channels"]),
            dropout=float(params["dropout"]),
            standardize=bool(params.get("standardize", False)),
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    metrics = compute_metrics(y_test_enc, y_pred, y_proba, n_classes=n_classes)
    return model_name, y_pred, y_proba, meta, metrics


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(data_root)

    if args.only_type is not None:
        datasets = [d for d in datasets if d["dataset_type"] == args.only_type]
    if args.only_dataset is not None:
        datasets = [d for d in datasets if d["dataset_name"] == args.only_dataset]

    if not datasets:
        raise RuntimeError("No datasets found with expected classification structure.")

    # Q20-style pipeline definition
    pipeline = build_pipeline_from_args(args)

    print("=" * 80)
    print("BASELINE CLASSIFICATION (Q20-STYLE PIPELINE)")
    print("=" * 80)
    print(f"Found {len(datasets)} datasets.")
    print(f"Output dir: {out_dir}")
    print("Pipeline steps:")
    for step in pipeline:
        print(f"  - {step['name']} ({step['model']})")

    summary_rows: List[Dict[str, Any]] = []

    for i, d in enumerate(datasets, start=1):
        ds_id = DatasetId(d["dataset_type"], d["dataset_name"])
        ds_path = d["path"]

        print("\n" + "=" * 100)
        print(f"[{i}/{len(datasets)}] Dataset: {ds_id.tag}")
        print(f"Path: {ds_path}")
        print("=" * 100)

        X_train, y_train, X_test, y_test = load_classification_dataset(ds_path)
        y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

        for step in pipeline:
            model_name = step["name"]

            try:
                mn, y_pred_enc, y_proba, meta, metrics = run_one_model(
                    model_spec=step,
                    X_train=X_train,
                    y_train_enc=y_train_enc,
                    X_test=X_test,
                    y_test_enc=y_test_enc,
                    le=le,
                    seed=args.seed,
                )

                y_pred_labels = le.inverse_transform(y_pred_enc.astype(int))

                metrics_payload = {
                    "dataset_type": ds_id.dataset_type,
                    "dataset_name": ds_id.dataset_name,
                    "model_name": mn,
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "n_features": int(X_train.shape[1]),
                    "n_classes": int(len(le.classes_)),
                    "classes": [str(c) for c in le.classes_],
                    "seed": int(args.seed),
                    "metrics": metrics,
                    "model_meta": meta,
                }

                save_outputs(
                    output_dir=out_dir,
                    dataset_id=ds_id,
                    model_name=mn,
                    y_true_labels=y_test,
                    y_pred_labels=y_pred_labels,
                    y_true_enc=y_test_enc,
                    y_pred_enc=y_pred_enc,
                    le=le,
                    y_proba=y_proba,
                    metrics_payload=metrics_payload,
                )

                row = {
                    "dataset_type": ds_id.dataset_type,
                    "dataset_name": ds_id.dataset_name,
                    "model_name": mn,
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "n_features": int(X_train.shape[1]),
                    "n_classes": int(len(le.classes_)),
                    "seed": int(args.seed),
                    **metrics,
                }
                summary_rows.append(row)

                print(f"✅ {mn}: acc={metrics.get('accuracy', float('nan')):.4f} "
                      f"bal_acc={metrics.get('balanced_accuracy', float('nan')):.4f}")

            except Exception as e:
                print(f"❌ {model_name} failed: {e}")

    df_new = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_metrics.csv"

    # If a previous summary exists, merge (append) and deduplicate
    if summary_path.exists():
        df_old = pd.read_csv(summary_path)
        df_all = pd.concat([df_old, df_new], axis=0, ignore_index=True)
    else:
        df_all = df_new

    # Deduplicate by dataset + model_name (keep the newest run)
    key_cols = ["dataset_type", "dataset_name", "model_name"]
    df_all = df_all.drop_duplicates(subset=key_cols, keep="last")

    # Optional sorting
    sort_cols = [c for c in ["balanced_accuracy", "accuracy"] if c in df_all.columns]
    if sort_cols:
        df_all = df_all.sort_values(sort_cols, ascending=False)

    df_all.to_csv(summary_path, index=False)

    print("\n✅ Done.")
    print(f"Saved summary → {summary_path}")


if __name__ == "__main__":
    main()
