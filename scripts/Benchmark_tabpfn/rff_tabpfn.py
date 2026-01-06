"""
Benchmark TabPFN with vs without Random Fourier Features (RFF) encoding.

- Based on baseline_sota.py pipeline style (nirs4all + PipelineRunner + TabPFN).
- Adds a deterministic sklearn-compatible RFF encoding inspired by "Spectral Adaptivity in TabPFN".
- Runs two pipelines:
    (A) TabPFN baseline (no RFF)
    (B) TabPFN + RFF (RFFEncoding inserted before the model)
- Prints top models and metrics for each dataset.

Notes:
- RFF increases feature dimensionality; TabPFN is configured with ignore_pretraining_limits=True.
- RFF hyperparameters matter (n_components, sigma/scale). This script exposes CLI flags.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from dotenv import load_dotenv
from huggingface_hub import login

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA

from tabpfn import TabPFNRegressor, TabPFNClassifier

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import SavitzkyGolay, ASLSBaseline


# ===================== Fix TabPFN CUDA incompatibility (RTX 50xx) ===================== #

def get_safe_tabpfn_device() -> str:
    """
    Returns 'cuda' if the GPU architecture is supported by TabPFN kernels,
    otherwise falls back to 'cpu' to prevent CUDA kernel crashes.
    """
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available → Using CPU for TabPFN.")
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)

    # TabPFN kernels may not be compiled for newer GPU architectures (e.g., SM >= 90).
    if major >= 9:
        print("⚠️ Detected GPU architecture possibly incompatible with TabPFN CUDA kernels (SM >= 90).")
        print("➡️ Switching TabPFN to CPU mode to avoid CUDA errors.")
        return "cpu"

    return "cuda"


TABPFN_DEVICE = get_safe_tabpfn_device()


# ===================== RFF Encoding Transformer ===================== #

class RFFEncoding(BaseEstimator, TransformerMixin):
    """
    Random Fourier Features (RFF) encoding for tabular inputs.

    This implements a common RFF map for the RBF kernel family:
        z(x) = sqrt(2 / D) * [cos(xW + b), sin(xW + b)]
    where:
        W ~ N(0, 1/sigma^2)
        b ~ Uniform(0, 2*pi)

    Parameters
    ----------
    n_components : int
        Number of random frequencies (D). Output adds 2*D features (cos + sin),
        plus optionally the raw input if append_raw=True.
    sigma : float
        Length-scale parameter. Smaller sigma => larger frequencies (more high-frequency content).
        Larger sigma => smoother, lower-frequency features.
    append_raw : bool
        If True, outputs [X, z(X)]. If False, outputs only z(X).
    use_float32 : bool
        If True, internal computations use float32 (often faster and enough for TabPFN).
    random_state : int
        RNG seed for deterministic W and b.
    """
    def __init__(
        self,
        n_components: int = 256,
        sigma: float = 1.0,
        append_raw: bool = True,
        use_float32: bool = True,
        random_state: int = 42,
    ):
        self.n_components = int(n_components)
        self.sigma = float(sigma)
        self.append_raw = bool(append_raw)
        self.use_float32 = bool(use_float32)
        self.random_state = int(random_state)

        # Learned at fit time
        self.W_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None):
        X = self._check_X(X)
        rng = np.random.RandomState(self.random_state)

        d = X.shape[1]
        # W shape: (d, n_components)
        # For RBF kernel: W ~ N(0, 1/sigma^2)
        W = rng.normal(loc=0.0, scale=1.0 / max(self.sigma, 1e-12), size=(d, self.n_components))
        b = rng.uniform(low=0.0, high=2.0 * np.pi, size=(self.n_components,))

        if self.use_float32:
            W = W.astype(np.float32, copy=False)
            b = b.astype(np.float32, copy=False)

        self.W_ = W
        self.b_ = b
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("RFFEncoding must be fitted before calling transform().")

        X = self._check_X(X)
        if self.use_float32:
            X = X.astype(np.float32, copy=False)

        proj = X @ self.W_  # (n, D)
        proj = proj + self.b_[None, :]  # broadcast phase

        # z(x) scaling
        scale = np.sqrt(2.0 / float(self.n_components))
        Zc = np.cos(proj) * scale
        Zs = np.sin(proj) * scale
        Z = np.concatenate([Zc, Zs], axis=1)

        if self.append_raw:
            return np.concatenate([X, Z], axis=1)
        return Z

    @staticmethod
    def _check_X(X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}.")
        return X


# ===================== CLI ===================== #

@dataclass
class Args:
    datasets: Optional[list[str]]
    workspace: str
    task_type: str
    aggregation_key: Optional[str]

    # RFF knobs
    rff_components: int
    rff_sigma: float
    rff_append_raw: bool
    rff_seed: int

    # General knobs
    seed: int
    n_estimators: int
    pca_variance: float


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of dataset folders to use instead of the default DATA_PATH."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace_rff_tabpfn",
        help="Workspace folder where predictions and logs will be stored."
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="regression",
        choices=["regression", "classification"],
        help="Task type for TabPFN."
    )
    parser.add_argument(
        "--aggregation_key",
        type=str,
        default="ID",
        help="Key used to aggregate predictions (set to 'None' to disable)."
    )

    # RFF parameters
    parser.add_argument("--rff_components", type=int, default=256, help="Number of RFF frequencies (D).")
    parser.add_argument("--rff_sigma", type=float, default=1.0, help="RFF length-scale sigma.")
    parser.add_argument(
        "--rff_append_raw",
        action="store_true",
        help="If set, concatenate raw X with RFF features (recommended)."
    )
    parser.add_argument("--rff_seed", type=int, default=42, help="Seed for RFF random matrix/phase.")

    # General parameters
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--n_estimators", type=int, default=16, help="TabPFN n_estimators.")
    parser.add_argument(
        "--pca_variance",
        type=float,
        default=0.99,
        help="PCA retained variance (e.g., 0.99)."
    )

    ns = parser.parse_args()

    agg_key = None if ns.aggregation_key.lower() == "none" else ns.aggregation_key

    return Args(
        datasets=ns.datasets,
        workspace=ns.workspace,
        task_type=ns.task_type,
        aggregation_key=agg_key,
        rff_components=ns.rff_components,
        rff_sigma=ns.rff_sigma,
        rff_append_raw=bool(ns.rff_append_raw),
        rff_seed=ns.rff_seed,
        seed=ns.seed,
        n_estimators=ns.n_estimators,
        pca_variance=ns.pca_variance,
    )


# ===================== Main ===================== #

def seed_everything(seed: int) -> None:
    """
    Make as many operations deterministic as possible.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hf_login_from_env() -> None:
    """
    Optional HF login to ensure TabPFN checkpoints can be downloaded.
    Mirrors baseline_sota.py behavior.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not found. TabPFN may need authentication depending on your setup.")


def build_pipelines(args: Args) -> Tuple[PipelineConfigs, PipelineConfigs]:
    """
    Build two pipeline configs:
      - baseline: TabPFN without RFF
      - rff:      TabPFN with RFF inserted before the model
    """
    is_reg = args.task_type == "regression"
    TabPFNModel = TabPFNRegressor if is_reg else TabPFNClassifier

    # You can optionally point to a specific TabPFN checkpoint like baseline_sota.py.
    # Here we keep default unless you want to uncomment and pass model_path.
    tabpfn_model = TabPFNModel(
        n_estimators=args.n_estimators,
        device=TABPFN_DEVICE,
        random_state=args.seed,
        ignore_pretraining_limits=True,
    )
    tabpfn_real_path = 'tabpfn-v2.5-regressor-v2.5_real.ckpt'
    tabpfn_model_rff = TabPFNModel(
        n_estimators=args.n_estimators,
        device=TABPFN_DEVICE,
        random_state=args.seed,
        model_path=tabpfn_real_path,
        ignore_pretraining_limits=True,
    )

    common_prefix = [
        ASLSBaseline(),
        {"split": SPXYGFold(n_splits=1, random_state=args.seed), "group": args.aggregation_key},
        {"split": SPXYGFold(n_splits=3, random_state=args.seed), "group": args.aggregation_key},
        PCA(n_components=args.pca_variance, random_state=args.seed, whiten=True),
    ]

    # Baseline pipeline (no RFF)
    pipeline_baseline = [
        *common_prefix,
        {"model": tabpfn_model, "name": "TabPFN"},
    ]

    # RFF pipeline (insert RFF just before TabPFN)
    # We apply RFF AFTER common preprocessing to encode an already normalized feature space.
    rff = RFFEncoding(
        n_components=args.rff_components,
        sigma=args.rff_sigma,
        append_raw=args.rff_append_raw,
        random_state=args.rff_seed,
    )

    pipeline_rff = [
        *common_prefix,
        rff,
        {"model": tabpfn_model_rff, "name": f"TabPFN+RFF(D={args.rff_components},sigma={args.rff_sigma},append_raw={args.rff_append_raw})"},
    ]

    return PipelineConfigs(pipeline_baseline, "TabPFN_NoRFF"), PipelineConfigs(pipeline_rff, "TabPFN_WithRFF")


def run_and_report(
    args: Args,
    dataset_config: DatasetConfigs,
    pipeline_config: PipelineConfigs,
) -> None:
    """
    Run a pipeline and print a compact report similar to baseline_sota.py.
    """
    runner = PipelineRunner(save_files=True, verbose=0, workspace_path=args.workspace)
    predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

    best_model_count = 5
    rank_metric = "rmse" if args.task_type == "regression" else "balanced_accuracy"
    display_metrics = ["rmse", "r2", "mape", "nrmse"] if args.task_type == "regression" else ["accuracy", "balanced_accuracy", "f1", "recall"]

    print("\n" + "#" * 100)
    print(f"PIPELINE: {pipeline_config.names}")
    print("#" * 100)

    for dataset_name, dataset_prediction in predictions_per_dataset.items():
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 80}")

        dataset_predictions = dataset_prediction["run_predictions"]

        print(f"\nTop {best_model_count} models based on validation {rank_metric.upper()}:")
        top_models_val = dataset_predictions.top(
            best_model_count,
            rank_metric,
            rank_partition="val",
            display_metrics=display_metrics
        )
        for idx, pred in enumerate(top_models_val):
            print(f"{idx + 1}. {Predictions.pred_short_string(pred, metrics=display_metrics)} - {pred['preprocessings']}")

        print(f"\nTop {best_model_count} models based on test {rank_metric.upper()}:")
        top_models_test = dataset_predictions.top(
            best_model_count,
            rank_metric,
            rank_partition="test",
            display_metrics=display_metrics
        )
        for idx, pred in enumerate(top_models_test):
            print(f"{idx + 1}. {Predictions.pred_short_string(pred, metrics=display_metrics)} - {pred['preprocessings']}")

        if args.aggregation_key is not None:
            print("*" * 80)
            print(f"\nTop Predictions (aggregated by {args.aggregation_key}):")

            print(f"\nTop {best_model_count} models based on validation {rank_metric.upper()}:")
            top_models_val_agg = dataset_predictions.top(
                best_model_count,
                rank_metric,
                rank_partition="val",
                display_metrics=display_metrics,
                aggregate=args.aggregation_key
            )
            for idx, pred in enumerate(top_models_val_agg):
                print(f"{idx + 1}. {Predictions.pred_short_string(pred, metrics=display_metrics)} - {pred['preprocessings']}")

            print(f"\nTop {best_model_count} models based on test {rank_metric.upper()}:")
            top_models_test_agg = dataset_predictions.top(
                best_model_count,
                rank_metric,
                rank_partition="test",
                display_metrics=display_metrics,
                aggregate=args.aggregation_key
            )
            for idx, pred in enumerate(top_models_test_agg):
                print(f"{idx + 1}. {Predictions.pred_short_string(pred, metrics=display_metrics)} - {pred['preprocessings']}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    hf_login_from_env()

    # Default dataset list if not provided
    if args.datasets is not None:
        data_path = args.datasets
    else:
        data_path = [
            "Data_nirs4all/Regression/Beer_OriginalExtract_60_KS",
        ]

    dataset_config = DatasetConfigs(data_path, task_type=args.task_type)

    pipeline_baseline, pipeline_rff = build_pipelines(args)

    # Run baseline
    run_and_report(args, dataset_config, pipeline_baseline)

    # Run RFF
    run_and_report(args, dataset_config, pipeline_rff)


if __name__ == "__main__":
    main()
