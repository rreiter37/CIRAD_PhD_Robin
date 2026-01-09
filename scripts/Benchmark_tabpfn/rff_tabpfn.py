#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline TabPFN pipeline with Random Fourier Features (RFF) encoding.

This script mirrors baseline_sota.py exactly, with the only difference
being the insertion of a Random Fourier Feature (RFF) encoding step
before the TabPFN model.
"""

from pathlib import Path
import os
import argparse
import numpy as np
import pandas as pd
import torch

from dotenv import load_dotenv
from huggingface_hub import login

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import ASLSBaseline

from tabpfn import TabPFNRegressor, TabPFNClassifier


# ===================== TabPFN device safety ===================== #

def get_safe_tabpfn_device():
    """Select CPU if CUDA architecture is unsupported by TabPFN."""
    if not torch.cuda.is_available():
        return "cpu"

    major, _ = torch.cuda.get_device_capability(0)
    if major >= 9:
        print("⚠️ Unsupported GPU architecture detected → forcing CPU.")
        return "cpu"

    return "cuda"


TABPFN_DEVICE = get_safe_tabpfn_device()


# ===================== RFF Transformer ===================== #

class RFFEncoding(BaseEstimator, TransformerMixin):
    """
    Random Fourier Feature encoding for RBF kernels.
    """

    def __init__(self, n_components=256, sigma=1.0, append_raw=True, random_state=42):
        self.n_components = n_components
        self.sigma = sigma
        self.append_raw = append_raw
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        d = X.shape[1]
        self.W_ = rng.normal(0.0, 1.0 / self.sigma, size=(d, self.n_components))
        self.b_ = rng.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        Z = X @ self.W_ + self.b_
        Z = np.sqrt(2 / self.n_components) * np.concatenate(
            [np.cos(Z), np.sin(Z)], axis=1
        )
        return np.hstack([X, Z]) if self.append_raw else Z


# ===================== Argument parsing ===================== #

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--workspace", type=str, default="workspace_rff_tabpfn")

    parser.add_argument("--rff_components", type=int, default=256)
    parser.add_argument("--rff_sigma", type=float, default=1.0)
    parser.add_argument("--rff_append_raw", action="store_true")
    parser.add_argument("--rff_seed", type=int, default=42)

    return parser.parse_args()


args = parse_args()


# ===================== Environment setup ===================== #

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ===================== Dataset configuration ===================== #

DATA_PATH = args.datasets if args.datasets is not None else [
    "Data/Regression/Beer_OriginalExtract_60_KS"
]

TASK_TYPE = "regression"
AGGREGATION_KEY = None

TabPFNModel = TabPFNRegressor if TASK_TYPE == "regression" else TabPFNClassifier
tabpfn_real_path = "tabpfn-v2.5-regressor-v2.5_real.ckpt"


# ===================== Pipeline definition ===================== #

pipeline = [
    ASLSBaseline(),
    {"y_processing": StandardScaler()},
    StandardScaler(),
    {"split": SPXYGFold(n_splits=3, random_state=42)},
    PCA(n_components=0.99, random_state=42, whiten=True),
    RFFEncoding(
        n_components=args.rff_components,
        sigma=args.rff_sigma,
        append_raw=args.rff_append_raw,
        random_state=args.rff_seed,
    ),
    {
        "model": TabPFNModel(
            n_estimators=4,
            device=TABPFN_DEVICE,
            random_state=42,
            model_path=tabpfn_real_path,
            ignore_pretraining_limits=True,
        ),
        "name": "TabPFN-RFF",
    },
]


pipeline_config = PipelineConfigs(pipeline, "TabPFN_RFF")
dataset_config = DatasetConfigs(DATA_PATH, task_type=TASK_TYPE)

runner = PipelineRunner(verbose=0, workspace_path=args.workspace)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)
