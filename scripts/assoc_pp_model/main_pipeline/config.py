"""
Configuration and environment setup module for the preprocessing/model association pipeline.

This file contains:
- argument parsing
- global configuration dictionary creation
- environment preparation (random seeds, low-level thread limits, CUDA determinism)
- optional TensorBoard auto-launch
All heavy pipeline logic is kept outside this module.
"""

import os
import argparse
import random
import numpy as np
import torch
import tensorflow as tf
import socket
import subprocess
import shutil


# -------------------------------------------------------------
# Utility: check if a given port is already in use
# -------------------------------------------------------------
def _is_port_in_use(port: int) -> bool:
    """Return True if the specified port is already occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


# -------------------------------------------------------------
# Environment setup: seed, threads, TensorBoard, etc.
# -------------------------------------------------------------
def setup_environment(cfg):
    """
    Prepare global environment:
    - Set environment variables
    - Fix random seeds for reproducibility
    - Configure CUDA determinism
    - Limit BLAS/OpenMP threads
    - Launch TensorBoard (optional)
    """

    # -----------------------------
    # Clean temporary directories
    # -----------------------------
    if os.path.exists(".cache"):
        shutil.rmtree(".cache")

    LOG_DIR = "lightning_logs"
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)

    # -----------------------------
    # Set OS-level env variables
    # -----------------------------
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["PYTHONHASHSEED"] = str(cfg["random_seed"])
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # -----------------------------
    # Random seeds (Python, NumPy)
    # -----------------------------
    random.seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])

    # -----------------------------
    # TensorFlow seed
    # -----------------------------
    tf.random.set_seed(cfg["random_seed"])

    # -----------------------------
    # PyTorch seed + deterministic settings
    # -----------------------------
    torch.manual_seed(cfg["random_seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -----------------------------
    # Optional: auto-launch TensorBoard
    # -----------------------------
    tensorboard_port = 6006
    if not _is_port_in_use(tensorboard_port):
        print(f"[INFO] Launching TensorBoard at http://localhost:{tensorboard_port}")
        subprocess.Popen(
            ["tensorboard", "--logdir", LOG_DIR, f"--port={tensorboard_port}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print(f"[INFO] TensorBoard already running at http://localhost:{tensorboard_port}")


# -------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------
def parse_arguments():
    """
    Parse all CLI arguments required by the pipeline.
    These arguments define:
    - mode (Regression/Classification)
    - dataset source
    - optimization strategy
    - adaptive batching
    - parallel execution
    - model selection
    """

    parser = argparse.ArgumentParser(
        description="Association model/preprocessing with heatmap"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["Regression", "Classification"],
        required=True,
        help="Type of task: Regression or Classification",
    )

    parser.add_argument(
        "--data_source",
        type=str,
        required=True,
        help="Name of the dataset to use",
    )

    parser.add_argument(
        "--top_n_preprocs",
        type=int,
        default=None,
        help="Display only the top N preprocessings (optional)",
    )

    parser.add_argument(
        "--model_names",
        nargs="+",
        type=str,
        default=None,
        help="Subset of model names to use (optional)",
    )

    parser.add_argument(
        "--progressive_optim",
        action="store_true",
        default=False,
        help="Enable progressive optimization strategy",
    )

    parser.add_argument(
        "--compare_optim_strat",
        action="store_true",
        default=False,
        help="Compare optimization strategies of a single model (optional)",
    )

    parser.add_argument(
        "--only_colors",
        action="store_true",
        default=False,
        help="Only display heatmap colors without annotation",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Global random seed (default: 42)",
    )

    parser.add_argument(
        "--use_parallelism",
        action="store_true",
        default=False,
        help="Enable joblib-based parallel execution",
    )

    parser.add_argument(
        "--adaptive_batch_size",
        type=str,
        choices=["False", "static", "dynamic"],
        default="False",
        help="Batch size adaptation strategy",
    )

    return parser.parse_args()


# -------------------------------------------------------------
# Build the global configuration dictionary
# -------------------------------------------------------------
def get_config():
    """
    Convert CLI arguments into a structured configuration dictionary.
    This dict is passed to all pipeline modules.
    """
    args = parse_arguments()

    cfg = {
        "mode": args.mode,
        "data_source": args.data_source,
        "top_n_preprocs": args.top_n_preprocs,
        "model_names": args.model_names,
        "progressive_optim": args.progressive_optim,
        "compare_optim_strat": args.compare_optim_strat,
        "only_colors": args.only_colors,
        "random_seed": args.random_seed,
        "use_parallelism": args.use_parallelism,
        "adaptive_batch_size": args.adaptive_batch_size,
    }

    return cfg
