# scripts/models/nicon/utils.py

import os
import random
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set global seed for Python, NumPy, and PyTorch (CPU & CUDA) to ensure reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Deterministic algorithms can slow things down but guarantee reproducibility
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """Worker init function to make DataLoader workers deterministic."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def reshape_input(X: Any) -> torch.Tensor:
    """Reshape input to (N, C, L) if it is currently (N, L)."""
    X = np.array(X)
    if len(X.shape) == 2:
        X = X[:, np.newaxis, :]
    return torch.tensor(X, dtype=torch.float32)
