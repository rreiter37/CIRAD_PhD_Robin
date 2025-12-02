"""
Data loading utilities for the preprocessing/model association pipeline.

This module provides:
- A safe and consistent interface for loading datasets
- Calibration/Validation separation
- Automatic conversion to NumPy arrays and flattening of label arrays
- Clear error messages when a dataset is missing or corrupted

The actual dataset reading functions should be implemented in
scripts.utils.load_dataset or any other domain-specific loaders.
"""

import numpy as np
import os

from scripts.utils.utils_bdd import split_data


# ======================================================================
# Internal helpers
# ======================================================================

def _validate_dataset(Xcal, Ycal, Xval, Yval):
    """
    Basic consistency checks to ensure the dataset is valid.

    Raises:
        ValueError: if shapes are inconsistent or arrays are empty.
    """
    if Xcal is None or Xval is None or Ycal is None or Yval is None:
        raise ValueError("[ERROR] Dataset returned None for one or more arrays.")

    if len(Xcal) == 0 or len(Xval) == 0:
        raise ValueError("[ERROR] Empty dataset detected.")

    if Xcal.shape[1] != Xval.shape[1]:
        raise ValueError(
            f"[ERROR] Calibration and validation feature dimensions differ: "
            f"{Xcal.shape[1]} vs {Xval.shape[1]}"
        )

    if len(Ycal.shape) > 2 or len(Yval.shape) > 2:
        raise ValueError("[ERROR] Y should be 1D or (n,1).")

    return True


def _to_numpy_flat(X, Y):
    """
    Convert X and Y to NumPy arrays and flatten Y to shape (n,).

    Returns:
        X (np.ndarray), Y (np.ndarray)
    """
    X = np.asarray(X)
    Y = np.asarray(Y).ravel()
    return X, Y


# ======================================================================
# Public interface
# ======================================================================

def load_dataset_safely(mode, data_source, data_dir=None, verbose=False):
    """
    Load a dataset using the project's canonical loader and perform
    validation and formatting.

    Args:
        mode (str): "Regression" or "Classification"
        data_source (str): dataset name ("DS1", "DS2", etc.)

    Returns:
        Xcal, Ycal, Xval, Yval (NumPy arrays)

    Raises:
        Exception if the dataset is missing, corrupted, or incompatible.
    """

    print(f"[INFO] Loading dataset '{data_source}'...")

    try:
        # Domain-specific loader (your project already uses this)
        Xcal, Ycal, Xval, Yval = split_data(mode=mode, data_source=data_source, data_dir=data_dir, verbose=verbose)

    except Exception as e:
        raise RuntimeError(
            f"[ERROR] Failed to load dataset '{data_source}'.\n"
            f"Original error: {e}"
        ) from e

    # Validate correctness
    _validate_dataset(Xcal, Ycal, Xval, Yval)

    # Convert to consistent NumPy shapes
    Xcal, Ycal = _to_numpy_flat(Xcal, Ycal)
    Xval, Yval = _to_numpy_flat(Xval, Yval)

    print(f"[INFO] Dataset loaded successfully.")
    print(f"       Xcal shape: {Xcal.shape}")
    print(f"       Xval shape: {Xval.shape}")

    return Xcal, Ycal, Xval, Yval
