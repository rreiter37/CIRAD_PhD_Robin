"""
Preprocessing pipeline builder for the model/preprocessing association pipeline.

This module provides:
- a list of simple preprocessing operations
- automatic generation of two-step preprocessing combinations
- optional PCA transformation
- consistent naming for all preprocessing pipelines

All transformations rely on the nirs4all.preprocessing package and sklearn Pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from itertools import combinations

import nirs4all.operators.transformations as pp


# -------------------------------------------------------------
# Simple preprocessing definitions
# -------------------------------------------------------------
def get_simple_preprocessors():
    """
    Return the base list of simple preprocessings as (name, transformer) tuples.
    These are the atomic blocks used later for combination generation.
    """
    simple_preprocs = [
        ("id", pp.IdentityTransformer()),
        ("baseline", pp.Baseline()),
        ("derivate", pp.Derivate()),
        ("detrend", pp.Detrend()),
        ("MSC", pp.MultiplicativeScatterCorrection()),
        ("normalize", pp.Normalize()),
        ("RNV", pp.RobustStandardNormalVariate()),
        ("savgol", pp.SavitzkyGolay()),
        ("simplescale", pp.SimpleScale()),
        ("SNV", pp.StandardNormalVariate()),
        ("haar", pp.Wavelet("haar")),
        ("gaussian", pp.Gaussian(order=2, sigma=1)),
    ]
    return simple_preprocs


# -------------------------------------------------------------
# Two-step preprocessing combinations
# -------------------------------------------------------------
def build_two_step_combinations(simple_preprocs):
    """
    Build preprocessing combinations of two simple preprocessors.
    The Identity preprocessor is excluded from combinations.

    Returns:
        list of (name, Pipeline) tuples
    """
    combos = []

    # Exclude "id" from 2-step combinations
    filtered = [p for p in simple_preprocs if p[0] != "id"]

    for (name1, trans1), (name2, trans2) in combinations(filtered, 2):
        combo_name = f"{name1}_{name2}"
        combo_pipe = Pipeline(
            [
                (name1, trans1),
                (name2, trans2),
            ]
        )
        combos.append((combo_name, combo_pipe))

    return combos


# -------------------------------------------------------------
# Complete preprocessing builder
# -------------------------------------------------------------
def build_preprocessing_list(random_seed=42):
    """
    Build the full list of preprocessing methods:
    - simple preprocessors
    - all 2-by-2 preprocessing combinations (except identity-based combinations)
    - PCA transformation

    Args:
        random_seed (int): Seed for PCA reproducibility.

    Returns:
        list of (name, transformer) tuples
    """

    # Base simple preprocessors
    simple_preprocs = get_simple_preprocessors()

    # Two-step combinations
    two_step = build_two_step_combinations(simple_preprocs)

    # PCA transformation
    pca_step = [("PCA", PCA(random_state=random_seed))]

    # Final list = simple + combos + PCA
    preprocessings = list(simple_preprocs) + two_step + pca_step

    return preprocessings