"""
Parallel evaluation module using joblib.

This module provides:
- A clean helper to run evaluate_combination() in parallel
- Proper error resilience and consistent return formatting
- A direct replacement for the original joblib code in association_pp_model.py
"""

from joblib import Parallel, delayed
from tqdm import tqdm

from .evaluation import evaluate_combination


# ======================================================================
# PARALLEL EXECUTION WRAPPER
# ======================================================================

def _parallel_worker(pp_name, pp_method, mdl_name, mdl, cfg, Xcal, Ycal, Xval, Yval):
    """
    Internal wrapper executed by each joblib worker.

    This wrapper exists only to keep joblib happy and avoid issues with closures.
    It directly calls evaluate_combination() and returns its output.
    """
    return evaluate_combination(
        pp_name, pp_method, mdl_name, mdl,
        cfg, Xcal, Ycal, Xval, Yval
    )


def run_parallel_evaluation(combinations, cfg, Xcal, Ycal, Xval, Yval):
    """
    Run the evaluation of all (preprocessing, model) combinations in parallel.

    Args:
        combinations (list): list of tuples (pp_name, pp_method, mdl_name, mdl)
        cfg (dict): global configuration dictionary
        Xcal, Ycal, Xval, Yval: calibration & validation datasets

    Returns:
        list of evaluation results, each returned exactly as evaluate_combination()
    """

    print("[INFO] Executing evaluations in parallel using joblib...")

    # joblib parallel execution
    results = Parallel(n_jobs=-1)(
        delayed(_parallel_worker)(
            pp_name, pp_method, mdl_name, mdl,
            cfg, Xcal, Ycal, Xval, Yval
        )
        for (pp_name, pp_method, mdl_name, mdl) in tqdm(
            combinations, desc="Parallel evaluations"
        )
    )

    return results
