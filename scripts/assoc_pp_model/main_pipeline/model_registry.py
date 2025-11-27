"""
Model registry for the preprocessing/model association pipeline.

This module provides:
- A centralized dictionary of all available models (regression & classification)
- Logic for selecting a user-specified subset of models
- Clean separation between model creation and pipeline orchestration

All models are imported lazily to avoid heavy import costs
unless this module is actually used.
"""

from sklearn.base import clone


# -------------------------------------------------------------
# Build model dictionary
# -------------------------------------------------------------
def get_models(mode, rd_seed, num_classes=None, device="cpu", adaptive_batch_size="False"):
    """
    Build the dictionary of available models depending on the task mode.

    Args:
        mode (str): "Regression" or "Classification"
        rd_seed (int): global random seed
        num_classes (int): number of classes (classification only)
        device (str): "cpu" or "cuda"
        adaptive_batch_size (str): "False" | "static" | "dynamic"

    Returns:
        dict: model_name -> instantiated model
    """

    # Lazy imports (large modules are imported only when needed)
    from scripts.Models.Ridge.Ridge_opti import RidgeCVRegressor
    from scripts.Models.Ridge.Ridge_opti_classif import RidgeCVClassifier

    from scripts.Models.PLS.PLS_opti import AutoPLSRegression
    from scripts.Models.PLS.PLS_opti_classif import AutoPLSDAClassifier

    from scripts.Models.LGBM.LGBM_optuna import LGBMOptuna
    from scripts.Models.LGBM.LGBM_optuna_classif import LGBMOptunaClassifier

    from scripts.Models.DeepLearning.NICON_CNN.regressor import NiconOptunaRegressor
    from scripts.Models.DeepLearning.Train_predict.nicon_optuna_classif import NiconOptunaClassifier

    # -------------------------
    # Regression models
    # -------------------------
    dict_models = {}

    if mode == "Regression":
        dict_models.update({
            "Ridge_reg": RidgeCVRegressor(
                alphas=None,  # internal selection
                cv=5,
                random_state=rd_seed
            ),

            "PLS_reg": AutoPLSRegression(
                cv=3,
                seed=rd_seed
            ),

            "LGBM_reg": LGBMOptuna(
                cv=5,
                n_trials=20,
                random_state=rd_seed,
                verbose=1,
                verbose_optuna=False
            ),

            "CNN_reg": NiconOptunaRegressor(
                n_trials=90,
                epochs=10000,
                patience=1000,
                cyclic_learning=True,
                lr_min=1e-6,
                lr_max=1e-3,
                epochs_optuna=10,
                random_state=rd_seed,
                device=device,
                verbose_optuna=True,
                adaptive_batch_size=adaptive_batch_size
            ),
        })

    # -------------------------
    # Classification models
    # -------------------------
    else:
        dict_models.update({
            "Ridge_classif": RidgeCVClassifier(
                alphas=None,
                cv=5,
                random_state=rd_seed
            ),

            "PLS_classif": AutoPLSDAClassifier(
                cv=5,
                scale=True,
                seed=rd_seed
            ),

            "LGBM_classif": LGBMOptunaClassifier(
                cv=5,
                n_trials=50,
                random_state=rd_seed,
                verbose=0
            ),

            "CNN_classif": NiconOptunaClassifier(
                num_classes=num_classes,
                n_trials=50,
                epochs=10000,
                patience=10,
                epochs_optuna=100,
                cyclic_learning=True,
                lr_min=1e-6,
                lr_max=1e-3,
                parallelize=False,
                random_state=rd_seed,
                verbose_optuna=True,
                device=device
            ),
        })

    return dict_models


# -------------------------------------------------------------
# Filter model set based on CLI argument
# -------------------------------------------------------------
def filter_models(model_names, mode, dict_models):
    """
    Filter the available models according to user CLI parameters.

    Args:
        model_names (list or None): model names without suffix (_reg/_classif)
        mode (str): task mode ("Regression" or "Classification")
        dict_models (dict): dictionary returned by get_models()

    Returns:
        list of (model_name, model_instance) tuples
    """

    suffix = "_reg" if mode == "Regression" else "_classif"

    # If no filtering: use all models
    if model_names is None:
        return [(name, clone(m)) for name, m in dict_models.items()]

    # User-specified subset
    filtered = []
    for base_name in model_names:
        full_name = base_name + suffix
        if full_name not in dict_models:
            raise ValueError(f"[ERROR] Unknown model name: '{full_name}'")
        filtered.append((full_name, clone(dict_models[full_name])))

    return filtered
