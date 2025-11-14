import os
import sys

# ---------------------------------------------------------------
# Add the Adastra scripts directory to PYTHONPATH
# This ensures imports work regardless of where the script is run
# ---------------------------------------------------------------
sys.path.append("/lus/home/CT10/cad17070/rreiter/phd_robin/scripts")

# ---------------------------------------------------------------
# Set environment variables for deterministic and silent execution
# ---------------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '42'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ---------------------------------------------------------------
# Clear local cache directories to avoid stale artifacts
# ---------------------------------------------------------------
import shutil
if os.path.exists(".cache"):
    shutil.rmtree(".cache")

LOG_DIR = "lightning_logs"
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)

# ---------------------------------------------------------------
# TensorBoard autofire utility
# ---------------------------------------------------------------
import socket
import subprocess

def is_port_in_use(port):
    """Check if a TCP port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

TENSORBOARD_PORT = 6006
if not is_port_in_use(TENSORBOARD_PORT):
    print(f"Lancement de TensorBoard sur http://localhost:{TENSORBOARD_PORT} …")
    subprocess.Popen(
        ["tensorboard", "--logdir", LOG_DIR, f"--port={TENSORBOARD_PORT}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
else:
    print(f"TensorBoard est déjà actif sur http://localhost:{TENSORBOARD_PORT}")

# ---------------------------------------------------------------
# Standard scientific imports
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import tensorflow as tf
import json
import time
import argparse

# ---------------------------------------------------------------
# Adastra absolute paths for DATA, RESULTS and FIGURES
# ---------------------------------------------------------------
BASE_DIR = "/lus/home/CT10/cad17070/rreiter/phd_robin"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "Figures")

# Ensure the dirs exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Imports from your project (updated path: scripts.*)
# ---------------------------------------------------------------
import nirs4all.operators.transformations as pp

from scripts.utils.ensure_dataframe import EnsureDataFrame
from scripts.utils.utils_bdd import split_data
from scripts.utils.build_filename import build_filename
from scripts.utils.make_serializable import make_json_serializable
from scripts.Model_optim.pls_components_hybrid import get_pls_component_candidates
from scripts.utils.correct_class_unbalances import correct_class_unbalances

from scripts.Models.DeepLearning.Train_predict.nicon_optuna import NiconOptunaRegressor
from scripts.Models.DeepLearning.Train_predict.nicon_optuna_classif import NiconOptunaClassifier
from scripts.Models.PLS.PLS_opti import AutoPLSRegression
from scripts.Models.PLS.PLS_opti_classif import AutoPLSDAClassifier
from scripts.Models.Ridge.Ridge_opti import RidgeCVRegressor
from scripts.Models.Ridge.Ridge_opti_classif import RidgeCVClassifier
from scripts.Models.LGBM.LGBM_optuna import LGBMOptuna
from scripts.Models.LGBM.LGBM_optuna_classif import LGBMOptunaClassifier

# ---------------------------------------------------------------
# sklearn and parallel utils
# ---------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error, accuracy_score, f1_score, confusion_matrix
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from itertools import combinations
from joblib import Parallel, delayed, Memory
memory = Memory(".cache", verbose=0)

from tqdm import tqdm

# ---------------------------------------------------------------
# Silence irrelevant warnings
# ---------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cross_decomposition._pls")

from warnings import simplefilter, filterwarnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------
parser = argparse.ArgumentParser(description="Association model/preprocessing with heatmap")

parser.add_argument('--mode', type=str, choices=["Regression", "Classification"], required=True,
                    help="Task type: 'Regression' or 'Classification'.")

parser.add_argument('--data_source', type=str, required=True,
                    help="Dataset name (e.g., 'BeerOriginalExtract', 'CoffeeSpecies', etc.).")

parser.add_argument('--top_n_preprocs', type=int, default=None,
                    help="Optional: keep only top N preprocessings based on score.")

parser.add_argument('--model_names', nargs='+', type=str, default=None,
                    help="Optional: restrict execution to specific models. If None, all models are used.")

parser.add_argument('--progressive_optim', action='store_true', default=False,
                    help="Enable progressive optimization: first combination deep search, next ones reduced search.")

parser.add_argument('--compare_optim_strat', action='store_true', default=False,
                    help="Compare optimization strategies of a single model on the same heatmap.")

parser.add_argument('--only_colors', action='store_true', default=False,
                    help="Heatmap without annotations.")

parser.add_argument('--random_seed', type=int, default=42,
                    help="Global random seed (default=42).")

parser.add_argument('--use_parallelism', action='store_true', default=False,
                    help="Evaluate model/preprocessing combinations in parallel.")

parser.add_argument('--adaptive_batch_size', type=str, choices=["False", "static", "dynamic"], default="False",
                    help="Adaptive batch size strategy for CNN models.")

args = parser.parse_args()
mode = args.mode
data_source = args.data_source
print(f"[INFO] Running on dataset: {data_source}")

only_colors = args.only_colors
use_parallelism = args.use_parallelism

# ---------------------------------------------------------------
# Global random seed for reproducibility
# ---------------------------------------------------------------
rd_seed = args.random_seed
np.random.seed(rd_seed)
random.seed(rd_seed)
tf.random.set_seed(rd_seed)

top_n = args.top_n_preprocs
model_names = args.model_names
progressive_optim = args.progressive_optim
compare_optim_strat = args.compare_optim_strat

adaptive_batch_size = args.adaptive_batch_size
if adaptive_batch_size not in ["False", "static", "dynamic"]:
    adaptive_batch_size = "False"

import torch
torch.manual_seed(rd_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()

# ---------------------------------------------------------------
# LOAD DATA (using absolute data directory)
# ---------------------------------------------------------------
# The split_data() function MUST internally use DATA_DIR.
# If it currently uses relative paths, update it accordingly.
# ---------------------------------------------------------------
Xcal, Ycal, Xval, Yval = split_data(mode, data_source, data_dir=DATA_DIR, verbose=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------------------------------------------------------
# Determine number of classes (classification only)
# ---------------------------------------------------------------
num_classes = len(np.unique(Ycal))
if mode == "Classification":
    print("Number of detected classes:", num_classes)

# ---------------------------------------------------------------
# Scale Y in regression mode
# ---------------------------------------------------------------
scaler_Y = None
if mode == "Regression":
    scaler_Y = MinMaxScaler()
    Ycal = scaler_Y.fit_transform(np.array(Ycal).reshape(-1, 1)).ravel()
    Yval = scaler_Y.transform(np.array(Yval).reshape(-1, 1)).ravel()

# ---------------------------------------------------------------
# PREPROCESSING PIPELINES
# ---------------------------------------------------------------
simple_preprocs = [
    ('id', pp.IdentityTransformer()),
    ('baseline', pp.Baseline()),
    ('derivate', pp.Derivate()),
    ('detrend', pp.Detrend()),
    ('MSC', pp.MultiplicativeScatterCorrection()),
    ('normalize', pp.Normalize()),
    ('RNV', pp.RobustStandardNormalVariate()),
    ('savgol', pp.SavitzkyGolay()),
    ('simplescale', pp.SimpleScale()),
    ('SNV', pp.StandardNormalVariate()),
    ('haar', pp.Wavelet('haar')),
    ('gaussian', pp.Gaussian(order=2, sigma=1)),
]

preprocessings = list(simple_preprocs)

# Add combinations of two preprocessing methods (except identity)
for (name1, trans1), (name2, trans2) in combinations(simple_preprocs[1:], 2):
    combo_name = f"{name1}_{name2}"
    combo_pipeline = Pipeline([
        (name1, trans1),
        (name2, trans2)
    ])
    preprocessings.append((combo_name, combo_pipeline))

# Add PCA transformation
preprocessings.append(('PCA', PCA(random_state=rd_seed)))

# ---------------------------------------------------------------
# PROGRESSIVE OPTIMIZATION PARAMETERS
# ---------------------------------------------------------------
if progressive_optim:
    # Deep search for first preprocessing
    n_trials_first = 500 if mode == "Regression" else 100
    n_trials_next  = 200  if mode == "Regression" else 20
    epochs_first   = 100
    epochs_next    = 40
    patience_optuna_first = 30
    patience_optuna_next = 15
else:
    n_trials_uniform = 90
    epochs_uniform   = 10

epochs, patience = 10000, 1000

# ---------------------------------------------------------------
# MODEL DICTIONARY
# ---------------------------------------------------------------
dict_models = {
    "Ridge_reg": RidgeCVRegressor(
        alphas=np.logspace(-4, 2, 50), cv=5, random_state=rd_seed
    ),
    "PLS_reg": AutoPLSRegression(
        cv=3, seed=rd_seed
    ),
    "LGBM_reg": LGBMOptuna(
        cv=5, n_trials=20, random_state=rd_seed, verbose=1, verbose_optuna=False
    ),
    "CNN_reg": NiconOptunaRegressor(
        n_trials=90, epochs=epochs, patience=patience, cyclic_learning=True,
        lr_min=1e-6, lr_max=1e-3, epochs_optuna=10,
        random_state=rd_seed, device=device, verbose_optuna=True
    ),

    "Ridge_classif": RidgeCVClassifier(
        alphas=np.logspace(-4, 2, 50), cv=5, random_state=rd_seed
    ),
    "PLS_classif": AutoPLSDAClassifier(
        cv=5, seed=rd_seed
    ),
    "LGBM_classif": LGBMOptunaClassifier(
        cv=5, n_trials=50, random_state=rd_seed, verbose=0
    ),
    "CNN_classif": NiconOptunaClassifier(
        num_classes=num_classes, n_trials=50, epochs=10000, patience=10,
        epochs_optuna=100, random_state=rd_seed, verbose_optuna=True,
        device=device
    ),
}

# ---------------------------------------------------------------
# SELECT THE MODELS TO USE
# ---------------------------------------------------------------
suffix = "_reg" if mode == "Regression" else "_classif"

if model_names is None:
    models = [
        ("Ridge" + suffix, dict_models["Ridge" + suffix]),
        ("PLS" + suffix,   dict_models["PLS"   + suffix]),
        ("LGBM" + suffix,  dict_models["LGBM"  + suffix]),
        ("CNN" + suffix,   dict_models["CNN"   + suffix]),
    ]
else:
    models = [(name + suffix, dict_models[name + suffix]) for name in model_names]

# ---------------------------------------------------------------
# STORAGE VARIABLES
# ---------------------------------------------------------------
prior_components = []
best_trials_nicon = None
best_trials_lgbm = None
cnn_batch_sizes = {}
metrics = []
timings = []

# Build list of combinations (pp, model)
combinations = [
    (pp_name, pp_method, mdl_name, mdl)
    for (pp_name, pp_method) in preprocessings
    for (mdl_name, mdl) in models
]

# ---------------------------------------------------------------
# Evaluation function for each (preprocessing, model) pair
# ---------------------------------------------------------------
def evaluate_combination(
    pp_name, pp_method, mdl_name, mdl, mode,
    Xcal, Ycal, Xval, Yval, metrics, progressive_optim,
    best_trials, rd_seed, scaler_Y=None
):
    """Evaluate a preprocessing-model pair and return metrics + timing.

    This function handles:
    - preprocessing pipelines
    - progressive Optuna optimization (for CNN + LGBM)
    - extraction of best_trials and PLS optimal components
    """

    combo_start = time.time()

    # Apply deterministic seeds
    np.random.seed(rd_seed)
    random.seed(rd_seed)
    tf.random.set_seed(rd_seed)

    X_train, X_test = np.asarray(Xcal), np.asarray(Xval)
    Y_train, Y_test = np.asarray(Ycal).ravel(), np.asarray(Yval).ravel()

    try:
        # -----------------------------------------------------------
        # Adjust parameters depending on dataset size
        # -----------------------------------------------------------
        big_dataset = Xcal.shape[0] > 1e3

        # -----------------------------------------------------------
        # ----------   PLS MODELS   ---------------------------------
        # -----------------------------------------------------------
        if mdl_name.startswith('PLS'):

            parallelism = big_dataset
            candidates = np.arange(1, 81)

            if mode == "Regression":
                mdl = AutoPLSRegression(
                    cv=5, scale=True, seed=rd_seed,
                    candidate_components=candidates
                )
            else:
                mdl = AutoPLSDAClassifier(
                    cv=5, scale=True, seed=rd_seed,
                    candidate_components=candidates,
                    parallelism=parallelism
                )

        # -----------------------------------------------------------
        # ----------   LGBM MODELS   ---------------------------------
        # -----------------------------------------------------------
        elif mdl_name.startswith("LGBM"):

            cv = 5 if big_dataset else 10
            subsampling_rate = 0.3 if big_dataset else None

            # Full Optuna search per preprocessing
            best_trials=None
            n_trials = 200 # Increased number for Adastra

            if mode == "Regression":
                mdl = LGBMOptuna(
                    cv=cv, 
                    n_trials=n_trials, 
                    random_state=rd_seed,
                    verbose=0, 
                    verbose_optuna=False,
                    scoring="neg_mean_squared_error",
                    best_trials=best_trials,
                    name_pp=pp_name,
                    subsampling_rate=subsampling_rate
                )
            else:
                mdl = LGBMOptunaClassifier(
                    cv=cv, 
                    n_trials=n_trials, 
                    random_state=rd_seed,
                    verbose=0, 
                    verbose_optuna=False,
                    scoring="log_loss",
                    best_trials=best_trials,
                    name_pp=pp_name,
                    subsampling_rate=subsampling_rate
                )

        # -----------------------------------------------------------
        # ----------   CNN MODELS (NICON)   --------------------------
        # -----------------------------------------------------------
        elif mdl_name.startswith('CNN'):

            if not progressive_optim:
                best_trials = None
            else:
                print("[INFO] Number of CNN best trials used:", len(best_trials) if best_trials else 0)

            if best_trials is None:
                # First deep search
                n_trials_use = n_trials_first if progressive_optim else n_trials_uniform
                epochs_optuna_use = epochs_first if progressive_optim else epochs_uniform
                patience_optuna = patience_optuna_first if progressive_optim else 100
            else:
                # Reduced search space
                n_trials_use = n_trials_next
                epochs_optuna_use = epochs_next
                patience_optuna = patience_optuna_next

            if mode == "Regression":
                mdl = NiconOptunaRegressor(
                    n_trials=n_trials_use,
                    epochs=epochs, 
                    patience=patience,
                    patience_optuna = patience_optuna,
                    cyclic_learning=True, lr_min=1e-6, lr_max=1e-3,
                    epochs_optuna=epochs_optuna_use,
                    random_state=rd_seed,
                    device=device,
                    verbose_optuna=True,
                    best_trials=best_trials,
                    name_pp=pp_name,
                    adaptive_batch_size=adaptive_batch_size
                )
            else:
                mdl = NiconOptunaClassifier(
                    num_classes=num_classes,
                    n_trials=n_trials_use,
                    epochs=epochs, patience=patience,
                    epochs_optuna=epochs_optuna_use,
                    cyclic_learning=True, lr_min=1e-6, lr_max=1e-3,
                    parallelize=False,
                    random_state=rd_seed,
                    verbose_optuna=True,
                    device=device,
                    best_trials=best_trials,
                    name_pp=pp_name
                )

        # -----------------------------------------------------------
        # BUILD PIPELINE
        # -----------------------------------------------------------
        if mdl_name.startswith("LGBM"):
            pipe = Pipeline([
                ("prep", pp_method),
                ("ensure_df", EnsureDataFrame()),
                ("model", clone(mdl)),
            ], memory=memory)
        else:
            pipe = Pipeline([
                ("prep", pp_method),
                ("model", clone(mdl)),
            ], memory=None if mdl_name.startswith("CNN") else memory)

        # -----------------------------------------------------------
        # FIT MODEL
        # -----------------------------------------------------------
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            pipe.fit(X_train, Y_train)

        # -----------------------------------------------------------
        # PREDICT + METRICS
        # -----------------------------------------------------------
        y_pred = pipe.predict(X_test)

        if mode == 'Regression':
            Y_true_orig = scaler_Y.inverse_transform(Y_test.reshape(-1, 1)).ravel()
            Y_pred_orig = scaler_Y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

            range_y = np.max(Y_true_orig) - np.min(Y_true_orig)
            rmse = root_mean_squared_error(Y_true_orig, Y_pred_orig)
            metric = rmse / range_y if range_y != 0 else np.nan

        else:
            metric = accuracy_score(Y_test, y_pred.ravel())
            metric_f1 = f1_score(Y_test, y_pred.ravel(), average='weighted')

            cm = confusion_matrix(Y_test, y_pred.ravel(), labels=np.unique(Y_test))
            fp = cm.sum(axis=0) - np.diag(cm)
            tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
            fpr = np.mean(fp / (fp + tn))

        # Detect abnormal values
        mean_metric = np.nanmean(metrics) if len(metrics) > 0 else None
        if mode == "Regression" and mean_metric is not None and abs(metric) > 10 * mean_metric:
            metric = np.nan
        elif mode == "Classification" and mean_metric is not None and metric < 0.4 * mean_metric:
            metric = np.nan

        metrics.append(metric)

        trained_model = pipe.named_steps["model"]

        # -----------------------------------------------------------
        # STORE HYPERPARAMS / OPTIMAL COMPONENTS
        # -----------------------------------------------------------
        updated_best_trials = None

        if mdl_name.startswith("PLS") and hasattr(trained_model, 'best_n_component_'):
            optimal_comp = trained_model.best_n_component_
            if pp_name not in prior_components:
                prior_components.append(optimal_comp)
                print("Updated PLS prior components:", prior_components)

        elif mdl_name.startswith("CNN"):
            # Store batch size(s)
            if adaptive_batch_size == "dynamic" and hasattr(trained_model, 'batch_size_history'):
                cnn_batch_sizes[pp_name] = trained_model.batch_size_history
            else:
                cnn_batch_sizes[pp_name] = trained_model.batch_size

            if hasattr(trained_model, 'best_trials'):
                updated_best_trials = trained_model.best_trials

        combo_time = time.time() - combo_start

        # -----------------------------------------------------------
        # RETURN VALUES
        # -----------------------------------------------------------
        if mode == "Regression":
            return (pp_name, mdl_name, metric, updated_best_trials, combo_time)

        else:
            return (pp_name, mdl_name, metric, metric_f1, fpr, updated_best_trials, combo_time)

    except Exception as e:
        print(f"[ERROR] {pp_name} + {mdl_name}: {e}")
        combo_time = time.time() - combo_start

        if mode == "Regression":
            return (pp_name, mdl_name, np.nan, None, combo_time)
        else:
            return (pp_name, mdl_name, np.nan, np.nan, np.nan, None, combo_time)


# ---------------------------------------------------------------
# EXECUTION OF ALL COMBINATIONS
# ---------------------------------------------------------------
results = []
results_f1 = [] if mode == "Classification" else None
results_fpr = [] if mode == "Classification" else None

if use_parallelism:
    print("[INFO] Running combinations in PARALLEL mode.")
    raw_results = Parallel(n_jobs=-1)(
        delayed(evaluate_combination)(
            pp_name, pp_method, mdl_name, mdl, mode,
            Xcal, Ycal, Xval, Yval, metrics,
            progressive_optim,
            best_trials_nicon if mdl_name.startswith("CNN") else best_trials_lgbm if mdl_name.startswith("LGBM") else None,
            rd_seed, scaler_Y
        )
        for (pp_name, pp_method, mdl_name, mdl) in tqdm(combinations, desc="Evaluations")
    )

    if mode == "Regression":
        results = [(pp, m, score) for (pp, m, score, _, _) in raw_results]
    else:
        results = [(pp, m, acc) for (pp, m, acc, _, _, _, _) in raw_results]
        results_f1 = [(pp, m, f1) for (pp, m, _, f1, _, _, _) in raw_results]
        results_fpr = [(pp, m, fpr) for (pp, m, _, _, fpr, _, _) in raw_results]

    timings = [(m, t) for (_, m, _, _, _, t) in raw_results]

    # Update best_trials
    for (_, mdl_name, _, trials, _) in raw_results:
        if mdl_name.startswith("CNN") and trials is not None:
            best_trials_nicon = trials
        elif mdl_name.startswith("LGBM") and trials is not None:
            best_trials_lgbm = trials

else:
    print("[INFO] Running combinations in SEQUENTIAL mode.")
    for (pp_name, pp_method, mdl_name, mdl) in tqdm(combinations, desc=f"Evaluations (sequential) {data_source}"):

        if mode == "Regression":
            pp_name, mdl_name, metric, trials, combo_time = evaluate_combination(
                pp_name, pp_method, mdl_name, mdl, mode,
                Xcal, Ycal, Xval, Yval,
                metrics, progressive_optim,
                best_trials_nicon if mdl_name.startswith("CNN") else None,
                rd_seed, scaler_Y
            )
            results.append((pp_name, mdl_name, metric))
        else:
            pp_name, mdl_name, metric, f1, fpr, trials, combo_time = evaluate_combination(
                pp_name, pp_method, mdl_name, mdl, mode,
                Xcal, Ycal, Xval, Yval,
                metrics, progressive_optim,
                best_trials_nicon if mdl_name.startswith("CNN") else None,
                rd_seed, scaler_Y
            )
            results.append((pp_name, mdl_name, metric))
            results_f1.append((pp_name, mdl_name, f1))
            results_fpr.append((pp_name, mdl_name, fpr))

        timings.append((mdl_name, combo_time))

        # Update best trials
        if mdl_name.startswith("CNN") and trials is not None:
            best_trials_nicon = trials
        elif mdl_name.startswith("LGBM") and trials is not None:
            best_trials_lgbm = trials

# ---------------------------------------------------------------
# BUILD DATAFRAMES FROM RAW RESULTS
# ---------------------------------------------------------------
df_scores = pd.DataFrame(results, columns=["Preprocessing", "Model", "Score"])
pivoted = df_scores.pivot(index="Model", columns="Preprocessing", values="Score")

if mode == "Classification":
    # F1-score pivot
    df_scores_f1 = pd.DataFrame(results_f1, columns=["Preprocessing", "Model", "Score"])
    pivoted_f1 = df_scores_f1.pivot(index="Model", columns="Preprocessing", values="Score")

    # FPR pivot
    df_scores_fpr = pd.DataFrame(results_fpr, columns=["Preprocessing", "Model", "Score"])
    pivoted_fpr = df_scores_fpr.pivot(index="Model", columns="Preprocessing", values="Score")

# ---------------------------------------------------------------
# SAVE MAIN SCORE PIVOT TO CSV (RESULTS_DIR)
# ---------------------------------------------------------------
optim_type = "progressive" if progressive_optim else "uniform"

output_dir_results = os.path.join(
    RESULTS_DIR, "assoc_pp_model", "per_dataset", data_source
)
os.makedirs(output_dir_results, exist_ok=True)

name_file = build_filename(
    prefix="results",
    data_source=data_source,
    top_n=top_n,
    optim_type=optim_type,
    model_names=model_names,
    adaptive_batch_size=adaptive_batch_size,
    extension="csv",
)
output_path_csv = os.path.join(output_dir_results, name_file)
pivoted.to_csv(output_path_csv)

if mode == "Classification":
    pivoted_f1.to_csv(
        os.path.join(output_dir_results, name_file.replace("results", "F1_results"))
    )
    pivoted_fpr.to_csv(
        os.path.join(output_dir_results, name_file.replace("results", "FPR_results"))
    )

# ---------------------------------------------------------------
# Formatting helpers for heatmap annotations
# ---------------------------------------------------------------
def format_value(x, classification=False):
    """Format value for display in heatmap annotations."""
    if pd.isnull(x):
        return ""
    return f"{(x * 100 if classification else x):.3f}"

def bold_best(df, classification=False):
    """Return a DataFrame of strings with the best value per column in bold."""
    formatted = df.copy()
    for col in df.columns:
        col_values = df[col]
        if col_values.isnull().all():
            continue

        best_idx = col_values.idxmax() if classification else col_values.idxmin()

        for idx in df.index:
            val = df.at[idx, col]
            if pd.isnull(val):
                formatted.at[idx, col] = ""
            elif idx == best_idx:
                formatted.at[idx, col] = r"$\bf{" + format_value(val, classification) + "}$"
            else:
                formatted.at[idx, col] = format_value(val, classification)
    return formatted

formatted_df = bold_best(pivoted, classification=(mode == "Classification"))

# ---------------------------------------------------------------
# Optionally restrict to top N preprocessings
# ---------------------------------------------------------------
if top_n is not None:
    if mode == "Regression":
        best_preprocs = pivoted.mean().sort_values().head(top_n).index
    else:
        best_preprocs = pivoted.mean().sort_values(ascending=False).head(top_n).index

    pivoted = pivoted[best_preprocs]
    formatted_df = formatted_df[best_preprocs]

# ---------------------------------------------------------------
# MAIN HEATMAP (ACCURACY or RELATIVE RMSE)
# ---------------------------------------------------------------
num_preprocs = len(pivoted.columns)
fig_width = max(14, num_preprocs * 0.25)

from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(fig_width, 5.5 if mode == "Classification" else 5))

heatmap = sns.heatmap(
    pivoted * (100 if mode == "Classification" else 1),
    annot=None if only_colors else formatted_df,
    fmt="" if not only_colors else None,
    linewidths=0.5,
    cmap="YlGnBu" if mode == "Classification" else "viridis",
    cbar_kws={"label": "Accuracy (%)" if mode == "Classification" else "Relative RMSE"},
    xticklabels=True,
    yticklabels=True,
    ax=ax,
)

# Outline best cell(s) in each column
for j, col in enumerate(pivoted.columns):
    values = pivoted[col]
    if values.isnull().all():
        continue

    best_val = values.max() if mode == "Classification" else values.min()
    best_indices = values[values == best_val].index

    for idx in best_indices:
        i = list(pivoted.index).index(idx)
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

plt.xticks(rotation=90, ha="center", fontsize=8 if only_colors else 7)
plt.yticks(rotation=0, fontsize=10 if only_colors else 9)
plt.title(
    f"Performance Heatmap ({'Accuracy' if mode == 'Classification' else 'Relative RMSE'}) / "
    f"{data_source} dataset ({mode})"
)
plt.tight_layout()

# ---------------------------------------------------------------
# SAVE MAIN HEATMAP TO FIGURES_DIR
# ---------------------------------------------------------------
output_dir_figures = os.path.join(
    FIGURES_DIR, "assoc_pp_model", "per_dataset", data_source
)
os.makedirs(output_dir_figures, exist_ok=True)

heatmap_filename = build_filename(
    prefix="heatmap",
    data_source=data_source,
    top_n=top_n,
    optim_type=optim_type,
    model_names=model_names,
    adaptive_batch_size=adaptive_batch_size,
    extension="png",
)

heatmap_path = os.path.join(output_dir_figures, heatmap_filename)
plt.savefig(heatmap_path, dpi=300)
plt.close(fig)

# ---------------------------------------------------------------
# F1 HEATMAP (Classification only)
# ---------------------------------------------------------------
if mode == "Classification":
    num_preprocs_f1 = len(pivoted_f1.columns)
    fig_width_f1 = max(14, num_preprocs_f1 * 0.25)

    fig_f1, ax_f1 = plt.subplots(figsize=(fig_width_f1, 5.5))

    # Note: here we reuse formatted_df for simplicity (can be adapted if needed)
    heatmap_f1 = sns.heatmap(
        pivoted_f1 * 100,
        annot=None if only_colors else bold_best(pivoted_f1, classification=True),
        fmt="" if not only_colors else None,
        linewidths=0.5,
        cmap="YlGnBu",
        cbar_kws={"label": "F1-score (%)"},
        xticklabels=True,
        yticklabels=True,
        ax=ax_f1,
    )

    for j, col in enumerate(pivoted_f1.columns):
        values = pivoted_f1[col]
        if values.isnull().all():
            continue

        best_val = values.max()
        best_indices = values[values == best_val].index

        for idx in best_indices:
            i = list(pivoted_f1.index).index(idx)
            ax_f1.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.xticks(rotation=90, ha="center", fontsize=8 if only_colors else 7)
    plt.yticks(rotation=0, fontsize=10 if only_colors else 9)
    plt.title(f"F1-score Heatmap - {data_source}")
    plt.tight_layout()

    f1_heatmap_path = os.path.join(
        output_dir_figures, heatmap_filename.replace("heatmap", "F1_heatmap")
    )
    plt.savefig(f1_heatmap_path, dpi=300)
    plt.close(fig_f1)

# ---------------------------------------------------------------
# FPR HEATMAP (Classification only)
# ---------------------------------------------------------------
if mode == "Classification":
    # Apply top_n for FPR if requested (best = minimal FPR)
    if top_n is not None:
        best_preprocs_fpr = pivoted_fpr.mean().sort_values().head(top_n).index
        pivoted_fpr = pivoted_fpr[best_preprocs_fpr]

    formatted_fpr = bold_best(pivoted_fpr, classification=False)

    fig_fpr, ax_fpr = plt.subplots(
        figsize=(max(14, len(pivoted_fpr.columns) * 0.25), 5.5)
    )
    heatmap_fpr = sns.heatmap(
        pivoted_fpr * 100,
        annot=None if only_colors else formatted_fpr,
        fmt="" if not only_colors else None,
        linewidths=0.5,
        cmap="YlGnBu_r",
        cbar_kws={"label": "False Positive Rate (%)"},
        xticklabels=True,
        yticklabels=True,
        ax=ax_fpr,
    )

    for j, col in enumerate(pivoted_fpr.columns):
        values = pivoted_fpr[col]
        if values.isnull().all():
            continue
        best_val = values.min()
        best_indices = values[values == best_val].index
        for idx in best_indices:
            i = list(pivoted_fpr.index).index(idx)
            ax_fpr.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.xticks(rotation=90, ha="center", fontsize=8 if only_colors else 7)
    plt.yticks(rotation=0, fontsize=10 if only_colors else 9)
    plt.title(f"False Positive Rate Heatmap - {data_source}")
    plt.tight_layout()

    fpr_heatmap_path = os.path.join(
        output_dir_figures, heatmap_filename.replace("heatmap", "FPR_heatmap")
    )
    plt.savefig(fpr_heatmap_path, dpi=300)
    plt.close(fig_fpr)

# ---------------------------------------------------------------
# GLOBAL TIMING CSV (RESULTS_DIR)
# ---------------------------------------------------------------
elapsed_time = time.time() - start_time

timing_output_dir = os.path.join(
    RESULTS_DIR, "assoc_pp_model", "per_dataset", data_source
)
os.makedirs(timing_output_dir, exist_ok=True)

timing_filename = build_filename(
    prefix="timing_results",
    data_source=data_source,
    top_n=top_n,
    optim_type=optim_type,
    model_names=model_names,
    adaptive_batch_size=adaptive_batch_size,
    extension="csv",
)
timing_csv_path = os.path.join(timing_output_dir, timing_filename)

timing_data = {
    "data_source": data_source,
    "epochs_final": epochs,
    "patience": patience,
    "optimization_type": optim_type,
    "time": elapsed_time,
}

if progressive_optim:
    timing_data["n_trials_first"] = n_trials_first
    timing_data["n_trials_next"] = n_trials_next
    timing_data["epochs_first"] = epochs_first
    timing_data["epochs_next"] = epochs_next
else:
    timing_data["n_trials"] = n_trials_uniform
    timing_data["epochs_optuna"] = epochs_uniform

if os.path.exists(timing_csv_path):
    df_time = pd.read_csv(timing_csv_path)
    df_time = pd.concat([df_time, pd.DataFrame([timing_data])], ignore_index=True)
else:
    df_time = pd.DataFrame([timing_data])

df_time.to_csv(timing_csv_path, index=False)
print(f"[INFO] Execution time saved to {timing_csv_path}")

# ---------------------------------------------------------------
# PER-MODEL TIMING CSV
# ---------------------------------------------------------------
df_time_models = pd.DataFrame(timings, columns=["Model", "Time_seconds"])
df_avg_time = df_time_models.groupby("Model", as_index=False)["Time_seconds"].mean()
df_avg_time["Data_source"] = data_source
df_avg_time["Optimization_type"] = optim_type
df_avg_time["epochs_final"] = epochs
df_avg_time["patience"] = patience

if progressive_optim:
    df_avg_time["n_trials_first"] = n_trials_first
    df_avg_time["n_trials_next"] = n_trials_next
    df_avg_time["epochs_first"] = epochs_first
    df_avg_time["epochs_next"] = epochs_next
else:
    df_avg_time["n_trials"] = n_trials_uniform
    df_avg_time["epochs_optuna"] = epochs_uniform

per_model_filename = timing_filename.replace("timing_results", "timing_per_model")
timing_models_path = os.path.join(timing_output_dir, per_model_filename)

if os.path.exists(timing_models_path):
    df_existing = pd.read_csv(timing_models_path)
    df_avg_time = pd.concat([df_existing, df_avg_time], ignore_index=True)

df_avg_time.to_csv(timing_models_path, index=False)
print(f"[INFO] Per-model execution times saved to {timing_models_path}")

# ---------------------------------------------------------------
# best_trials & batch sizes JSON EXPORTS (RESULTS_DIR)
# ---------------------------------------------------------------
if adaptive_batch_size == "static":
    adaptive_suffix = "_static_batch_size"
elif adaptive_batch_size == "dynamic":
    adaptive_suffix = "_dynamic_batch_size"
else:
    adaptive_suffix = ""

# CNN best_trials
if best_trials_nicon is not None and progressive_optim:
    trials_path_cnn = os.path.join(
        RESULTS_DIR, "assoc_pp_model", "per_dataset", data_source,
        f"best_trials_CNN_{data_source}{adaptive_suffix}.json"
    )
    try:
        with open(trials_path_cnn, "w") as f:
            json.dump(make_json_serializable(best_trials_nicon), f, indent=2)
        print(f"[INFO] CNN best_trials saved to: {trials_path_cnn}")
    except Exception as e:
        print(f"[WARNING] Error while saving CNN best_trials: {e}")

# LGBM best_trials
if best_trials_lgbm is not None and progressive_optim:
    trials_path_lgbm = os.path.join(
        RESULTS_DIR, "assoc_pp_model", "per_dataset", data_source,
        f"best_trials_LGBM_{data_source}{adaptive_suffix}.json"
    )
    try:
        with open(trials_path_lgbm, "w") as f:
            json.dump(make_json_serializable(best_trials_lgbm), f, indent=2)
        print(f"[INFO] LGBM best_trials saved to: {trials_path_lgbm}")
    except Exception as e:
        print(f"[WARNING] Error while saving LGBM best_trials: {e}")

# CNN batch sizes per preprocessing
if len(cnn_batch_sizes) > 0:
    batch_sizes_path = os.path.join(
        RESULTS_DIR, "assoc_pp_model", "per_dataset", data_source,
        f"batch_sizes_CNN_{data_source}{adaptive_suffix}.json"
    )
    try:
        with open(batch_sizes_path, "w") as f:
            json.dump(make_json_serializable(cnn_batch_sizes), f, indent=2)
        print(f"[INFO] CNN batch sizes saved to: {batch_sizes_path}")
    except Exception as e:
        print(f"[WARNING] Error while saving CNN batch sizes: {e}")
