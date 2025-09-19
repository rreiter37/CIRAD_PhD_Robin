import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '42'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Clear the cache
import shutil
if os.path.exists(".cache"):
    shutil.rmtree(".cache")
LOG_DIR = "lightning_logs"
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)

import socket
import subprocess
# Function to verify if a port is already occupied
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Launch Tensorboard if not yet
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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import tensorflow as tf
import json
import time

import argparse

import nirs4all.transformations as pp

from Scripts_python.utils.ensure_dataframe import EnsureDataFrame
from Scripts_python.utils.utils_bdd import split_data
from Scripts_python.utils.make_serializable import make_json_serializable
from Scripts_python.Model_optim.pls_components_hybrid import get_pls_component_candidates
from Scripts_python.utils.correct_class_unbalances import correct_class_unbalances

from Scripts_python.Models.DeepLearning.Train_predict.nicon_optuna import NiconOptunaRegressor
from Scripts_python.Models.DeepLearning.Train_predict.nicon_optuna_classif import NiconOptunaClassifier
from Scripts_python.Models.PLS.PLS_opti import AutoPLSRegression
from Scripts_python.Models.PLS.PLS_opti_classif import AutoPLSDAClassifier
from Scripts_python.Models.Ridge.Ridge_opti import RidgeCVRegressor
from Scripts_python.Models.Ridge.Ridge_opti_classif import RidgeCVClassifier
from Scripts_python.Models.LGBM.LGBM_optuna import LGBMOptuna
from Scripts_python.Models.LGBM.LGBM_optuna_classif import LGBMOptunaClassifier

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error, accuracy_score, f1_score, confusion_matrix
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from itertools import combinations
from joblib import Parallel, delayed, Memory
memory = Memory(".cache", verbose=0)  # Cache for parallel processing
from tqdm import tqdm

# import warnings filter
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cross_decomposition._pls")

from warnings import simplefilter, filterwarnings
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="Association modèle / preprocessing avec heatmap")

# Regression: 'BeerOriginalExtract' or 'Digest_0.8' or 'YamProtein' //
# Classification: 'CoffeeSpecies' or 'Malaria2024' or 'mDigest_custom3' or 'WhiskyConcentration' or 'YamMould'
parser.add_argument('--mode', type=str, choices=["Regression", "Classification"], required=True,
                    help="Type of task: 'Regression' or 'Classification'")

parser.add_argument('--data_source', type=str, required=True,
                    help="Name of the dataset to use (e.g., 'BeerOriginalExtract', 'CoffeeSpecies', etc.)")

parser.add_argument('--top_n_preprocs', type=int, default=None,
                    help="Display only the top N preprocessings based on their performance (optional). If None, all preprocessings are displayed.")

parser.add_argument('--model_names', nargs='+', type=str, default=None,
                    help="Perform the association pp/model with the models specified in the list of model names (optional). If None, all models are used (Ridge, PLS, LGBM, NICON).")

parser.add_argument('--progressive_optim', action='store_true', default=False,
                    help="Activate progressive optimization : first combination with a deep hyperparameter search, the next ones using the best_trials. If False, all researches are the same.")

parser.add_argument('--compare_optim_strat', action='store_true', default=False,
                    help="Compare optimization strategies of a single model on the same heatmap (optional). Required model_names with only one model name.")

parser.add_argument('--only_colors', action='store_true', default=False,
                    help="Display only colors in the heatmap without values (optional)")

parser.add_argument('--random_seed', type=int, default=42,
                    help="Global random seed (default: 42)")

parser.add_argument('--use_parallelism', action='store_true', default=False,
                    help="Use parallelization during the assessment of preprocessing-model combinations (optional)")

# Retrieve the arg values from the parser
args = parser.parse_args()
mode = args.mode
data_source = args.data_source
print(f"[INFO] Running on the dataset named {data_source}.")
only_colors = args.only_colors
use_parallelism = args.use_parallelism
# Set the seed for a reproductible script
rd_seed = args.random_seed
np.random.seed(rd_seed)
random.seed(rd_seed)
tf.random.set_seed(rd_seed)
# Keep only the top N preprocessings if specified
top_n = args.top_n_preprocs
model_names = args.model_names
progressive_optim = args.progressive_optim
print(f"[INFO] {'Progressive' if progressive_optim else 'Uniform'} optimization activated.")
compare_optim_strat = args.compare_optim_strat

import torch
torch.manual_seed(rd_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()
# Set the calibration and test data sets
Xcal, Ycal, Xval, Yval = split_data(mode, data_source, verbose=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Number of classes in the target variable
num_classes = len(np.unique(Ycal)) # Useful only in the case of Classification
if mode == "Classification": print("Number of classes detected : ", num_classes)

# Apply MinMax scaling to Y if regression mode
scaler_Y = None
if mode == 'Regression':
    scaler_Y = MinMaxScaler()
    Ycal = scaler_Y.fit_transform(np.array(Ycal).reshape(-1, 1)).ravel()
    Yval = scaler_Y.transform(np.array(Yval).reshape(-1, 1)).ravel()

# List of basic preprocessings
simple_preprocs = [
    ('id', pp.IdentityTransformer()),
    ('baseline', pp.Baseline()),
    ('derivate', pp.Derivate()),
    ('detrend', pp.Detrend()),
    ('MSC', pp.MultiplicativeScatterCorrection()),
    ('normalize', pp.Normalize()),
    ('RNV', pp.RobustNormalVariate()),
    ('savgol', pp.SavitzkyGolay()),
    ('simplescale', pp.SimpleScale()),
    ('SNV', pp.StandardNormalVariate()),
    ('haar', pp.Wavelet('haar')),
    ('gaussian', pp.Gaussian(order=2, sigma=1)),
]
preprocessings = list(simple_preprocs)

# Add 2-combinations of simple preprocessing methods
for (name1, trans1), (name2, trans2) in combinations(simple_preprocs[1:], 2): # we do not take "id" into account
    combo_name = f'{name1}_{name2}'
    combo_pipeline = Pipeline([
        (name1, trans1),
        (name2, trans2)
    ])
    preprocessings.append((combo_name, combo_pipeline))

# Add PCA transformation
preprocessings.append(('PCA', PCA(random_state=rd_seed)))

# Parameters related to the progressive optimization of NICON
if progressive_optim:
    n_trials_first = 500 if mode=="Regression" else 100
    n_trials_next = 90 if mode=="Regression" else 20
    epochs_first = 30
    epochs_next = 10
else:
    n_trials_uniform = 90
    epochs_uniform = 10
epochs, patience = 10000, 1000

# Create a dictionary storing each model
dict_models = {
    "Ridge_reg": RidgeCVRegressor(alphas=np.logspace(-4, 2, 50), cv=5, random_state=rd_seed),
    "PLS_reg": AutoPLSRegression(cv=3, seed=rd_seed),
    "LGBM_reg": LGBMOptuna(cv=5, n_trials=20, random_state=rd_seed, verbose=1, verbose_optuna=False),
    "NICON_reg": NiconOptunaRegressor(n_trials=90, epochs=epochs, patience=patience, cyclic_learning=True, lr_min=1e-6, lr_max=1e-3, epochs_optuna=10, random_state=rd_seed, device=device, verbose_optuna=True),
    "Ridge_classif": RidgeCVClassifier(alphas=np.logspace(-4, 2, 50), cv=5, random_state=rd_seed),
    "PLS_classif": AutoPLSDAClassifier(cv=5, seed=rd_seed),
    "LGBM_classif": LGBMOptunaClassifier(cv=5, n_trials=50, random_state=rd_seed, verbose=0),
    "NICON_classif": NiconOptunaClassifier(num_classes=num_classes, n_trials=50, epochs=10000, patience=10, epochs_optuna=100, random_state=rd_seed),
}

# Define models
suffixe = "_reg" if mode=="Regression" else "_classif"
if model_names is None:
    models = [
        ("Ridge" + suffixe, dict_models["Ridge" + suffixe]),
        ("PLS" + suffixe, dict_models["PLS" + suffixe]),
        ("LGBM" + suffixe, dict_models["LGBM" + suffixe]),
        ("NICON" + suffixe, dict_models["NICON" + suffixe]),
    ]
else:
    models = [
        (name_model + suffixe, dict_models[name_model + suffixe]) for name_model in model_names
    ]

# Dictionnary to store the optimal n_components of the PLS, per preprocessing
prior_components = []

# Initialize the best_trials variables of the NICON and LGBM models
best_trials_nicon = None
best_trials_lgbm = None

# Fonction modifiée pour collecter le nombre de composantes optimales
#@memory.cache
def evaluate_combination(pp_name, pp_method, mdl_name, mdl, mode, Xcal, Ycal, Xval, Yval, mean_metric, progressive_optim, best_trials, rd_seed, scaler_Y=None):

    combo_start = time.time()  # Start timer for this model-preproc pair

    np.random.seed(rd_seed)
    random.seed(rd_seed)
    tf.random.set_seed(rd_seed)

    X_train, X_test = np.asarray(Xcal), np.asarray(Xval)
    Y_train, Y_test = np.asarray(Ycal).ravel(), np.asarray(Yval).ravel()

    try:
        # Decide whether it is a big dataset or not, so to adjust the model parameters
        big_dataset = Xcal.shape[0] > 1e3

        if mdl_name.startswith('PLS'):
            n_wavelengths = Xcal.shape[1]
            total_evals = max(100, n_wavelengths // 10)
            parallelism = big_dataset
            
            # Hybrid candidate selection
            max_evals = total_evals // 5 if (len(prior_components) > 0 and progressive_optim) else total_evals
            candidates = None
            if progressive_optim:
                candidates = get_pls_component_candidates(
                    n_spectra = Xcal.shape[0],
                    n_wavelengths=n_wavelengths,
                    prior_components=prior_components,
                    max_evals=max_evals,
                    cv=5,
                    big_dataset=big_dataset,
                    rd_seed=rd_seed
                )
            
            if mode == "Regression":
                # Define the PLS model with adapted hyperparameters
                mdl = AutoPLSRegression(
                    cv=5,
                    scale=True,
                    seed=rd_seed,
                    candidate_components=candidates
                )
            else:
                if big_dataset: print("Big dataset: optimal number of components constrained between 1 and 20.")
                else: print("Number of trials tested for PLSDA : ", len(candidates) if candidates is not None else max_evals)
                # Define the PLS-DA model with adapted hyperparameters
                mdl = AutoPLSDAClassifier(
                    cv=5,
                    scale=True,
                    seed=rd_seed,
                    candidate_components=candidates,
                    parallelism=parallelism
                )
                # Correct class unbalances before applying PLS-DA
                #X_train, Y_train = correct_class_unbalances(X_train, Y_train, type_correction="duplicate", random_state=rd_seed)
        
        elif mdl_name.startswith("LGBM"):
            cv = 3 if big_dataset else 5
            subsampling_rate = 0.3 if big_dataset else None

            if not progressive_optim:
                best_trials = None
            else:
                print("Number of best trials used to optimize the LGBM model : ", len(best_trials) if best_trials is not None else 0)

            if best_trials is None:  # first optimization = large search
                n_trials = 100
            else:  # reduced search space from best_trials
                n_trials = 20

            if mode == "Regression":
                mdl = LGBMOptuna(cv=cv, n_trials=n_trials, random_state=rd_seed,
                                verbose=0, verbose_optuna=True, scoring="neg_mean_squared_error",
                                best_trials=best_trials, name_pp=pp_name, subsampling_rate=subsampling_rate)
            else:
                if big_dataset: print(f"[INFO] Big dataset: 3-Fold CV activated & {int(subsampling_rate*100)}% data subsampling during Optuna phase")
                mdl = LGBMOptunaClassifier(cv=cv, n_trials=n_trials, random_state=rd_seed,
                                        verbose=0, verbose_optuna=True, scoring="log_loss",
                                        best_trials=best_trials, name_pp=pp_name, subsampling_rate=subsampling_rate)

        elif mdl_name.startswith('NICON'):
            if not progressive_optim:
                best_trials = None
            else:
                print("Number of best trials used to optimize the NICON model : ", len(best_trials) if best_trials is not None else 0)

            if best_trials is None: # if this is the first optimization, it must be wide and deep
                if progressive_optim:
                    n_trials = n_trials_first
                    epochs_optuna = epochs_first
                else:
                    n_trials = n_trials_uniform
                    epochs_optuna = epochs_uniform
            else: # if we can use the previous results to reduce the optimization space
                n_trials = n_trials_next
                epochs_optuna = epochs_next

            if mode == "Regression":
                mdl = NiconOptunaRegressor(n_trials=n_trials, epochs=epochs, patience=patience, cyclic_learning=True, lr_min=1e-6, lr_max=1e-3, 
                                           epochs_optuna=epochs_optuna, random_state=rd_seed, device=device, verbose_optuna=True, 
                                           best_trials=best_trials, name_pp=pp_name)
            else:
                mdl = NiconOptunaClassifier(num_classes=num_classes, n_trials=n_trials, epochs=epochs, patience=patience, epochs_optuna=epochs_optuna, 
                                            cyclic_learning=True, lr_min=1e-6, lr_max=1e-3, parallelize=False, random_state=rd_seed, 
                                            verbose_optuna=True, device=device, best_trials=best_trials, name_pp=pp_name)

        if mdl_name.startswith("LGBM"):
            pipe = Pipeline([
                ("prep", pp_method),
                ("ensure_df", EnsureDataFrame()),
                ("model", clone(mdl))
            ], memory=memory)
        else:
            pipe = Pipeline([
                ("prep", pp_method),
                ("model", clone(mdl))
            ], memory=memory)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipe.fit(X_train, Y_train)

        y_pred = pipe.predict(X_test)

        if mode == 'Regression':
            # Inverse transform the predictions and targets to compute RMSE in original scale
            Y_true_original = scaler_Y.inverse_transform(Y_test.reshape(-1, 1)).ravel()
            Y_pred_original = scaler_Y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            metric = root_mean_squared_error(Y_true_original, Y_pred_original)
        else:
            metric = accuracy_score(Y_test, y_pred.ravel())
            metric_f1 = f1_score(Y_test, y_pred.ravel(), average='weighted')
            # Compute FPR
            cm = confusion_matrix(Y_test, y_pred.ravel(), labels=np.unique(Y_test))
            # For multi-class: take macro-average FPR
            fp = cm.sum(axis=0) - np.diag(cm)
            tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
            fpr = np.mean(fp / (fp + tn))

        if mode == "Regression" and abs(metric) > 1e2 * mean_metric:
            metric = np.nan
        
        elif mode== "Classification" and metric < 0.4 * mean_metric:
            metric = np.nan

        trained_model = pipe.named_steps["model"]

        # if the model is PLS, store the best number of components
        if mdl_name.startswith("PLS") and hasattr(trained_model, 'best_n_component_'):
            optimal_comp = trained_model.best_n_component_
            if pp_name not in prior_components:
                prior_components.append(optimal_comp)
                print("Prior components list updated : ", prior_components)
        
        # if the model is LGBM, store the optimal hyperparameters
        elif mdl_name.startswith("LGBM") and hasattr(trained_model, 'best_trials'):
            best_trials = trained_model.best_trials

        # if the model is NICON, store the optimal hyperparameters
        elif mdl_name.startswith('NICON') and hasattr(trained_model, 'best_trials'):
            best_trials = trained_model.best_trials
        
        # Store timing and performances
        combo_time = time.time() - combo_start  # Time for this combination
        if mode=="Regression":
            print(f"{pp_name} + {mdl_name} = {metric:.2f} (Time: {combo_time:.2f}s)")
            return (pp_name, mdl_name, metric, best_trials, combo_time)
        else: # mode="Classification"
            print(f"{pp_name} + {mdl_name} = ACC={metric:.2f} / F1={metric_f1:.2f} / FPR={fpr:.2f} (Time: {combo_time:.2f}s)")
            return (pp_name, mdl_name, metric, metric_f1, fpr, best_trials, combo_time)

    except Exception as e:
        metric = np.nan
        if mode == "Classification":
            metric_f1 = np.nan
            fpr = np.nan
        print(f"[ERROR] {pp_name} + {mdl_name}: {e}")
    

# Construction of the combinations (pp, model)
combinations = []
for (pp_name, pp_method) in preprocessings:
    for (mdl_name, mdl) in models:
        combinations.append((pp_name, pp_method, mdl_name, mdl))

# Initial approximation of the mean score (to filter the extreme outliers in the heatmap)
mean_metric = 1.0 if mode == 'Regression' else 0.5

if use_parallelism:
    print("[INFO] Execution of combinations in parallel mode (using joblib).")
    raw_results = Parallel(n_jobs=-1)(
        delayed(evaluate_combination)(
            pp_name, pp_method, mdl_name, mdl, mode, Xcal, Ycal, Xval, Yval, mean_metric,
            progressive_optim, best_trials_nicon if mdl_name.startswith("NICON") else best_trials_lgbm if mdl_name.startswith("LGBM") else None,
            rd_seed, scaler_Y
        )
        for (pp_name, pp_method, mdl_name, mdl) in tqdm(combinations, desc="Evaluations")
    )
    if mode == "Regression": 
        results = [(pp, mdl_name, score) for pp, mdl_name, score, _, _ in raw_results]
    else:
        results = [(pp, mdl_name, acc) for pp, mdl_name, acc, _, _, _, _ in raw_results]
        results_f1 = [(pp, mdl_name, f1) for pp, mdl_name, _, f1, _, _, _ in raw_results]
        results_fpr = [(pp, mdl_name, fpr) for pp, mdl_name, _, _, fpr, _, _ in raw_results]
    timings = [(mdl_name, combo_time) for _, mdl_name, _, _, _, combo_time in raw_results]

    # Update the right best_trials list
    for _, mdl, _, trials, _ in raw_results:
        if mdl.startswith("NICON") and trials is not None:
            best_trials_nicon = trials
        elif mdl.startswith("LGBM") and trials is not None:
            best_trials_lgbm = trials

else:
    print("[INFO] Execution of combinations in sequential mode.")
    results = []
    results_f1 = [] if mode == "Classification" else None
    results_fpr = [] if mode == "Classification" else None
    timings = []
    for (pp_name, pp_method, mdl_name, mdl) in tqdm(combinations, desc=f"Evaluations (sequential) {data_source}"):
        try:
            if mode == "Regression":
                pp_name, mdl_name, metric, trials, combo_time = evaluate_combination(
                    pp_name, pp_method, mdl_name, mdl, mode, Xcal, Ycal, Xval, Yval,
                    mean_metric, progressive_optim, best_trials_nicon if mdl_name.startswith("NICON") else best_trials_lgbm if mdl_name.startswith("LGBM") else None,
                    rd_seed, scaler_Y
                )
            else:
                pp_name, mdl_name, metric, f1, fpr, trials, combo_time = evaluate_combination(
                    pp_name, pp_method, mdl_name, mdl, mode, Xcal, Ycal, Xval, Yval,
                    mean_metric, progressive_optim, best_trials_nicon if mdl_name.startswith("NICON") else best_trials_lgbm if mdl_name.startswith("LGBM") else None,
                    rd_seed, scaler_Y
                )

            results.append((pp_name, mdl_name, metric))
            if mode == "Classification":
                results_f1.append((pp_name, mdl_name, f1))
                results_fpr.append((pp_name, mdl_name, fpr))
            timings.append((mdl_name, combo_time))

            # Update the right best_trials list
            if mdl_name.startswith("NICON") and trials is not None:
                best_trials_nicon = trials
            elif mdl_name.startswith("LGBM") and trials is not None:
                best_trials_lgbm = trials

        except Exception as e:
            print(f"[ERROR] {pp_name} + {mdl_name} : {e}")
            results.append((pp_name, mdl_name, np.nan))
            timings.append((mdl_name, np.nan))
            if mode == "Classification":
                results_f1.append((pp_name, mdl_name, np.nan))
                results_fpr.append((pp_name, mdl_name, np.nan))

# store the results in a DataFrame
df_scores = pd.DataFrame(results, columns=["Preprocessing", "Model", "Score"])
pivoted = df_scores.pivot(index="Model", columns="Preprocessing", values="Score")  # preprocessing as columns

if mode == "Classification":
    ### F1 score results
    df_scores_f1 = pd.DataFrame(results_f1, columns=["Preprocessing", "Model", "Score"])
    pivoted_f1 = df_scores_f1.pivot(index="Model", columns="Preprocessing", values="Score")
    ### FPR results
    df_scores_fpr = pd.DataFrame(results_fpr, columns=["Preprocessing", "Model", "Score"])
    pivoted_fpr = df_scores_fpr.pivot(index="Model", columns="Preprocessing", values="Score")

# save the results to a CSV file
output_dir = os.path.join("Results", "assoc_pp_model", data_source)
os.makedirs(output_dir, exist_ok=True)
optim_type = "progressive" if progressive_optim else "uniform"
if model_names is not None:
    names = "_".join(model_names)
    if "NICON" in names:
        name_file = f"results_{data_source}_{optim_type}_optim_{epochs}_epc_{patience}_ptc_{names}.csv"
    else:
        name_file = f"results_{data_source}_{optim_type}_optim_{names}.csv"
else:
    name_file = f"results_{data_source}_{optim_type}_optim_{epochs}_epc_{patience}_ptc.csv"
output_path = os.path.join(output_dir, name_file)
pivoted.to_csv(output_path)

if mode == "Classification":
    pivoted_f1.to_csv(os.path.join(output_dir, name_file.replace("results", "F1_results")))
    pivoted_fpr.to_csv(os.path.join(output_dir, name_file.replace("results", "FPR_results")))

# ──────────────────────────────────────────────────────
# Format functions for heatmap annotation and bolding best

def format_value(x, classification=False):
    if pd.isnull(x):
        return ""
    return f"{(x * 100 if classification else x):.2f}"

def bold_best(df, classification=False):
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

if top_n is not None:
    if mode == 'Regression':
        best_preprocs = pivoted.mean().sort_values().head(top_n).index
    else:  # Classification
        best_preprocs = pivoted.mean().sort_values(ascending=False).head(top_n).index

    pivoted = pivoted[best_preprocs]
    formatted_df = formatted_df[best_preprocs]

# ──────────────────────────────────────────────────────
# Plotting the heatmap
num_preprocs = len(pivoted.columns)
fig_width = max(14, num_preprocs * 0.25)

from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(fig_width, 5.5 if mode == "Classification" else 5))

# Draw the base heatmap
heatmap = sns.heatmap(
    pivoted * (100 if mode == "Classification" else 1),
    annot=None if only_colors else formatted_df,
    fmt="" if not only_colors else None,
    linewidths=0.5,
    cmap="YlGnBu" if mode == "Classification" else "viridis",
    cbar_kws={"label": "Accuracy (%)" if mode == "Classification" else "RMSE"},
    xticklabels=True,
    yticklabels=True,
    ax=ax
)

# Square the best value of each column (preprocessing)
for j, col in enumerate(pivoted.columns):
    values = pivoted[col]
    if values.isnull().all():
        continue

    # Find the best score of the column
    if mode == "Classification":
        best_val = values.max()
    else:
        best_val = values.min()

    best_indices = values[values == best_val].index

    for idx in best_indices:
        i = list(pivoted.index).index(idx)
        # Rectangle(xy, width, height), xy = bottom left
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))

# Finalize the display
plt.xticks(rotation=90, ha="center", fontsize=8 if only_colors else 7)
plt.yticks(rotation=0, fontsize=10 if only_colors else 9)
plt.title(f"Performance Heatmap ({'Accuracy' if mode == 'Classification' else 'RMSE'}) / {data_source} dataset ({mode})")
plt.tight_layout()
output_dir = os.path.join("Figures", "assoc_pp_model", data_source)
os.makedirs(output_dir, exist_ok=True)

# Name of the heatmap file
optim_type = "progressive" if progressive_optim else "uniform"
if model_names is not None:
    names = "_".join(model_names)
    if top_n is not None:
        if "NICON" in names:
            heatmap_filename = f"heatmap_{data_source}_top_{top_n}_{epochs}epochs_{patience}patience_{optim_type}_optim_{names}.png"
        else:
            heatmap_filename = f"heatmap_{data_source}_top_{top_n}_{optim_type}_optim_{names}.png"
    else:
        if "NICON" in names:
            heatmap_filename = f"heatmap_{data_source}_{epochs}epochs_{patience}patience_{optim_type}_optim_{names}.png"
        else:
            heatmap_filename = f"heatmap_{data_source}_{optim_type}_optim_{names}.png"
else:
    if top_n is not None:
        heatmap_filename = f"heatmap_{data_source}_top_{top_n}_{epochs}epochs_{patience}patience_{optim_type}_optim.png"
    else:
        heatmap_filename = f"heatmap_{data_source}_{epochs}epochs_{patience}patience_{optim_type}_optim.png"

output_path = os.path.join(output_dir, heatmap_filename)
plt.savefig(output_path, dpi=300)

# --------------------------------------------------------------
# Generate F1 heatmap in the case of classification
if mode == "Classification":
    num_preprocs = len(pivoted_f1.columns)
    fig_width = max(14, num_preprocs * 0.25)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    # Draw the base heatmap
    heatmap = sns.heatmap(
        pivoted_f1 * 100,
        annot=None if only_colors else formatted_df,
        fmt="" if not only_colors else None,
        linewidths=0.5,
        cmap="YlGnBu",
        cbar_kws={"label": "F1-score (%)"},
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    # Square the best value of each column (preprocessing)
    for j, col in enumerate(pivoted_f1.columns):
        values = pivoted_f1[col]
        if values.isnull().all():
            continue

        # Find the best score for this column
        best_val = values.max()

        best_indices = values[values == best_val].index

        for idx in best_indices:
            i = list(pivoted_f1.index).index(idx)
            # Rectangle(xy, width, height), xy = bottom left
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    # Finalize the display
    plt.xticks(rotation=90, ha="center", fontsize=8 if only_colors else 7)
    plt.yticks(rotation=0, fontsize=10 if only_colors else 9)
    plt.title(f"F1-score Heatmap - {data_source}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, heatmap_filename.replace("heatmap", "F1_heatmap")), dpi=300)

# -------------------------------------------------------------------------------
# In the case of Classification, generate FPR heatmap
if mode == "Classification":
    df_scores_fpr = pd.DataFrame(results_fpr, columns=["Preprocessing", "Model", "Score"])
    pivoted_fpr = df_scores_fpr.pivot(index="Model", columns="Preprocessing", values="Score")

    # Apply top_n filtering if needed (best = minimal FPR)
    if top_n is not None:
        best_preprocs_fpr = pivoted_fpr.mean().sort_values().head(top_n).index
        pivoted_fpr = pivoted_fpr[best_preprocs_fpr]

    fig, ax = plt.subplots(figsize=(max(14, len(pivoted_fpr.columns) * 0.25), 5.5))
    # Annotate best (lowest) value in each column
    formatted_fpr = bold_best(pivoted_fpr, classification=False)  # False = lower is better
    heatmap = sns.heatmap(
        pivoted_fpr * 100,  # percentage
        annot=None if only_colors else formatted_fpr,
        fmt="" if not only_colors else None,
        linewidths=0.5,
        cmap="YlGnBu_r",  # Inverted Reds (low = white, high = dark red)
        cbar_kws={"label": "False Positive Rate (%)"},
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )
    for j, col in enumerate(pivoted_fpr.columns):
        values = pivoted_fpr[col]
        if values.isnull().all():
            continue
        best_val = values.min()  # Best = smallest
        best_indices = values[values == best_val].index
        for idx in best_indices:
            i = list(pivoted_fpr.index).index(idx)
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))
    plt.xticks(rotation=90, ha="center", fontsize=8 if only_colors else 7)
    plt.yticks(rotation=0, fontsize=10 if only_colors else 9)
    plt.title(f"False Positive Rate Heatmap - {data_source}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, heatmap_filename.replace("heatmap", "FPR_heatmap")), dpi=300)

# ──────────────────────────────────────────────────────
# Save the execution time (if it exists) into a csv file

elapsed_time = time.time() - start_time

### Save the total execution time

timing_output_path = os.path.join("Results", "assoc_pp_model", data_source)
os.makedirs(timing_output_path, exist_ok=True)
optim_type = "progressive" if progressive_optim else "uniform"
if model_names is not None:
    names = "_".join(model_names)
    if "NICON" in names:
          file_name = f"timing_results_{epochs}epochs_{patience}patience_{optim_type}_optim_{names}.csv"
          timing_csv_path = os.path.join(timing_output_path, file_name)
    else:
        file_name = f"timing_results_{optim_type}_optim_{names}.csv"
        timing_csv_path = os.path.join(timing_output_path, file_name)
else:
    file_name = f"timing_results_{epochs}epochs_{patience}patience_{optim_type}_optim.csv"
    timing_csv_path = os.path.join(timing_output_path, file_name)

timing_data = {
    "data_source": data_source,
    "epochs_final": epochs,
    "patience": patience,
    "optimization_type": optim_type,
    "time": elapsed_time
    }
if progressive_optim:
    timing_data["n_trials_first"] = n_trials_first
    timing_data["n_trials_next"] = n_trials_next
    timing_data["epochs_first"] = epochs_first
    timing_data["epochs_next"] = epochs_next
else:
    timing_data["n_trials"] = n_trials_uniform
    timing_data["epochs_optuna"] = epochs_uniform

# Ajout ou création du fichier CSV
if os.path.exists(timing_csv_path):
    df_time = pd.read_csv(timing_csv_path)
    df_time = pd.concat([df_time, pd.DataFrame([timing_data])], ignore_index=True)
else:
    df_time = pd.DataFrame([timing_data])

df_time.to_csv(timing_csv_path, index=False)
print(f"[INFO] Execution time saved to {timing_csv_path}")

### Save the execution time per model

# Aggregate average execution time per model
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


### Save per-model timing CSV

file_name = file_name.replace("results", "per_model")
timing_models_path = os.path.join("Results", "assoc_pp_model", data_source, file_name)
if os.path.exists(timing_models_path):
    df_existing = pd.read_csv(timing_models_path)
    df_avg_time = pd.concat([df_existing, df_avg_time], ignore_index=True)

df_avg_time.to_csv(timing_models_path, index=False)
print(f"[INFO] Per-model execution times saved to {timing_models_path}")

# ──────────────────────────────────────────────────────
# Save the variable best_trials of the NICON model (if it exists) into a JSON file
if best_trials_nicon is not None and progressive_optim:
    trials_path = os.path.join(output_dir, f"best_trials_NICON_{data_source}.json")
    try:
        with open(trials_path, "w") as f:
            json.dump(make_json_serializable(best_trials_nicon), f, indent=2)
        print(f"[INFO]NICON's best_trials saved to path : {trials_path}")
    except Exception as e:
        print(f"[WARNING] Error while saving NICON's best_trials : {e}")

# ──────────────────────────────────────────────────────
# Save the variable best_trials of the NICON model (if it exists) into a JSON file
if best_trials_lgbm is not None and progressive_optim:
    trials_path = os.path.join(output_dir, f"best_trials_LGBM_{data_source}.json")
    try:
        with open(trials_path, "w") as f:
            json.dump(make_json_serializable(best_trials_lgbm), f, indent=2)
        print(f"[INFO]LGBM's best_trials saved to path : {trials_path}")
    except Exception as e:
        print(f"[WARNING] Error while saving LGBM's best_trials : {e}")