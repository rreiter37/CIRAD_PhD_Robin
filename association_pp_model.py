
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '42'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Clear the cache
import shutil
shutil.rmtree(".cache")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import tensorflow as tf
import json

import argparse

import nirs4all.transformations as pp
from nirs4all.presets.ref_models import nicon

from ensure_dataframe import EnsureDataFrame
from utils_bdd import split_data

from nicon_optuna import NiconOptunaRegressor
from nicon_optuna_classif import NiconOptunaClassifier
from PLS_opti import AutoPLSRegression
from PLS_opti_classif import AutoPLSClassifier
from Ridge_opti import RidgeCVRegressor
from Ridge_opti_classif import RidgeCVClassifier
from LGBM_optuna import LGBMOptuna
from LGBM_optuna_classif import LGBMOptunaClassifier

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.base import clone

import torch

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

    

# Regression: 'BeerOriginalExtract' or 'Digest_0.8' or 'YamProtein' //
# Classification: 'CoffeeSpecies' or 'Malaria2024' or 'mDigest_custom3' or 'WhiskyConcentration' or 'YamMould'

parser = argparse.ArgumentParser(description="Association modèle / preprocessing avec heatmap")

parser.add_argument('--mode', type=str, choices=["Regression", "Classification"], required=True,
                    help="Type of task: 'Regression' or 'Classification'")

parser.add_argument('--data_source', type=str, required=True,
                    help="Name of the dataset to use (e.g., 'BeerOriginalExtract', 'CoffeeSpecies', etc.)")

parser.add_argument('--top_n_preprocs', type=int, default=None,
                    help="Display only the top N preprocessings based on their performance (optional). If None, all preprocessings are displayed.")

parser.add_argument('--only_colors', action='store_true', default=False,
                    help="Display only colors in the heatmap without values (optional)")

parser.add_argument('--random_seed', type=int, default=42,
                    help="Global random seed (default: 42)")


# Ajout dans argparse
parser.add_argument('--use_parallelism', action='store_true', default=False,
                    help="Utilise la parallélisation pour évaluer les combinaisons (optionnel)")

# Retrieve the arg values from the parser
args = parser.parse_args()
mode = args.mode
data_source = args.data_source
only_colors = args.only_colors
use_parallelism = args.use_parallelism
# Set the seed for a reproductible script
rd_seed = args.random_seed
np.random.seed(rd_seed)
random.seed(rd_seed)
tf.random.set_seed(rd_seed)

# Set the calibration and test data sets
Xcal, Ycal, Xval, Yval = split_data(mode, data_source, verbose=True)

# List of basic preprocessings
simple_preprocs = [
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
for (name1, trans1), (name2, trans2) in combinations(simple_preprocs, 2):
    combo_name = f'{name1}_{name2}'
    combo_pipeline = Pipeline([
        (name1, trans1),
        (name2, trans2)
    ])
    preprocessings.append((combo_name, combo_pipeline))

# Add identity and PCA models
preprocessings.append(('id', pp.IdentityTransformer()))
preprocessings.append(('PCA', PCA()))


# Define models 
if mode == 'Regression':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    models = [
        ("Ridge_opt", RidgeCVRegressor(alphas=np.logspace(-4, 2, 50), cv=5, random_state=rd_seed)),
        ('PLS', AutoPLSRegression(max_components=Xcal.shape[1], cv=3, scale=True, seed=rd_seed, max_evals=60)),
        ("LGBM_opt", LGBMOptuna(cv=5, n_trials=20, random_state=rd_seed, verbose=1, verbose_optuna=False)),
        ('NICON', NiconOptunaRegressor(n_trials=50, epochs=5000, patience=10, epochs_optuna=20, random_state=rd_seed, device=device)),
    ]
    
else:  # Classification
    num_classes = len(np.unique(Ycal))  # Number of classes in the target variable
    models = [
        ("Ridge_classif", RidgeCVClassifier(alphas=np.logspace(-4, 2, 50), cv=5, random_state=rd_seed)),
        ("PLS_classif", AutoPLSClassifier(max_components=Xcal.shape[1], cv=5)),
        ("LGBM_classif", LGBMOptunaClassifier(cv=5, n_trials=50, random_state=rd_seed, verbose=0)),
        ("NICON_classif", NiconOptunaClassifier(num_classes=num_classes, n_trials=50, epochs=10000, patience=10, epochs_optuna=100, random_state=rd_seed)),
    ]


# Dictionnaire pour stocker les n_composantes optimaux par prétraitement
prior_components = []

# Fonction modifiée pour collecter le nombre de composantes optimales
#@memory.cache
def evaluate_combination(pp_name, pp_method, mdl_name, mdl, mode, Xcal, Ycal, Xval, Yval, mean_metric, rd_seed):

    np.random.seed(rd_seed)
    random.seed(rd_seed)
    tf.random.set_seed(rd_seed)

    X_train, X_test = np.asarray(Xcal), np.asarray(Xval)
    Y_train, Y_test = np.asarray(Ycal).ravel(), np.asarray(Yval).ravel()

    try:
        if 'PLS' in mdl_name:
            if len(prior_components) > 0:
                median = int(np.median(prior_components))
                lower = max(1, median - 10)
                upper = min(Xcal.shape[1], median + 10)
                max_evals = 10
            else:
                lower = 1
                upper = Xcal.shape[1]
                max_evals = 100
            
            mdl = AutoPLSRegression(max_components=Xcal.shape[1], cv=5, scale=True, seed=rd_seed,
                                    max_evals=max_evals, component_range=(lower, upper))

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
            metric = root_mean_squared_error(Y_test, y_pred)
        else:
            metric = accuracy_score(Y_test, y_pred.ravel())

        if abs(metric) > 1e3 * mean_metric:
            metric = np.nan

        # Si c'est un modèle PLS, extraire le nombre de composantes
        if mdl_name == 'PLS' and hasattr(mdl, 'best_n_components_'):
            optimal_comp = mdl.best_n_components_
            if pp_name not in prior_components:
                prior_components.append(optimal_comp)

    except Exception as e:
        metric = np.nan
        print(f"[ERROR] {pp_name} + {mdl_name}: {e}")

    print(f"{pp_name} + {mdl_name} = {metric:.2f}")

    return (pp_name, mdl_name, metric)


# Construction of the combinations (pp, model)
combinations = []
for (pp_name, pp_method) in preprocessings:
    for (mdl_name, mdl) in models:
        combinations.append((pp_name, pp_method, mdl_name, mdl))

# Initial approximation of the mean score (to filter the extreme outliers in the heatmap)
mean_metric = 1.0 if mode == 'Regression' else 0.5

if use_parallelism:
    print("[INFO] Execution in parallel mode (using joblib).")
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_combination)(
            pp_name, pp_method, mdl_name, mdl, mode, Xcal, Ycal, Xval, Yval, mean_metric, rd_seed
        )
        for (pp_name, pp_method, mdl_name, mdl) in tqdm(combinations, desc="Evaluations")
    )
else:
    print("[INFO] Execution in sequential mode.")
    results = []
    for (pp_name, pp_method, mdl_name, mdl) in tqdm(combinations, desc="Evaluations (sequential)"):
        res = evaluate_combination(pp_name, pp_method, mdl_name, mdl, mode, Xcal, Ycal, Xval, Yval, mean_metric, rd_seed)
        results.append(res)

# store the results in a DataFrame
df_scores = pd.DataFrame(results, columns=["Preprocessing", "Model", "Score"])
pivoted = df_scores.pivot(index="Model", columns="Preprocessing", values="Score")  # preprocessing as columns

# save the results to a CSV file
output_dir = os.path.join("Results", "assoc_pp_model", data_source)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"results_{data_source}.csv")
pivoted.to_csv(output_path)

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

# Keep only the top N preprocessings if specified
top_n = args.top_n_preprocs

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

plt.figure(figsize=(fig_width, 5.5 if mode == "Classification" else 5))

sns.heatmap(
    pivoted * (100 if mode == "Classification" else 1),  # scale for accuracy
    annot=None if only_colors else formatted_df,
    fmt="" if not only_colors else None,
    linewidths=0.5,
    cmap="YlGnBu" if mode == "Classification" else "viridis",
    cbar_kws={"label": "Accuracy (%)" if mode == "Classification" else "RMSE"},
    xticklabels=True,
    yticklabels=True
)


plt.xticks(rotation=90, ha="center", fontsize=8 if only_colors else 7)
plt.yticks(rotation=0, fontsize=10 if only_colors else 9)
plt.title(f"Performance Heatmap ({'Accuracy' if mode == 'Classification' else 'RMSE'}) / {data_source} dataset ({mode})")

plt.tight_layout()
output_dir = os.path.join("Figures", "assoc_pp_model", data_source)
os.makedirs(output_dir, exist_ok=True)

# Nom du fichier heatmap avec ou sans mention top_N
if top_n is not None:
    heatmap_filename = f"heatmap_{data_source}_top_{top_n}.png"
else:
    heatmap_filename = f"heatmap_{data_source}.png"

output_path = os.path.join(output_dir, heatmap_filename)
plt.savefig(output_path, dpi=300)
plt.show()

output_dir = os.path.join("Results", "assoc_pp_model", data_source)
os.makedirs(output_dir, exist_ok=True)
components_output_path = os.path.join(output_dir, f"PLS_optimal_components_{data_source}.json")
with open(components_output_path, 'w') as f:
    json.dump(prior_components, f, indent=2)

print(f"\nOptimal PLS components saved to: {components_output_path}")