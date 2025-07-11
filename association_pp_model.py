
import os
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import argparse

import nirs4all.transformations as pp
from nirs4all.presets.ref_models import nicon

from nicon_optuna import NiconOptunaRegressor
from PLS_opti import AutoPLSRegression
from Ridge_optuna import RidgeOptuna

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import RidgeCV, LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from dissimilarity_functions import *
import json

import lightgbm as lgb

# import warnings filter
import warnings
from warnings import simplefilter, filterwarnings
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
filterwarnings("ignore", category=FutureWarning)


# Function to load a CSV file with automatic separator detection

def load_csv_auto_sep(mode, data_source, type_data, verbose=True, delimiter=None):

    ## Importation of the datasets with the adapted path
    file_name = Path("Data/%s/%s"% (mode,data_source))
    full_path = str(file_name.resolve()).replace("\\", "/")
    path = full_path + "/%s.csv" % type_data
    
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:

        if delimiter is not None:
            sep = delimiter
        
        else:
            # Read a small portion of the file to detect the separator
            excerpt = f.read(1024)
            f.seek(0)  # return to the beginning of the file

            # Detection of the dialect
            dialect = csv.Sniffer().sniff(excerpt)
            sep = dialect.delimiter

        if verbose: print("Detected separator for %s: %s" % (type_data, sep))
        
        # Load the file with pandas
        df = pd.read_csv(f, delimiter=sep)

        if type_data[0]=='Y' and len(df.columns) > 1:
            # Drop the useless column if it exists
            df = df.drop(columns=[df.columns[1]])
        
        return df
    

rd_seed = 42  # Set a random seed for reproducibility

# Regression: 'BeerOriginalExtract' or 'Digest_0.8' or 'YamProtein' //
# Classification: 'CoffeeSpecies' or 'Malaria2024' or 'mDigest_custom3' or 'WhiskyConcentration' or 'YamMould'

parser = argparse.ArgumentParser(description="Association modèle / preprocessing avec heatmap")

parser.add_argument('--mode', type=str, choices=["Regression", "Classification"], required=True,
                    help="Type de tâche : Regression ou Classification")

parser.add_argument('--data_source', type=str, required=True,
                    help="Nom du dataset (ex: YamMould, CoffeeSpecies, etc.)")

args = parser.parse_args()

mode = args.mode
data_source = args.data_source

Xcal = load_csv_auto_sep(mode=mode, data_source=data_source, type_data='Xcal')
Xval = load_csv_auto_sep(mode=mode, data_source=data_source, type_data='Xval')
Ycal = load_csv_auto_sep(mode=mode, data_source=data_source, type_data='Ycal', delimiter=' ')
Yval = load_csv_auto_sep(mode=mode, data_source=data_source, type_data='Yval', delimiter=' ')

print("Number of spectra for calibration: ", len(Ycal))
print("Number of spectra for test: ", len(Yval))

# Define preprocessing methods 
preprocessings = [   ('id', pp.IdentityTransformer()),
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
                    ('gaussian', pp.Gaussian(order = 2, sigma = 1)),
                    ('PCA', PCA())
                ]

# Define models 
models = [
    ("Ridge_opt", RidgeOptuna(n_trials=10, random_state=rd_seed)),
    ("PLS", AutoPLSRegression(max_components=50, cv=5)),
    ("LGBM", lgb.LGBMRegressor(n_estimators=100, random_state=rd_seed, verbose=-1)),
    ("NICON", NiconOptunaRegressor(n_trials=10, epochs=100, patience=10, random_state=rd_seed)),
]


# Calibration and test part
results = []
mean_metric = 0
for (pp_name, pp_method) in preprocessings:
    for (mdl_name, mdl) in models:
        pipe = Pipeline([
            ("prep", pp_method),
            ("model", mdl)
        ])
        try:
            pipe.fit(Xcal, Ycal)
            metric = root_mean_squared_error(Yval, pipe.predict(Xval))
            n = len(results)
            if n > 1 and abs(metric) > 1e3 * mean_metric:
                metric = np.nan  # Handle extreme values
        except Exception as e:
            metric = np.nan
            print(f"[ERROR] {pp_name} + {mdl_name}: {e}")

        results.append((pp_name, mdl_name, metric))
        mean_metric = 1/(n+1) * (mean_metric * n + metric) if metric != np.nan else mean_metric
        print(f"{pp_name} + {mdl_name} → {metric:.2f}")

# store the results in a DataFrame
df_scores = pd.DataFrame(results, columns=["Preprocessing", "Model", "Score"])
pivoted = df_scores.pivot(index="Model", columns="Preprocessing", values="Score") # pivot the DataFrame to have preprocessing methods as columns and models as rows

# save the results to a CSV file
output_dir = os.path.join("Results", "assoc_pp_model", data_source)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"results_{data_source}.csv")
pivoted.to_csv(output_path)

# Format the ouput for better visualization
def format_value(x):
    if pd.isnull(x):
        return ""
    return f"{x:.2f}"

# Bold best score per column (i.e. per preprocessing)
def bold_best(df):
    formatted = df.copy()
    for col in df.columns:
        col_values = df[col]
        if col_values.isnull().all():
            continue
        best_idx = col_values.idxmin()
        for idx in df.index:
            val = df.at[idx, col]
            if pd.isnull(val):
                formatted.at[idx, col] = ""
            elif idx == best_idx:
                formatted.at[idx, col] = r"$\bf{" + format_value(val) + "}$"
            else:
                formatted.at[idx, col] = format_value(val)
    return formatted

formatted_df = bold_best(pivoted)

# ────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
sns.heatmap(
    pivoted,  # raw scores (used for coloring)
    annot=formatted_df,  # formatted text overlay
    fmt="",  # already formatted
    linewidths=0.5,
    cmap="viridis",
    cbar_kws={"label": "RMSE"},
)
plt.title(f"Performance Heatmap (RMSE) / {data_source} dataset ({mode})")
plt.tight_layout()
output_dir = os.path.join("Figures", "assoc_pp_model", data_source)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"heatmap_{data_source}.png")
plt.savefig(output_path, dpi=300)
plt.show()