import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Configuration
# ==============================

TABPFN_RESULTS_CSV = "Results/tabpfn/tabpfn_results.csv"
ASSOC_RESULTS_CSV = "Results/association/assoc_results.csv"

DATA_DIR = "Data/Regression"
OUTDIR = "Results/comp_assoc_tabpfn"
os.makedirs(OUTDIR, exist_ok=True)

# ==============================
# Utility functions
# ==============================

def compute_rrmse(rmse, y_std):
    """
    Convert RMSE to normalized RRMSE.

    Parameters
    ----------
    rmse : float
        Root Mean Squared Error.
    y_std : float
        Standard deviation of the reference target values.

    Returns
    -------
    float
        Normalized RRMSE.
    """
    return rmse / y_std


def load_target_std(dataset_name):
    """
    Load target values and compute their standard deviation.
    Assumes y values are stored in ycal.csv and yval.csv.

    Parameters
    ----------
    dataset_name : str

    Returns
    -------
    float
    """
    y_vals = []

    for split in ["ycal.csv", "yval.csv"]:
        path = os.path.join(DATA_DIR, dataset_name, split)
        if os.path.exists(path):
            y = pd.read_csv(path, sep=";", header=None).values.ravel()
            y_vals.append(y)

    if not y_vals:
        raise FileNotFoundError(f"No ycal/yval found for dataset {dataset_name}")

    y_all = np.concatenate(y_vals)
    return np.std(y_all, ddof=1)


# ==============================
# Load results
# ==============================

df_tabpfn = pd.read_csv(TABPFN_RESULTS_CSV)
df_assoc = pd.read_csv(ASSOC_RESULTS_CSV)

# Expected columns:
# df_tabpfn: dataset, RMSE
# df_assoc : dataset, model, RRMSE_norm

# ==============================
# Normalize TabPFN metrics
# ==============================

rrmse_tabpfn = []

for _, row in df_tabpfn.iterrows():
    dataset = row["dataset"]
    rmse = row["RMSE"]

    y_std = load_target_std(dataset)
    rrmse = compute_rrmse(rmse, y_std)

    rrmse_tabpfn.append({
        "dataset": dataset,
        "model": "TabPFN",
        "RRMSE_norm": rrmse
    })

df_tabpfn_rrmse = pd.DataFrame(rrmse_tabpfn)

# ==============================
# Merge all models
# ==============================

df_all = pd.concat([df_assoc, df_tabpfn_rrmse], ignore_index=True)

# ==============================
# Pivot table (models x datasets)
# ==============================

heatmap_df = df_all.pivot_table(
    index="model",
    columns="dataset",
    values="RRMSE_norm",
    aggfunc="mean"
)

heatmap_df.to_csv(os.path.join(OUTDIR, "rrmse_table.csv"))

# ==============================
# Heatmap visualization
# ==============================

plt.figure(figsize=(0.6 * heatmap_df.shape[1] + 4,
                    0.6 * heatmap_df.shape[0] + 2))

sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    cbar_kws={"label": "Normalized RRMSE"}
)

plt.title("Model comparison (Normalized RRMSE)")
plt.xlabel("Dataset")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "heatmap_rrmse.png"), dpi=300)
plt.close()

# ==============================
# Colored table: best / worst
# ==============================

def color_best_worst(val, col):
    if pd.isna(val):
        return ""
    if val == col.min():
        return "background-color: #a8e6a3"  # green
    if val == col.max():
        return "background-color: #f4a6a6"  # red
    return ""

styled = heatmap_df.style.apply(
    lambda x: [color_best_worst(v, x) for v in x],
    axis=0
)

styled.to_excel(os.path.join(OUTDIR, "rrmse_colored_table.xlsx"))

print(f"Results saved to {OUTDIR}")
