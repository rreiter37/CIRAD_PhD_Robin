#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
Visual comparison of selected (model, preprocessing) pairs
across three adaptive pipelines.

Adds:
------
1. Model × Preprocessing heatmap colored by number of
   pipelines that keep the couple.
   (0 = red, 1 = orange, 2 = yellow, 3 = green)

2. Optional argument --task_only {regression, classification}
   to filter rows based on dataset_name.
===========================================================
"""

# ============================================================
# Imports
# ============================================================
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
from upsetplot import from_contents, UpSet
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Argument parser
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_only",
    type=str,
    default=None,
    choices=["regression", "classification"],
    help="If specified, keep only datasets of the selected task type."
)
args = parser.parse_args()

# ============================================================
# Configuration
# ============================================================
BASE_DIR = "Results/assoc_pp_model/All_datasets"
PIPELINES = {
    "Gatekeeping": "Pipeline_gatekeeping/pipeline_gatekeeping_selected.csv",
    "Graph": "Pipeline_graph/pipeline_graph_pruning_selected.csv",
    "Weakness coverage": "Pipeline_weakness_coverage/pipeline_weakness_coverage_selected.csv"
}

OUTPUT_DIR = os.path.join(BASE_DIR.replace("Results", "Figures"), "Pipeline_comparison_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Utility functions
# ============================================================
def filter_task(df):
    """
    Filter dataframe based on the --task_only argument.
    Converts dataset_name to string to avoid .str errors.
    """
    # If no filtering requested → return unchanged
    if args.task_only is None:
        return df

    # If dataset_name column missing → do nothing
    if 'dataset_name' not in df.columns:
        return df

    # Convert all dataset_name entries to string to avoid .str.contains errors
    df['dataset_name'] = df['dataset_name'].astype(str)

    # Build mask according to task type
    if args.task_only == "regression":
        mask = df['dataset_name'].str.contains("_reg", case=False, na=False)
    else:
        mask = df['dataset_name'].str.contains("_classif", case=False, na=False)

    return df[mask]


def normalize_model_name(model):
    """
    Normalize model names so that variants are merged.
    Example:
        - 'NICON_reg' and 'CNN_reg' become 'NICON'
    Extend this mapping if needed.
    """
    m = str(model).lower()

    # Merge NICON_reg and CNN_reg into a single model name
    if "nicon" in m or "cnn" in m:
        return "NICON"

    # Default: return unchanged
    return model

def load_pipeline_csv(name, path):
    """
    Load CSV file for one pipeline and create a 'couple' identifier.
    Also applies task filtering.
    """
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Missing file for {name}: {full_path}")

    df = pd.read_csv(full_path)

    # Apply regression/classification filtering
    df = filter_task(df)

    # Ensure necessary columns exist
    if not {'model_name', 'preprocessing_name'}.issubset(df.columns):
        raise ValueError(f"{name} CSV must contain 'model_name' and 'preprocessing_name' columns")

    # Normalize model names before building couples
    df['model_name'] = df['model_name'].apply(normalize_model_name)
    df['couple'] = df['model_name'] + " + " + df['preprocessing_name']

    # Ensure metric_mean exists
    if 'metric_mean' not in df.columns:
        df['metric_mean'] = np.nan

    return df

# ============================================================
# Load pipelines
# ============================================================
data = {name: load_pipeline_csv(name, path) for name, path in PIPELINES.items()}
sets = {name: set(df['couple']) for name, df in data.items()}
all_couples = sorted(list(set.union(*sets.values())))
names = list(sets.keys())

print(f"Loaded {sum(len(s) for s in sets.values())} total selected pairs across pipelines.")
if args.task_only:
    print(f"Task filtered to: {args.task_only}")

# ============================================================
# 1. Venn diagram
# ============================================================
plt.figure(figsize=(7, 6))
venn3(
    [sets["Gatekeeping"], sets["Graph"], sets["Weakness coverage"]],
    set_labels=('Gatekeeping', 'Graph', 'Weakness coverage')
)
plt.title("Overlap of selected (model, preprocessing) pairs")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "venn_overlap.png"), dpi=300)
plt.close()

# ============================================================
# 2. UpSet plot
# ============================================================
contents = {name: sets[name] for name in names}
upset = UpSet(from_contents(contents), show_counts=True)
upset.plot()
plt.title("Intersections of selected pairs across pipelines")
plt.savefig(os.path.join(OUTPUT_DIR, "upset_intersections.png"), dpi=300)
plt.close()

# ============================================================
# 3. Jaccard similarity heatmap
# ============================================================
matrix = np.zeros((len(names), len(names)))
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        inter = len(sets[n1] & sets[n2])
        union = len(sets[n1] | sets[n2])
        matrix[i, j] = inter / union if union > 0 else 0

plt.figure(figsize=(5, 4))
sns.heatmap(matrix, annot=True, cmap="crest", xticklabels=names, yticklabels=names, fmt=".2f")
plt.title("Jaccard similarity between pipeline selections")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "jaccard_similarity.png"), dpi=300)
plt.close()

# ============================================================
# 4. Binary selection heatmap (pipeline × couple)
# ============================================================
df_matrix = pd.DataFrame(index=all_couples)
for name in names:
    df_matrix[name] = df_matrix.index.isin(sets[name])

plt.figure(figsize=(12, 4))
sns.heatmap(df_matrix.T, cmap="Greens", cbar=False)
plt.title("Binary selection map of (model, preprocessing) pairs")
plt.xlabel("Couples")
plt.ylabel("Pipelines")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "binary_selection_map.png"), dpi=300)
plt.close()

# ============================================================
# 5. Scatter plot performance vs selection frequency
# ============================================================
merged = pd.concat([
    df[['couple', 'metric_mean']].assign(pipeline=name)
    for name, df in data.items()
], ignore_index=True)

perf_df = merged.groupby('couple', as_index=False).agg({'metric_mean': 'mean'})
perf_df['selection_rate'] = perf_df['couple'].apply(
    lambda x: sum(x in s for s in sets.values()) / len(sets)
)

plt.figure(figsize=(7, 5))
sns.scatterplot(
    perf_df, x='metric_mean', y='selection_rate',
    hue='selection_rate', palette='viridis', s=70
)
plt.title("Performance vs selection frequency across pipelines")
plt.xlabel("Mean performance (metric_mean)")
plt.ylabel("Selection frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_performance_selection.png"), dpi=300)
plt.close()

# ============================================================
# 6. NEW: Heatmap model × preprocessing colored by number of pipelines
# ============================================================
# --- Extract model and preprocessing names ---
models = sorted(set(c.split(" + ")[0] for c in all_couples))
preps = sorted(set(c.split(" + ")[1] for c in all_couples))

# --- Count number of pipelines keeping each couple ---
count_map = pd.DataFrame(0, index=models, columns=preps)

for couple in all_couples:
    model, prep = couple.split(" + ")
    n = sum(couple in s for s in sets.values())
    count_map.at[model, prep] = n

# --- Custom colormap for 0/1/2/3 discrete categories ---
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["red", "orange", "yellow", "green"])

plt.figure(figsize=(14, 6))
sns.heatmap(
    count_map,
    cmap=cmap,
    annot=True,
    fmt="d",
    cbar=False
)
plt.title("Model × Preprocessing — Number of pipelines that keep the couple")
plt.xlabel("Preprocessing")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_preprocessing_heatmap.png"), dpi=300)
plt.close()

# ============================================================
# Summary
# ============================================================
print(f"Visualizations saved in: {OUTPUT_DIR}")
print("Generated files:")
for f in os.listdir(OUTPUT_DIR):
    print("  -", f) 