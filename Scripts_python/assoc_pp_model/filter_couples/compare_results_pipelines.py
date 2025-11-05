#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
Visual comparison of selected (model, preprocessing) pairs
across three adaptive pipelines:
  - Gatekeeping
  - Graph pruning
  - Weakness coverage

Directory structure:
--------------------
Results/assoc_pp_model/All_datasets/Pipeline_{name}/selected_pairs.csv
where {name} ∈ {"gatekeeping", "graph", "weakness_coverage"}

The script automatically loads each file, merges couples,
and produces multiple visualizations:
  1. Venn diagram
  2. UpSet plot (for large intersections)
  3. Jaccard similarity heatmap
  4. Binary selection heatmap
  5. Scatter plot: performance vs selection frequency
===========================================================
"""

# ============================================================
# Imports
# ============================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
from upsetplot import from_contents, UpSet
import warnings
warnings.filterwarnings("ignore")

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
# Utility: Load and normalize pipeline data
# ============================================================
def load_pipeline_csv(name, path):
    """Load CSV file for one pipeline and create a 'couple' identifier."""
    full_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Missing file for {name}: {full_path}")

    df = pd.read_csv(full_path)
    if not {'model_name', 'preprocessing_name'}.issubset(df.columns):
        raise ValueError(f"{name} CSV must contain 'model_name' and 'preprocessing_name' columns")

    df['couple'] = df['model_name'].astype(str) + ' + ' + df['preprocessing_name'].astype(str)
    if 'metric_mean' not in df.columns:
        df['metric_mean'] = np.nan  # Optional, for scatter plot
    return df

# ============================================================
# Load all pipelines
# ============================================================
data = {name: load_pipeline_csv(name, path) for name, path in PIPELINES.items()}

sets = {name: set(df['couple']) for name, df in data.items()}
all_couples = sorted(list(set.union(*sets.values())))

print(f"Loaded {sum(len(s) for s in sets.values())} total selected pairs across pipelines.")

# ============================================================
# 1. Venn Diagram
# ============================================================
plt.figure(figsize=(7,6))
venn3([sets["Gatekeeping"], sets["Graph"], sets["Weakness coverage"]],
      set_labels=('Gatekeeping', 'Graph', 'Weakness coverage'))
plt.title("Overlap of selected (model, preprocessing) pairs")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "venn_overlap.png"), dpi=300)
plt.close()

# ============================================================
# 2. UpSet plot (useful for large sets)
# ============================================================
contents = {
    "Gatekeeping": sets["Gatekeeping"],
    "Graph": sets["Graph"],
    "Weakness coverage": sets["Weakness coverage"]
}
upset = UpSet(from_contents(contents), show_counts=True)
upset.plot()
plt.title("Intersections of selected pairs across pipelines")
plt.savefig(os.path.join(OUTPUT_DIR, "upset_intersections.png"), dpi=300)
plt.close()

# ============================================================
# 3. Jaccard similarity heatmap
# ============================================================
names = list(sets.keys())
matrix = np.zeros((len(names), len(names)))

for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        inter = len(sets[n1] & sets[n2])
        union = len(sets[n1] | sets[n2])
        matrix[i, j] = inter / union if union > 0 else 0

plt.figure(figsize=(5,4))
sns.heatmap(matrix, annot=True, cmap="crest", xticklabels=names, yticklabels=names, fmt=".2f")
plt.title("Jaccard similarity between pipeline selections")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "jaccard_similarity.png"), dpi=300)
plt.close()

# ============================================================
# 4. Binary selection heatmap
# ============================================================
df_matrix = pd.DataFrame(index=all_couples)
for name in names:
    df_matrix[name] = df_matrix.index.isin(sets[name])

plt.figure(figsize=(12,4))
sns.heatmap(df_matrix.T, cmap="Greens", cbar=False)
plt.title("Binary selection map of (model, preprocessing) pairs")
plt.xlabel("Couples")
plt.ylabel("Pipelines")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "binary_selection_map.png"), dpi=300)
plt.close()

# ============================================================
# 5. Scatter plot: performance vs selection frequency
# ============================================================
# Combine all metric_mean values if available
merged = pd.concat([
    df[['couple', 'metric_mean']].assign(pipeline=name)
    for name, df in data.items()
], ignore_index=True)

perf_df = (
    merged.groupby('couple', as_index=False)
    .agg({'metric_mean': 'mean'})
)
perf_df['selection_rate'] = perf_df['couple'].apply(
    lambda x: sum(x in s for s in sets.values()) / len(sets)
)

plt.figure(figsize=(7,5))
sns.scatterplot(perf_df, x='metric_mean', y='selection_rate',
                hue='selection_rate', palette='viridis', s=70)
plt.title("Performance vs selection frequency across pipelines")
plt.xlabel("Mean performance (metric_mean)")
plt.ylabel("Selection frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_performance_selection.png"), dpi=300)
plt.close()

# ============================================================
# Summary
# ============================================================
print(f"Visualizations saved in: {OUTPUT_DIR}")
print("Generated files:")
for f in os.listdir(OUTPUT_DIR):
    print("  -", f)
