import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Comparison of optimization strategies on the results of models")

# Regression: 'BeerOriginalExtract' or 'Digest_0.8' or 'YamProtein' //
# Classification: 'CoffeeSpecies' or 'Malaria2024' or 'mDigest_custom3' or 'WhiskyConcentration' or 'YamMould'
parser.add_argument('--mode', type=str, choices=["Regression", "Classification"], required=True,
                    help="Type of task: 'Regression' or 'Classification'")

parser.add_argument('--data_source', type=str, required=True,
                    help="Name of the dataset to use (e.g., 'BeerOriginalExtract', 'CoffeeSpecies', etc.)")

# Retrieve the arg values from the parser
args = parser.parse_args()
mode = args.mode
data_source = args.data_source

# Folder containing all files to compare
results_dir = f"Results/assoc_pp_model/{data_source}"

# List of files to compare
files_to_compare = [
    f for f in os.listdir(results_dir)
    if f.endswith(".csv") and f.startswith("results_")
]

# Metric used to evaluate the performances of a model
metric = "RMSE" if mode=='Regression' else "Accuracy"

# Store flattened data
all_scores = []

for fname in files_to_compare:
    fpath = os.path.join(results_dir, fname)
    if not os.path.exists(fpath):
        print(f"[WARN] File not found : {fpath}")
        continue
    
    df = pd.read_csv(fpath, index_col=0)  # each file = heatmap of scores
    values = df.values.flatten()
    
    # Remove eventual NaNs
    values = values[~pd.isna(values)]
    
    for v in values:
        all_scores.append({
            metric: v,
            "file": fname.replace("results_", "").replace(".csv", "").replace(f"{data_source}_", "").replace('optim_', "")
        })
# Create a long dataframe for seaborn
df_long = pd.DataFrame(all_scores)

# Display
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_long, x="file", y=metric, palette="Set3", hue="file", legend=False)
plt.title("Distribution of the scores per optimization strategy")
plt.ylabel(metric)
plt.xlabel("Optim Strategy")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.tight_layout()

# Calculate the median per file
medians = df_long.groupby("file")[metric].median()

# Display the median value on each boxplot
ax = plt.gca()
for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
    label_text = label.get_text()
    if label_text in medians:
        median_val = medians[label_text]
        ax.text(tick, median_val, f"median={median_val:.2f}",
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=8,
                color='black')

fig_path = f"Figures/assoc_pp_model/{data_source}/"
plt.savefig(fig_path + "boxplot_comparison_heatmaps.png", dpi=300)