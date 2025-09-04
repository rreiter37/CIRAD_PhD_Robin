import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Path to the global results folder
base_path = "Results/assoc_pp_model"

# Datasets inside the folder
datasets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Define which datasets are regression
regression_datasets = ["BeerOriginalExtract", "Digest_0.8", "YamProtein"]

for dataset_name in datasets:
    results_path = os.path.join(base_path, dataset_name)

    # Collect all CSV files starting with 'results', excluding unwanted models
    csv_files = [
        f for f in glob.glob(os.path.join(results_path, "results*.csv"))
        if not any(excl in f for excl in ["PLS", "NICON", "Ridge", "LGBM"])
    ]

    if not csv_files:
        continue  # Skip datasets without valid files

    # Concatenate all CSVs into one dataframe
    df_list = [pd.read_csv(f) for f in csv_files]
    data = pd.concat(df_list, ignore_index=True)

    # List of models in the dataset
    models = data["Model"].unique()

    # Preprocessing columns (exclude 'Model' and 'id')
    pp_columns = [c for c in data.columns if c not in ["Model", "id"]]

    # Classification or regression
    task_type = "Regression" if dataset_name in regression_datasets else "Classification"

    # Create subplots (one per model)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)
    axes = axes.flatten()

    for ax, model in zip(axes, models):
        # Filter rows of the current model
        df_model = data[data["Model"] == model]

        # Reshape into long format
        df_melted = df_model.melt(id_vars="Model", value_vars=pp_columns,
                                  var_name="Preprocessing", value_name="Performance")

        # Boxplot with matplotlib
        df_melted.boxplot(column="Performance", by="Preprocessing", ax=ax, rot=90)

        ax.set_title(f"Model: {model}")
        ax.set_xlabel("Preprocessing")
        ax.set_ylabel("Performance")

    # Adjust layout and add global title
    plt.suptitle(f"{task_type} - {dataset_name}: Performance distributions per model and preprocessing")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
output_dir = "Figures/assoc_pp_model/All_datasets"
file_name = "boxplots_perf_pp_per_model.png"
output_path = os.path.join(output_dir, file_name)
plt.savefig(output_path, dpi=300)
print(f"[INFO] Saving the boxplots to: {output_path}")