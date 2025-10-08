import os
import pandas as pd
import matplotlib.pyplot as plt

# Root directory where all dataset subfolders are located
root_dir = "Results/assoc_pp_model"
fig_dir = "Figures/assoc_pp_model"

# Lists to accumulate results across datasets
global_results = []

# Walk through each dataset folder
for dataset_name in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset_name)
    fig_path = os.path.join(fig_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    # Identify the files of interest
    files = os.listdir(dataset_path)
    
    # "Classic" results file: starts with "results_" and excludes specific keywords
    classic_files = [
        f for f in files
        if f.startswith("results_")
        and not any(exclude in f for exclude in ["PLS", "Ridge", "LGBM", "NICON", "CNN"])
        and f.endswith(".csv")
    ]
    
    # "Adaptive batch size" CNN results file
    adaptive_files = [
        f for f in files
        if f.startswith("results_") and f.endswith("CNN_adaptive_batch_size.csv")
    ]
    
    # Skip datasets where one of the files is missing
    if not classic_files or not adaptive_files:
        print(f"Skipping {dataset_name}, required files not found.")
        continue
    
    classic_file = os.path.join(dataset_path, classic_files[0])
    adaptive_file = os.path.join(dataset_path, adaptive_files[0])
    
    # Load CSVs
    df_classic = pd.read_csv(classic_file)
    df_adaptive = pd.read_csv(adaptive_file)
    
    # Extract the row for CNN or NICON in the "classic" file
    classic_row = df_classic[df_classic.iloc[:, 0].str.contains("CNN|NICON", case=False, na=False)]
    if classic_row.empty:
        print(f"No CNN/NICON row found in {classic_file}, skipping.")
        continue
    
    # Extract the adaptive row (only one line expected)
    adaptive_row = df_adaptive.iloc[0]
    
    # Now compare all preprocessing columns (exclude first column = model name)
    preprocessings = df_classic.columns[1:]
    
    for pp in preprocessings:
        classic_rrmse = classic_row[pp].values[0]
        adaptive_rrmse = adaptive_row[pp]
        
        # Store for global comparison
        global_results.append({
            "Dataset": dataset_name,
            "Preprocessing": pp,
            "Classic": classic_rrmse,
            "Adaptive": adaptive_rrmse
        })
    
    # Per-dataset visualization
    df_plot = pd.DataFrame(global_results)
    df_plot = df_plot[df_plot["Dataset"] == dataset_name]

    # Dynamic figure size depending on the number of preprocessings
    n_pp = len(df_plot["Preprocessing"].unique())
    fig_width = max(10, n_pp * 0.6)   # scale width with number of preprocessings

    plt.figure(figsize=(fig_width, 6))
    plt.plot(df_plot["Preprocessing"], df_plot["Classic"], marker="o", label="Classic CNN/NICON")
    plt.plot(df_plot["Preprocessing"], df_plot["Adaptive"], marker="s", label="Adaptive CNN")

    # Improve readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("RRMSE (lower is better)", fontsize=12)
    plt.title(f"RRMSE Comparison - {dataset_name}", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"RRMSE_comparison_{dataset_name}.png"))
    plt.close()

# Global comparison across datasets
if global_results:
    df_global = pd.DataFrame(global_results)
    
    # Scatterplot: Classic vs Adaptive
    plt.figure(figsize=(7, 7))
    plt.scatter(df_global["Classic"], df_global["Adaptive"], alpha=0.7)
    plt.plot([df_global["Classic"].min(), df_global["Classic"].max()],
             [df_global["Classic"].min(), df_global["Classic"].max()],
             "r--", label="y=x")
    
    plt.xlabel("Classic CNN/NICON RRMSE")
    plt.ylabel("Adaptive CNN RRMSE")
    plt.title("Global RRMSE Comparison Across Datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "All_datasets", "Global_RRMSE_scatter.png"))
    plt.close()
    
    print("Global comparison saved as Global_RRMSE_scatter.png")
else:
    print("No valid results found for comparison.")
