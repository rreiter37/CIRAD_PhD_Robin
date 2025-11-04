import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Scripts_python.utils.utils_bdd import split_data

# Root folder containing the datasets
root_folder = "Results/assoc_pp_model/per_dataset"

# Dictionary to store results {model: {dataset_size: execution_time}}
results = {}

regression_datasets = ["BeerOriginalExtract", "Digest_0.8", "YamProtein", 
                       "ALPINE_C_424_KS", "ALPINE_N_552_KS", "ALPINE_P_291_KS",
                       "Biscuit_Fat_40_RandomSplit", "Biscuit_Flour_40_RandomSplit", "Biscuit_Sucrose_40_RandomSplit", "Biscuit_Water_40_RandomSplit",
                       "LUCAS_SOC_all_26650_NocitaKS", "Rice_Amylose_313_YbasedSplit",
                       ]

# Walk through all dataset folders
for dataset_name in os.listdir(root_folder):
    mode = "Regression" if dataset_name in regression_datasets else "Classification"
    Xcal, _, _, _ = split_data(mode, dataset_name, verbose=False)
    dataset_size = Xcal.shape[0]
    dataset_path = os.path.join(root_folder, dataset_name)
    if not os.path.isdir(dataset_path):
        continue
    
    # Look for CSV files starting with "timing_per_model"
    for file in os.listdir(dataset_path):
        if file.startswith("timing_per_model") and file.endswith(".csv"):
            file_path = os.path.join(dataset_path, file)
            
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Expecting columns: Model, Time_seconds
            for _, row in df.iterrows():
                model = row["Model"]
                time = row["Time_seconds"]
                
                if model not in results:
                    results[model] = {}
                results[model][dataset_size] = time

# Convert results to a DataFrame for easier plotting
plot_data = []
for model, sizes_times in results.items():
    for size, time in sizes_times.items():
        plot_data.append({"Model": model, "DatasetSize": size, "Time": time})

df_plot = pd.DataFrame(plot_data)

# Plot execution time vs dataset size for each model
plt.figure(figsize=(10, 6))
for model in df_plot["Model"].unique():
    sub_df = df_plot[df_plot["Model"] == model]
    sub_df = sub_df.sort_values("DatasetSize")
    plt.plot(sub_df["DatasetSize"], sub_df["Time"], marker="o", label=model)

plt.xlabel("Training dataset size")
plt.ylabel("Execution time (s)")
plt.title("Execution time per model vs dataset size")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
output_dir = "Figures/assoc_pp_model/All_datasets/Heatmap_exploitation"
file_name = "scatter_time_model_size.png"
output_path = os.path.join(output_dir, file_name)
plt.savefig(output_path, dpi=300)
print(f"[INFO] Saving the scatter plot to: {output_path}")







### Boxplots of execution time per model
plt.figure(figsize=(8, 6))
sns.boxplot(x="Model", y="Time", data=df_plot)

plt.title("Execution time per model")
plt.xlabel("Model")
plt.ylabel("Execution time (s)")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure
output_dir = "Figures/assoc_pp_model/All_datasets/Heatmap_exploitation"
file_name = "boxplots_time_model_size.png"
output_path = os.path.join(output_dir, file_name)
plt.savefig(output_path, dpi=300)
print(f"[INFO] Saving the boxplots to: {output_path}")