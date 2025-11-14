import os
import pandas as pd
import matplotlib.pyplot as plt

# Root directory where all dataset subfolders are located
root_dir = "Results/assoc_pp_model/per_dataset"
fig_dir = "Figures/assoc_pp_model/per_dataset"

# Lists to accumulate results across datasets
global_results = []

# Walk through each dataset folder
for dataset_name in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset_name)
    fig_path = os.path.join(fig_dir, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    os.makedirs(fig_path, exist_ok=True)

    # Identify the files of interest
    files = os.listdir(dataset_path)

    # 1️⃣ Classic results (non-adaptive CNN/NICON)
    classic_files = [
        f for f in files
        if f.startswith("results_")
        and not any(exclude in f for exclude in ["PLS", "Ridge", "LGBM", "NICON", "CNN_adaptive", "dynamic"])
        and f.endswith(".csv")
    ]

    # 2️⃣ Adaptive batch size results
    adaptive_files = [
        f for f in files
        if f.startswith("results_") and "adaptive" in f and f.endswith(".csv") and not "dynamic" in f
    ]

    # 3️⃣ Dynamic batch size results (nouvelle approche)
    dynamic_files = [
        f for f in files
        if f.startswith("results_") and "dynamic" in f and f.endswith(".csv")
    ]

    # Vérification : au moins 2 fichiers requis pour comparaison
    if not classic_files or not adaptive_files or not dynamic_files:
        print(f"Skipping {dataset_name}, one of the required files is missing.")
        continue

    # Load CSVs
    classic_file = os.path.join(dataset_path, classic_files[0])
    adaptive_file = os.path.join(dataset_path, adaptive_files[0])
    dynamic_file = os.path.join(dataset_path, dynamic_files[0])

    df_classic = pd.read_csv(classic_file)
    df_adaptive = pd.read_csv(adaptive_file)
    df_dynamic = pd.read_csv(dynamic_file)

    # Extract CNN/NICON rows for classic
    classic_row = df_classic[df_classic.iloc[:, 0].str.contains("CNN|NICON", case=False, na=False)]
    if classic_row.empty:
        print(f"No CNN/NICON row found in {classic_file}, skipping.")
        continue

    adaptive_row = df_adaptive.iloc[0]
    dynamic_row = df_dynamic.iloc[0]

    preprocessings = df_classic.columns[1:]

    # Accumulate results
    for pp in preprocessings:
        global_results.append({
            "Dataset": dataset_name,
            "Preprocessing": pp,
            "Classic": classic_row[pp].values[0],
            "Adaptive": adaptive_row[pp],
            "Dynamic": dynamic_row[pp]
        })

    # Per-dataset plot
    df_plot = pd.DataFrame(global_results)
    df_plot = df_plot[df_plot["Dataset"] == dataset_name]

    n_pp = len(df_plot["Preprocessing"].unique())
    fig_width = max(10, n_pp * 0.6)

    plt.figure(figsize=(fig_width, 6))
    plt.plot(df_plot["Preprocessing"], df_plot["Classic"], marker="o", label="Classic CNN/NICON")
    plt.plot(df_plot["Preprocessing"], df_plot["Adaptive"], marker="s", label="Adaptive CNN")
    plt.plot(df_plot["Preprocessing"], df_plot["Dynamic"], marker="^", label="Dynamic CNN")

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("RRMSE (lower is better)", fontsize=12)
    plt.title(f"RRMSE Comparison - {dataset_name}", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"RRMSE_comparison_{dataset_name}.png"))
    plt.close()

# Global comparison
if global_results:
    df_global = pd.DataFrame(global_results)

    # Global scatterplots comparing each pair
    pairs = [
        ("Classic", "Adaptive"),
        ("Classic", "Dynamic"),
        ("Adaptive", "Dynamic"),
    ]

    for x_col, y_col in pairs:
        plt.figure(figsize=(7, 7))
        plt.scatter(df_global[x_col], df_global[y_col], alpha=0.7)
        min_val = min(df_global[[x_col, y_col]].min())
        max_val = max(df_global[[x_col, y_col]].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
        plt.xlabel(f"{x_col} RRMSE")
        plt.ylabel(f"{y_col} RRMSE")
        plt.title(f"Global RRMSE Comparison: {x_col} vs {y_col}")
        plt.legend()
        plt.tight_layout()

        out_dir = os.path.join("Figures", "assoc_pp_model", "All_datasets", y_col.lower())
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, f"Global_RRMSE_scatter_{x_col}_vs_{y_col}.png")
        plt.savefig(fig_path)
        plt.close()

    # mean RRMSE across datasets for global comparison
    mean_df = df_global.groupby("Dataset")[["Classic", "Adaptive", "Dynamic"]].mean().mean()
    plt.figure(figsize=(6, 5))
    mean_df.plot(kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylabel("Mean RRMSE (lower is better)", fontsize=12)
    plt.title("Average Performance Across All Datasets", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join("Figures", "assoc_pp_model", "All_datasets", "adaptive_batch_size", "mean_RRMSE_barplot.png")
    plt.savefig(out_path)
    plt.close()

    print("[INFO] Global comparisons and barplot saved successfully.")
else:
    print("No valid results found for comparison.")