import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Base directory containing per-dataset subfolders with results*.csv files
base_path = "Results/assoc_pp_model"

# Which dataset names are regression (others will be treated as classification)
regression_datasets = ["BeerOriginalExtract", "Digest_0.8", "YamProtein"]

# Output directory for figures
output_dir = "Figures/assoc_pp_model/All_datasets"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------
# Collect melted (long) tables for regression and classification
# ------------------------------------------------------------------
regression_melted = []
classification_melted = []

# List dataset subfolders
datasets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

for dataset_name in datasets:
    results_path = os.path.join(base_path, dataset_name)

    # Collect CSV files (exclude filenames containing certain substrings)
    csv_files = [
        f for f in glob.glob(os.path.join(results_path, "results*.csv"))
        if not any(excl in f for excl in ["PLS", "NICON", "Ridge", "LGBM"])
    ]

    if not csv_files:
        print(f"[INFO] No matching CSVs in dataset folder: {dataset_name}")
        continue

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[WARN] Could not read {fpath}: {e}")
            continue

        if "Model" not in df.columns:
            print(f"[WARN] 'Model' column not found in {fpath} — skipping file.")
            continue

        pp_columns = [c for c in df.columns if c not in ("Model", "id")]
        if not pp_columns:
            print(f"[WARN] No preprocessing columns found in {fpath} — skipping file.")
            continue

        df_melted = df.melt(id_vars=["Model"], value_vars=pp_columns,
                            var_name="Preprocessing", value_name="Performance")

        df_melted["Performance"] = pd.to_numeric(df_melted["Performance"], errors="coerce")
        df_melted = df_melted.dropna(subset=["Performance"])

        if df_melted.empty:
            print(f"[INFO] After coercion no numeric performance in {fpath} — skipping.")
            continue

        df_melted["Dataset"] = dataset_name
        df_melted["SourceFile"] = os.path.basename(fpath)

        if dataset_name in regression_datasets:
            regression_melted.append(df_melted)
        else:
            classification_melted.append(df_melted)

# ------------------------------------------------------------------
# Helper to plot heatmaps for up to 4 models
# ------------------------------------------------------------------
def plot_models_heatmap(melted_list, y_label, suptitle, out_filename, minimize=True):
    """
    melted_list: list of dataframes with ['Model','Preprocessing','Performance',...]
    y_label: str for colorbar label (e.g., 'RMSE' or 'Accuracy')
    suptitle: main title for the figure
    out_filename: filename (basename) to save under output_dir
    minimize: True if lower values are better (regression), False otherwise
    """
    if not melted_list:
        print(f"[INFO] No data to plot for {suptitle}")
        return

    df_all = pd.concat(melted_list, ignore_index=True)

    # Deterministic selection of up to 4 models
    models = sorted(df_all["Model"].unique())
    if not models:
        print(f"[INFO] No models found for {suptitle}")
        return
    models_to_plot = models[:4]

    # Compute mean performance per (Model, Preprocessing)
    df_mean = (
        df_all[df_all["Model"].isin(models_to_plot)]
        .groupby(["Model", "Preprocessing"])["Performance"]
        .mean()
        .reset_index()
    )

    # Remove extreme outliers (beyond 10 std in the "bad" direction)
    mu = df_mean["Performance"].mean()
    sigma = df_mean["Performance"].std()

    if minimize:
        mask_outliers = df_mean["Performance"] > mu + 10 * sigma
    else:
        mask_outliers = df_mean["Performance"] < mu - 10 * sigma

    df_mean.loc[mask_outliers, "Performance"] = np.nan

    # Pivot to wide format
    heatmap_data = df_mean.pivot(index="Model", columns="Preprocessing", values="Performance")
    heatmap_data = heatmap_data.reindex(models_to_plot)

    # Convert to numpy matrix
    data_matrix = heatmap_data.values
    fig, ax = plt.subplots(figsize=(18, 6))
    cmap = "viridis" if minimize else "YlGnBu"
    im = ax.imshow(data_matrix, aspect="auto", cmap=cmap)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(y_label)

    # Tick labels
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=90)
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    # Highlight best values (ignoring NaNs)
    for j, col in enumerate(heatmap_data.columns):
        col_values = heatmap_data[col].values
        if np.all(np.isnan(col_values)):
            continue
        if minimize:
            best_value = np.nanmin(col_values)
        else:
            best_value = np.nanmax(col_values)

        for i, val in enumerate(col_values):
            if np.isfinite(val) and np.isclose(val, best_value, rtol=1e-6, atol=1e-8):
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     fill=False, edgecolor="red", linewidth=2)
                ax.add_patch(rect)

    ax.set_title(suptitle, fontsize=14, pad=20)
    plt.tight_layout()

    out_path = os.path.join(output_dir, out_filename)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved heatmap to: {out_path}")

# ------------------------------------------------------------------
# Plot regression (RMSE → minimize)
# ------------------------------------------------------------------
plot_models_heatmap(
    regression_melted,
    y_label="RMSE",
    suptitle="Regression: Mean RMSE per model and preprocessing",
    out_filename="heatmap_regression_all_datasets.png",
    minimize=True
)

# ------------------------------------------------------------------
# Plot classification (Accuracy → maximize)
# ------------------------------------------------------------------
plot_models_heatmap(
    classification_melted,
    y_label="Accuracy",
    suptitle="Classification: Mean Accuracy per model and preprocessing",
    out_filename="heatmap_classification_all_datasets.png",
    minimize=False
)