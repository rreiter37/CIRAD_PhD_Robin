import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Base directory containing per-dataset subfolders with results*.csv files
base_path = "Results/assoc_pp_model"

# Which dataset names are regression (others will be treated as classification)
regression_datasets = ["BeerOriginalExtract", "Digest_0.8", "YamProtein", 
                       "ALPINE_C_424_KS", "ALPINE_N_552_KS", "ALPINE_P_291_KS",
                       "Biscuit_Fat_40_RandomSplit", "Biscuit_Flour_40_RandomSplit", "Biscuit_Sucrose_40_RandomSplit", "Biscuit_Water_40_RandomSplit",
                       "LUCAS_SOC_all_26650_NocitaKS", "Rice_Amylose_313_YbasedSplit",
                       ]

# Output directory for figures
output_dir = "Figures/assoc_pp_model/All_datasets/Heatmap_exploitation"
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

    # Read each CSV and convert to long format (Model, Preprocessing, Performance)
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"[WARN] Could not read {fpath}: {e}")
            continue

        # Expect a 'Model' column and then one-or-more preprocessing columns holding metric values
        if "Model" not in df.columns:
            print(f"[WARN] 'Model' column not found in {fpath} — skipping file.")
            continue

        # Consider all columns except 'Model' and 'id' as preprocessing columns
        pp_columns = [c for c in df.columns if c not in ("Model", "id")]

        if not pp_columns:
            print(f"[WARN] No preprocessing columns found in {fpath} — skipping file.")
            continue

        # Melt to long format: Model | Preprocessing | Performance
        df_melted = df.melt(id_vars=["Model"], value_vars=pp_columns,
                            var_name="Preprocessing", value_name="Performance")

        # Convert performance values to numeric (coerce errors) and drop NaNs
        df_melted["Performance"] = pd.to_numeric(df_melted["Performance"], errors="coerce")
        df_melted = df_melted.dropna(subset=["Performance"])

        if df_melted.empty:
            print(f"[INFO] After coercion no numeric performance in {fpath} — skipping.")
            continue

        # Add metadata columns
        df_melted["Dataset"] = dataset_name
        df_melted["SourceFile"] = os.path.basename(fpath)

        # Append to appropriate collection
        if dataset_name in regression_datasets:
            regression_melted.append(df_melted)
        else:
            classification_melted.append(df_melted)

# ------------------------------------------------------------------
# Helper to plot up to 4 models in a 2x2 figure
# ------------------------------------------------------------------
def plot_models_boxplots(melted_list, y_label, suptitle, out_filename):
    """
    melted_list: list of dataframes with columns ['Model','Preprocessing','Performance',...]
    y_label: str for y-axis (e.g., 'RMSE' or 'Accuracy')
    suptitle: main title for the figure
    out_filename: filename (basename) to save under output_dir
    """
    if not melted_list:
        print(f"[INFO] No data to plot for {suptitle}")
        return

    df_all = pd.concat(melted_list, ignore_index=True)

    # Deterministic selection of up to 4 models (alphabetical order)
    models = sorted(df_all["Model"].unique())
    if not models:
        print(f"[INFO] No models found for {suptitle}")
        return
    models_to_plot = models[:4]

    # Prepare 2x2 axes
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for ax_idx in range(4):
        ax = axes[ax_idx]
        if ax_idx < len(models_to_plot):
            model = models_to_plot[ax_idx]
            df_model = df_all[df_all["Model"] == model]

            # If there are many preprocessing names, we plot one box per preprocessing
            # Grouped boxplots by Preprocessing
            try:
                df_model.boxplot(column="Performance", by="Preprocessing", ax=ax, rot=90)
            except Exception as e:
                # Fallback: single aggregated box if grouped boxplot fails
                print(f"[WARN] grouped boxplot failed for model {model}: {e} — plotting aggregated box")
                ax.boxplot(df_model["Performance"].values)
                ax.set_xticklabels([model], rotation=0)

            ax.set_title(f"Model: {model}")
            ax.set_xlabel("Preprocessing")
            ax.set_ylabel(y_label)
        else:
            # No model for this subplot — hide axis
            ax.set_visible(False)

    # Global title and layout
    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    out_path = os.path.join(output_dir, out_filename)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved figure to: {out_path}")

# ------------------------------------------------------------------
# Plot regression (RMSE)
# ------------------------------------------------------------------
plot_models_boxplots(
    regression_melted,
    y_label="RMSE",
    suptitle="Regression: RMSE distributions per model across all regression datasets",
    out_filename="boxplots_regression_rmse_per_model.png"
)

# ------------------------------------------------------------------
# Plot classification (Accuracy)
# ------------------------------------------------------------------
plot_models_boxplots(
    classification_melted,
    y_label="Accuracy",
    suptitle="Classification: Accuracy distributions per model across all classification datasets",
    out_filename="boxplots_classification_accuracy_per_model.png"
)