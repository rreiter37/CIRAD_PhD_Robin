import os
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------
# Import your split_data() function
# -----------------------------------------------------------
from your_module import split_data   # <-- replace with your actual module


# -----------------------------------------------------------
# Scan the directory structure and collect statistics
# -----------------------------------------------------------
def scan_nirs_datasets(base_path="Data"):
    """
    Scans the folder structure:
        Data/
            Regression/
                DB1/
            Classification/
                DB2/

    For each dataset, loads the CSV files using:
        Xcal, Ycal, Xval, Yval = split_data(mode, dataset_name, data_dir=base_path)

    Returns a DataFrame summarizing dataset statistics.
    """

    dataset_types = ["Regression", "Classification"]
    entries = []

    for dtype in dataset_types:
        dtype_path = os.path.join(base_path, dtype)
        mode = dtype.lower()   # expected by split_data()

        if not os.path.exists(dtype_path):
            print(f"Warning: folder {dtype_path} does not exist.")
            continue

        for db_name in os.listdir(dtype_path):
            db_path = os.path.join(dtype_path, db_name)
            if not os.path.isdir(db_path):
                continue

            print(f"\nLoading dataset: {db_name} ({dtype})")

            # -----------------------------------------------------------
            # Load dataset via the user’s function split_data()
            # -----------------------------------------------------------
            try:
                Xcal, Ycal, Xval, Yval = split_data(
                    mode=mode,
                    data_source=db_name,
                    data_dir=base_path,
                    verbose=False
                )
            except Exception as e:
                print(f"Error loading dataset {db_name}: {e}")
                continue

            # Count samples
            n_cal = len(Xcal) if Xcal is not None else 0
            n_val = len(Xval) if Xval is not None else 0
            n_total = n_cal + n_val

            entries.append({
                "dataset": db_name,
                "type": dtype,
                "n_cal": n_cal,
                "n_val": n_val,
                "n_total": n_total
            })

    df = pd.DataFrame(entries)
    return df


# -----------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------
def plot_dataset_distribution(df, output_folder="Figures"):
    """
    Creates several plots summarizing dataset statistics:
      - Bar chart: total samples per dataset
      - Pie chart: proportion of samples
      - Cumulative distribution curve (Pareto)
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ---- Bar chart ----
    plt.figure(figsize=(12, 6))
    df_sorted = df.sort_values("n_total", ascending=False)
    plt.bar(df_sorted["dataset"], df_sorted["n_total"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Number of samples per NIRS dataset")
    plt.ylabel("Total samples")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "bar_samples_per_dataset.png"))
    plt.close()

    # ---- Pie chart ----
    plt.figure(figsize=(10, 10))
    plt.pie(df_sorted["n_total"], labels=df_sorted["dataset"], autopct="%1.1f%%")
    plt.title("Proportion of samples per dataset")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "pie_samples_distribution.png"))
    plt.close()

    # ---- Cumulative distribution (Pareto) ----
    plt.figure(figsize=(12, 6))
    cumulative = df_sorted["n_total"].cumsum() / df_sorted["n_total"].sum()
    plt.plot(df_sorted["dataset"], cumulative, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Cumulative sample distribution (Pareto)")
    plt.ylabel("Cumulative proportion")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "cumulative_distribution.png"))
    plt.close()


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":

    # Step 1: scan full dataset collection
    df = scan_nirs_datasets(base_path="Data")

    # Print summary
    print("\n========== SUMMARY ==========")
    print(df)

    print("\nTotal number of datasets:", len(df))
    print("Total number of samples:", df['n_total'].sum())
    print("Average samples per dataset:", df['n_total'].mean())

    print("\nSamples per type:")
    print(df.groupby("type")["n_total"].sum())

    # Step 2: generate plots
    output_path = "Figures/assoc_pp_model/All_datasets/stats_datasets/"
    plot_dataset_distribution(df, output_folder=output_path)

    print(f"\nAll plots saved in {output_path}")
