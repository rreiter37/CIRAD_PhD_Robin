# Force the working directory to be the one of the Github repo
import os
os.chdir("/home/robinr/Desktop/VSCode/CIRAD_PhD_Robin")
print("Working dir:", os.getcwd())

import csv
from pathlib import Path
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

# Function to load a CSV file with automatic separator detection

def load_csv_auto_sep(mode, data_source, type_data, verbose=True, delimiter=None, index_col=None):

    ## Importation of the datasets with the adapted path
    file_name = Path("Data/%s/%s"% (mode,data_source))
    full_path = str(file_name.resolve()).replace("\\", "/")
    path = full_path + "/%s.csv" % type_data
    
    with open(path, 'r', newline='', encoding='utf-8-sig') as f:

        if delimiter is not None:
            sep = delimiter
        
        else:
            # Read a small portion of the file to detect the separator
            excerpt = f.read(1024)
            f.seek(0)  # return to the beginning of the file

            # Detection of the dialect
            dialect = csv.Sniffer().sniff(excerpt)
            sep = dialect.delimiter

        if verbose: print("Detected separator for %s: %s" % (type_data, sep))
        
        # Load the file with pandas
        df = pd.read_csv(f, delimiter=sep, index_col=index_col)

        if type_data[0]=='Y' and len(df.columns) > 1:
            # Drop the useless column if it exists
            df = df.drop(columns=[df.columns[1]])
        
        return df
    


######### Function to split indices with the KS method #########

def kennard_stone(X, n_samples):
    """
    Perform Kennard-Stone algorithm to select representative samples.
    
    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix (samples x features).
    n_samples : int
        Number of samples to select for calibration set.
    
    Returns
    -------
    list
        Indices of selected samples for calibration set.
    """
    # Compute distance matrix between all samples
    dist_matrix = pairwise_distances(X, metric='euclidean')
    
    # Find the two most distant samples
    i1, i2 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    selected = [i1, i2]
    
    # Iteratively select samples farthest from already selected ones
    while len(selected) < n_samples:
        remaining = list(set(range(X.shape[0])) - set(selected))
        min_distances = []
        
        for i in remaining:
            # Distance of a candidate sample to all selected ones
            dist_to_selected = [dist_matrix[i, j] for j in selected]
            # Take the minimum distance (closest to the selected set)
            min_distances.append(min(dist_to_selected))
        
        # Select the sample that maximizes the minimum distance
        next_sample = remaining[np.argmax(min_distances)]
        selected.append(next_sample)
    
    return selected
    

######## EXTRACT THE DATASETS FROM THE INNOSPECTRA MEASUREMENTS ########

# Load Innospectra measurements
df_nirs = load_csv_auto_sep(mode="Raw", data_source="Grapevines_chloride", type_data="innospectra_reflectance", verbose=True, delimiter=None, index_col=0)

# Load the chloridometer readings
df_chloride = load_csv_auto_sep(mode="Raw", data_source="Grapevines_chloride", type_data="chloridometer_readings", verbose=True, delimiter=None)

# Drop missing measures with missing values
to_drop = df_nirs[df_nirs["pot number"] == 266].index
df_nirs.drop(labels=to_drop, inplace=True)
to_drop = df_chloride[df_chloride["pot number"] == 266].index
df_chloride.drop(labels=to_drop, inplace=True)

# Keep spectra only
df_nirs.drop(labels="pot number", axis=1, inplace=True)

# Keep the averaged chloride content measure only
df_chloride = df_chloride["average"]



# Number of calibration samples (70% of dataset)
n_total = df_nirs.shape[0]
n_cal = int(0.7 * n_total)

# Apply Kennard-Stone selection on X
cal_indices = kennard_stone(df_nirs.values, n_cal)

# Validation set = the rest
val_indices = list(set(range(n_total)) - set(cal_indices))

# Split X and Y into calibration and validation sets
Xcal = df_nirs.iloc[cal_indices, :]
Ycal = df_chloride.iloc[cal_indices]

Xval = df_nirs.iloc[val_indices, :]
Yval = df_chloride.iloc[val_indices]

# Save the four CSV files
path = os.path.join("Data", "Regression", "grapevine_chloride_260_KS")
os.makedirs(path, exist_ok=True)
Xcal.to_csv(os.path.join(path, "Xcal.csv"), index=False)
Ycal.to_csv(os.path.join(path, "Ycal.csv"), index=False)
Xval.to_csv(os.path.join(path, "Xval.csv"), index=False)
Yval.to_csv(os.path.join(path, "Yval.csv"), index=False)

print("Files Xcal.csv, Ycal.csv, Xval.csv, Yval.csv have been generated for the Innospectra measures.")



######### EXTRACT THE DATASETS FROM SVC MEASUREMENTS #########

# Load SVC measurements
df_nirs_1 = load_csv_auto_sep(mode="Raw", data_source="Grapevines_chloride", type_data="230606_svc_reflectance", verbose=True, delimiter=None)
df_nirs_2 = load_csv_auto_sep(mode="Raw", data_source="Grapevines_chloride", type_data="230718_svc_reflectance", verbose=True, delimiter=None)

# Load the chloridometer readings
df_chloride_1 = load_csv_auto_sep(mode="Raw", data_source="Grapevines_chloride", type_data="chloridometer_readings", verbose=True, delimiter=None)
df_chloride_1.rename(columns={"svc_id": "scan"}, inplace=True)

df_chloride_2 = load_csv_auto_sep(mode="Raw", data_source="Grapevines_chloride", type_data="chloridometer_readings (1)", verbose=True, delimiter=None)
df_chloride_2.rename(columns={"svc_id": "scan"}, inplace=True)

df1 = df_chloride_1.merge(df_nirs_1, how="outer", on="scan")
df1.dropna(how="any", inplace=True)

df2 = df_chloride_2.merge(df_nirs_2, how="outer", on="scan")
df2.dropna(how="any", inplace=True)

df = pd.concat([df1, df2], axis=0)

X = df.iloc[:,9:]
Y = df["average"]

# Number of calibration samples (70% of dataset)
n_total = X.shape[0]
n_cal = int(0.7 * n_total)

# Apply Kennard-Stone selection on X
cal_indices = kennard_stone(X.values, n_cal)

# Validation set = the rest
val_indices = list(set(range(n_total)) - set(cal_indices))

# Split X and Y into calibration and validation sets
Xcal = X.iloc[cal_indices, :]
Ycal = Y.iloc[cal_indices]

Xval = X.iloc[val_indices, :]
Yval = Y.iloc[val_indices]

# Save the four CSV files
path = os.path.join("Data", "Regression", "grapevine_chloride_556_KS")
os.makedirs(path, exist_ok=True)
Xcal.to_csv(os.path.join(path, "Xcal.csv"), index=False)
Ycal.to_csv(os.path.join(path, "Ycal.csv"), index=False)
Xval.to_csv(os.path.join(path, "Xval.csv"), index=False)
Yval.to_csv(os.path.join(path, "Yval.csv"), index=False)

print("Files Xcal.csv, Ycal.csv, Xval.csv, Yval.csv have been generated for the SVC measures.")