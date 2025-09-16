# Force the working directory to be the one of the Github repo
import os
os.chdir("/home/robinr/Desktop/VSCode/CIRAD_PhD_Robin")
print("Working dir:", os.getcwd())

import csv
from pathlib import Path
import pandas as pd


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

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

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



### Load the raw dataset
df = load_csv_auto_sep(mode="Raw", data_source="milk", type_data="data_table", verbose=True, delimiter=None)

### CREATE DATASETS FOR FAT CONTENT ###
# Construct the target vector
Y = df["Fat"]

# Construct the spectra dataset with appropriate column names
X = df.loc[:,"Trans_Tot_1":"Trans_Tot_256"]
X.rename(columns={f"Trans_Tot_{i}": f"X_{round(960 + 2.86*i, 1)}" for i in range(256)}, inplace=True)

# Split the dataset with the Kennard Stone method
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
path = os.path.join("Data", "Regression", "Milk_Fat_1224_KS")
os.makedirs(path, exist_ok=True)
Xcal.to_csv(os.path.join(path, "Xcal.csv"), index=False)
Ycal.to_csv(os.path.join(path, "Ycal.csv"), index=False)
Xval.to_csv(os.path.join(path, "Xval.csv"), index=False)
Yval.to_csv(os.path.join(path, "Yval.csv"), index=False)

print("Files Xcal.csv, Ycal.csv, Xval.csv, Yval.csv have been generated for the Fat content.")



### CREATE DATASETS FOR PROTEIN CONTENT ###
# Construct the target vector
Y = df["Prot"]

# Construct the spectra dataset with appropriate column names
X = df.loc[:,"Trans_Tot_1":"Trans_Tot_256"]
X.rename(columns={f"Trans_Tot_{i}": f"X_{round(960 + 2.86*i, 1)}" for i in range(256)}, inplace=True)

# Split the dataset with the Kennard Stone method
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
path = os.path.join("Data", "Regression", "Milk_Protein_1224_KS")
os.makedirs(path, exist_ok=True)
Xcal.to_csv(os.path.join(path, "Xcal.csv"), index=False)
Ycal.to_csv(os.path.join(path, "Ycal.csv"), index=False)
Xval.to_csv(os.path.join(path, "Xval.csv"), index=False)
Yval.to_csv(os.path.join(path, "Yval.csv"), index=False)

print("Files Xcal.csv, Ycal.csv, Xval.csv, Yval.csv have been generated for the Protein content.")


### CREATE DATASETS FOR LACTOSE CONTENT ###
# Construct the target vector
Y = df["Lact"]

# Construct the spectra dataset with appropriate column names
X = df.loc[:,"Trans_Tot_1":"Trans_Tot_256"]
X.rename(columns={f"Trans_Tot_{i}": f"X_{round(960 + 2.86*i, 1)}" for i in range(256)}, inplace=True)

# Split the dataset with the Kennard Stone method
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
path = os.path.join("Data", "Regression", "Milk_Lactose_1224_KS")
os.makedirs(path, exist_ok=True)
Xcal.to_csv(os.path.join(path, "Xcal.csv"), index=False)
Ycal.to_csv(os.path.join(path, "Ycal.csv"), index=False)
Xval.to_csv(os.path.join(path, "Xval.csv"), index=False)
Yval.to_csv(os.path.join(path, "Yval.csv"), index=False)

print("Files Xcal.csv, Ycal.csv, Xval.csv, Yval.csv have been generated for the Lactose content.")


### CREATE DATASETS FOR UREA CONTENT ###
# Construct the target vector
Y = df["Urea"]

# Construct the spectra dataset with appropriate column names
X = df.loc[:,"Trans_Tot_1":"Trans_Tot_256"]
X.rename(columns={f"Trans_Tot_{i}": f"X_{round(960 + 2.86*i, 1)}" for i in range(256)}, inplace=True)

# Split the dataset with the Kennard Stone method
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
path = os.path.join("Data", "Regression", "Milk_Urea_1224_KS")
os.makedirs(path, exist_ok=True)
Xcal.to_csv(os.path.join(path, "Xcal.csv"), index=False)
Ycal.to_csv(os.path.join(path, "Ycal.csv"), index=False)
Xval.to_csv(os.path.join(path, "Xval.csv"), index=False)
Yval.to_csv(os.path.join(path, "Yval.csv"), index=False)

print("Files Xcal.csv, Ycal.csv, Xval.csv, Yval.csv have been generated for the Urea content.")


### CREATE DATASETS FOR SOMATIC CELL COUNT ###
# Construct the target vector
Y = df["SCC"]

# Construct the spectra dataset with appropriate column names
X = df.loc[:,"Trans_Tot_1":"Trans_Tot_256"]
X.rename(columns={f"Trans_Tot_{i}": f"X_{round(960 + 2.86*i, 1)}" for i in range(256)}, inplace=True)

# Split the dataset with the Kennard Stone method
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
path = os.path.join("Data", "Regression", "Milk_SCC_1224_KS")
os.makedirs(path, exist_ok=True)
Xcal.to_csv(os.path.join(path, "Xcal.csv"), index=False)
Ycal.to_csv(os.path.join(path, "Ycal.csv"), index=False)
Xval.to_csv(os.path.join(path, "Xval.csv"), index=False)
Yval.to_csv(os.path.join(path, "Yval.csv"), index=False)

print("Files Xcal.csv, Ycal.csv, Xval.csv, Yval.csv have been generated for the Somatic Cell Count.")