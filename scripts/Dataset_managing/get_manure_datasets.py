import pandas as pd
import os
import numpy as np
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

### CREATE DATASETS FOR the targeted analyte ###
name_target = "CaO"
type_animal = "Poultry" # "Poultry" or "Cattle"

# Read the xlsx file of chemical measurements
df_chem = pd.read_excel("Data/Raw/manure/chemical_analysis.xlsx")
df_chem = df_chem[df_chem["Manure_type"]==type_animal+" manure"]
df_chem = df_chem[["Id_sample", name_target]]

# Read the xlsx file of dry manure
df_nirs = pd.read_csv("Data/Raw/manure/spectra-1.csv", decimal=",", quotechar='"')

df = pd.merge(left=df_chem, right=df_nirs, how="inner", on="Id_sample")
df.drop("Id_sample", axis=1, inplace=True)

X = df.iloc[:,1:]
Y = df[name_target]

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
path = os.path.join("Data", "Regression", f"{type_animal}_manure_{name_target}_KS")
os.makedirs(path, exist_ok=True)
Xcal.to_csv(os.path.join(path, "Xcal.csv"), index=False)
Ycal.to_csv(os.path.join(path, "Ycal.csv"), index=False)
Xval.to_csv(os.path.join(path, "Xval.csv"), index=False)
Yval.to_csv(os.path.join(path, "Yval.csv"), index=False)

print(f"Files Xcal.csv, Ycal.csv, Xval.csv, Yval.csv have been generated for the {type_animal} manure targeting {name_target}.")

print("min : ", Y.min())
print("max : ", Y.max())
print("mean : ", Y.mean())