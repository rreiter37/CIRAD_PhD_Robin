import os
import pandas as pd
import numpy as np

# Read the xlsx file of chemical measurements
df = pd.read_excel("Data/Raw/Beef/Data NIR Marbling.xlsx")

# Remove the column "Animal Number"
df.drop("Animal Number", axis=1, inplace=True)

# Set new column names: column 0 keeps its original name, others take the value from row 1
df.columns = [df.columns[0]] + df.iloc[0, 1:].tolist()

# Remove the first row which contained new header names
df.drop(index=0, inplace=True)

# Reset index so iloc works correctly
df = df.reset_index(drop=True)

# Split X (features) and Y (target)
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]

# Number of calibration samples (2/3 of dataset)
n_total = X.shape[0]
n_cal = int(2/3 * n_total)

# Ensure reproducibility for random splitting
np.random.seed(42)

# ---------------------------------------------------------
# Random split ensuring the max-Y sample is in calibration
# ---------------------------------------------------------

# *** Position *** of the maximum value of Y  
idx_max_y = int(np.argmax(Y.values))   # gives a position, safe for iloc

# All row positions
all_positions = np.arange(n_total)

# Positions excluding the max-Y sample
remaining_positions = np.setdiff1d(all_positions, [idx_max_y])

# Number of additional calibration samples needed
n_cal_remaining = n_cal - 1

# Randomly select the remaining calibration positions
cal_random = np.random.choice(remaining_positions, size=n_cal_remaining, replace=False)

# Final calibration positions
cal_positions = np.concatenate(([idx_max_y], cal_random))

# Validation positions
val_positions = np.setdiff1d(all_positions, cal_positions)

# Create calibration and validation sets
Xcal = X.iloc[cal_positions, :]
Ycal = Y.iloc[cal_positions]

Xval = X.iloc[val_positions, :]
Yval = Y.iloc[val_positions]

# Save the four CSV files
path = os.path.join("Data", "Regression", "Beef_Marbling_RandomSplit")
os.makedirs(path, exist_ok=True)

Xcal.to_csv(os.path.join(path, "Xcal.csv"), index=False)
Ycal.to_csv(os.path.join(path, "Ycal.csv"), index=False)
Xval.to_csv(os.path.join(path, "Xval.csv"), index=False)
Yval.to_csv(os.path.join(path, "Yval.csv"), index=False)

print("Files Xcal.csv, Ycal.csv, Xval.csv, Yval.csv have been generated for the beef marbling dataset.")
print("min : ", Y.min())
print("max : ", Y.max())
print("mean : ", Y.mean())
