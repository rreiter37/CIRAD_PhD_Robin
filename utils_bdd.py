import csv
import pandas as pd
from pathlib import Path


# Function to load a CSV file with automatic separator detection

def load_csv_auto_sep(mode, data_source, type_data, verbose=True, delimiter=None):

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
        
        # Load the file with pandas
        df = pd.read_csv(f, delimiter=sep)
        

        if type_data[0]=='Y' and len(df.columns) > 1:
            # Drop the useless column if it exists
            df = df.drop(columns=[df.columns[1]])
        
        if verbose: 
            print("Detected separator for %s: %s" % (type_data, sep))
            if type_data == 'Xcal':
                print("Number of spectra for calibration: ", len(df))
            elif type_data == 'Xval':
                print("Number of spectra for test: ", len(df))

        return df
    

def split_data(mode, data_source, verbose=True):
    """
    Function to split the data into calibration and validation sets.
    """
    # Load the data
    Xcal = load_csv_auto_sep(mode, data_source, "Xcal", verbose=verbose)
    Ycal = load_csv_auto_sep(mode, data_source, "Ycal", verbose=verbose, delimiter= ' ')
    Xval = load_csv_auto_sep(mode, data_source, "Xval", verbose=verbose)
    Yval = load_csv_auto_sep(mode, data_source, "Yval", verbose=verbose, delimiter= ' ')

    return Xcal, Ycal, Xval, Yval