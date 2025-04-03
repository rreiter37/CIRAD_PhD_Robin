##### Getting started with the Pinard package

### Importation of required libraries
import time
import numpy as np
import pinard as pn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display

from pinard import utils
from pinard import preprocessing as pp
from pinard.model_selection import train_test_split_idx

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


### Importation of the datasets

path = 'C:/Users/reiter/CIRAD_PhD_Robin/CIRAD_PhD_Robin/Data'
mode = 'Regression' # 'Classification' or 'Regression'
 
## Choose the source of data to be imported
# 'BeerOriginalExtract' or 'Digest_0.8' or 'YamProtein' for regression //
# 'CoffeeSpecies' or 'Malaria2024' or 'mDigest_custom3' or 'WhiskyConcentration' or 'YamMould' for classification
data_source = 'BeerOriginalExtract'


## Importation of the datasets with the adapted path

Xcal = pd.read_csv(path + f'/{mode}/{data_source}/Xcal.csv', sep=';')
Xval = pd.read_csv(path + f'/{mode}/{data_source}/Xval.csv', sep=';')
Ycal = pd.read_csv(path + f'/{mode}/{data_source}/Ycal.csv', sep=';')
Yval = pd.read_csv(path + f'/{mode}/{data_source}/Yval.csv', sep=';')

display(Ycal.head(5))



### Function that allows a visualization of the spectra

def plot_spectra(X, names, title='Visualization of the spectra'):
    if isinstance(X, pd.DataFrame):
        X = [X.values]
    elif isinstance(X, np.ndarray):
            X = [X]

    plt.figure(figsize=(10, 5))
    colors = sns.color_palette(palette = "Paired")
    for i, dataset in enumerate(X):
         plt.plot(dataset.T, color=colors[i])
         plt.plot([], [], color=colors[i], label=names[i])

    plt.legend(loc='upper left', fontsize='small')
    plt.title(title)
    plt.xlabel('Wavelength')
    plt.ylabel('Absorbance')
    plt.xticks(np.arange(0, dataset.shape[1], dataset.shape[1]//10), rotation=45)
    plt.show(block=False)



### Test of pipelines with Pinard

# Simple pipeline declaration
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('preprocessing', pp.SavitzkyGolay()),
    ('pls', PLSRegression(n_components=10, scale=False))
    ])

# Transform the data with the desired part of the pipeline
Xcal_transformed = pipeline[1].fit_transform(Xcal)


### Visualization of the spectra
plot_spectra([Xcal, Xcal_transformed], 
             names = ['Before transformation', 'After transformation'], 
             title= 'Visualization of the spectra with or without preprocessing')


### 