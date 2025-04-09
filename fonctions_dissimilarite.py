# Importing the required libraries

import time
import os
import math
import csv
from pathlib import Path
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
from sklearn.linear_model import RidgeCV, LinearRegression
from xgboost import XGBRegressor
from scipy.stats import skew, kurtosis, ks_2samp, mode
from scipy.spatial.distance import pdist, squareform
from itertools import combinations


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)







# Functions that enable the computation of dissimilarities between two datasets using a multiple Kolmogorov-Smirnov test.

def ks_tests_paired(X1, X2):
    """
    Make a Kolmogorov-Smirnov test for each pair of variables in the spectral datasets X1 and X2, 
    each pair corresponding to a given wavelength.
    
    Parameters :
        X1 (list of arrays): Set of spectra with preprocessing method 1.
        X2 (list of arrays): Set of spectra with preprocessing method 2.
        
    Returns :
        results (list of dicts): List of dictionaries containing the KS statistic and p-value for each wavelength.
    """
    if len(X1) != len(X2):
        raise ValueError("The two lists must have the same length.")
    
    results = []
    
    for i, (x1_i, x2_i) in enumerate(zip(X1, X2)):
        ks_stat, p_val = ks_2samp(x1_i, x2_i)
        results.append({
            "paire": i,
            "ks_stat": ks_stat,
            "p_value": p_val
        })
    
    return results


def dissim_KS(X1, X2):
    """
    Compute the dissimilarity between two spectral datasets using the Kolmogorov-Smirnov test.
    
    Parameters:
        X1 (list of arrays): Set of spectra with preprocessing method 1.
        X2 (list of arrays): Set of spectra with preprocessing method 2.
        
    Returns:
        dissimilarity (float): Dissimilarity value between the two spectral datasets.
    """
    ks_results = ks_tests_paired(X1, X2)
    
    # Compute the dissimilarity as the mean of the KS statistics
    dissimilarity = np.mean([result["ks_stat"] for result in ks_results])
    
    return dissimilarity






# Function that enable the comparison betwwen diverse preprocessing methods based on their dissimilarities.

def compare_preprocessings(preprocessing_list, preprocessing_names, Xcal, distance_fn=dissim_KS, normalize_dissim=False):
    """
    Computes dissimetries between the datasets transformed by the selected preprocessing methods.
    
    Parameters:
        preprocessing_list (list of arrays): A list where each element is a preprocessing method.
        preprocessing_names (list of strings): A list with the name of each preprocessing method.
        Xcal (pandas dataframe): A dataset containing the calibration spectra.
        distance_fn (function): A function to compute the distance between two datasets.
        normalize_dissim (bool): If True, normalize the dissimilarity values by their maximum value.
        
    Returns:
        dissim_matrix (ndarray): A square matrix of pairwise distances.
    """
    n = len(preprocessing_list)
    dict = {}
    for i, j in combinations(range(n), 2):
        u, v = preprocessing_list[i], preprocessing_list[j]
        name1, name2 = preprocessing_names[i], preprocessing_names[j]
        x, y = u.fit_transform(Xcal), v.fit_transform(Xcal)
        dist = distance_fn(x, y)
        dict[(name1, name2)] = float(dist)

        # If normalization is needed to get a normalized dissimilarity function
        if normalize_dissim:
            val_max = max(dict.values())
            dict = {key: value / val_max for key, value in dict.items()}

    return dict









# Functions that enable the computation of dissimilarities based on the comparison of several meta-attributes on each variable.

def get_meta_att_variable(x):
    """
    Compute the meta attributes of the variable.
    
    Parameters:
        x (numpy array): A dataset containing spectra.
        
    Returns:
        meta_attributes (dict): A dictionnary of meta attributes.
    """
    meta_attributes = {}
    
    # Test if the values distribution is uniform with a KS test
    _, p_val = ks_2samp(x, np.random.uniform(size=len(x)))
    IsUniform = p_val > 0.05
    meta_attributes["is_uniform"] = 1 if IsUniform else 0

    # Get the minimum and maximum values
    meta_attributes["min"] = x.min()
    meta_attributes["max"] = x.max()

    # Get the kurtosis and skewness
    meta_attributes["kurtosis"] = kurtosis(x)
    meta_attributes["skewness"] = skew(x)

    # Get the mean, median, mode, variance and standard deviation for each variable
    meta_attributes["mean"] = x.mean(axis=0)
    meta_attributes["std"] = x.std(axis=0)
    meta_attributes["var"] = x.var(axis=0)
    meta_attributes["median"] = np.median(x)
    meta_attributes["mode"] = mode(x)[0]

    # Get the difference between the max and min values
    meta_attributes["value_range"] = x.max() - x.min()
    
    # Get the lower and upper outer fences of the values
    meta_attributes["lower_outer_fence"] = np.quantile(x, 0.25) - 3 * (np.quantile(x, 0.75) - np.quantile(x, 0.25))
    meta_attributes["upper_outer_fence"] = np.quantile(x, 0.75) + 3 * (np.quantile(x, 0.75) - np.quantile(x, 0.25))

    # Get the lower and higher quartiles
    meta_attributes["lower_quartile"] = np.quantile(x, 0.25)
    meta_attributes["higher_quartile"] = np.quantile(x, 0.75)

    # Get the lower and higher confidence intervals
    meta_attributes["lower_confidence_interval"] = x.mean() - 1.96 * (x.std() / math.sqrt(len(x)))
    meta_attributes["higher_confidence_interval"] = x.mean() + 1.96 * (x.std() / math.sqrt(len(x)))

    # Get the number of positive and negative values
    meta_attributes["positive_count"] = np.sum(x > 0)
    meta_attributes["negative_count"] = np.sum(x < 0)

    # Gry the percentage of positive and negative values
    meta_attributes["positive_percentage"] = np.sum(x > 0) / len(x) * 100
    meta_attributes["negative_percentage"] = np.sum(x < 0) / len(x) * 100

    # Test if there at least one positive value, and test if there is at least one negative value
    meta_attributes["has_positive"] = 1 if np.sum(x > 0) > 0 else 0
    meta_attributes["has_negative"] = 1 if np.sum(x < 0) > 0 else 0
    
    return meta_attributes


def meta_att_var_paired(X1, X2):
    """
    Compute the dissimilarity for each pair of variables in the spectral datasets X1 and X2, 
    each pair corresponding to a given wavelength.
    
    Parameters :
        X1 (list of arrays): Set of spectra with preprocessing method 1.
        X2 (list of arrays): Set of spectra with preprocessing method 2.
        
    Returns :
        results (list of dicts): List of dictionaries containing the KS statistic and p-value for each wavelength.
    """
    if len(X1) != len(X2):
        raise ValueError("The two lists must have the same length.")
    
    results = []
    
    for i, (x1_i, x2_i) in enumerate(zip(X1, X2)):
        meta_attributes_x1_i = get_meta_att_variable(x1_i)
        meta_attributes_x2_i = get_meta_att_variable(x2_i)

        # Calculate the Manhattan distance between the meta attributes of the two variables
        distance = sum(abs(meta_attributes_x1_i[key] - meta_attributes_x2_i[key]) for key in meta_attributes_x1_i.keys())
        results.append({
            "paire": i,
            "distance": distance
            })
    
    return results

def dissim_meta_att_var(X1, X2):
    """
    Compute the dissimilarity between two spectral datasets comparing meta attributes on pairs of variables.
    
    Parameters:
        X1 (list of arrays): Set of spectra with preprocessing method 1.
        X2 (list of arrays): Set of spectra with preprocessing method 2.
        
    Returns:
        dissimilarity (float): Dissimilarity value between the two spectral datasets.
    """

    results = meta_att_var_paired(X1, X2)
    
    # Compute the dissimilarity as the mean of the KS statistics
    dissimilarity = np.mean([result["distance"] for result in results])
    
    return dissimilarity