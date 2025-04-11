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
import networkx as nx

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

def compare_preprocessings(preprocessings, Xcal, distance_fn=dissim_KS, normalize_dissim=False, threshold=("value",0.1)):
    """
    Computes dissimetries between the datasets transformed by the selected preprocessing methods.
    
    Parameters:
        preprocessings (dict): - key: preprocessing name (string)
        - value: preprocessing method (object).
        Xcal (pandas dataframe or numpy ndarray): A dataset containing the calibration spectra.
        distance_fn (function): A function to compute the distance between two datasets.
        normalize_dissim (bool): If True, normalize the dissimilarity values by their maximum value.
        threshold (string, float): - string: whether "value" or "proportion",
        - float: if string is "value", the threshold is the maximum dissimilarity value to consider two preprocessing methods as similar.
        IF string is "proportion", the threshold gives the proportion of preprocessing pairs to consider as similar.

    Returns:
        dict, similar_pairs (dict, list of string couples): dict gathers a couple of preprocessing names as a key,
        the dissimilarity associated as a value. 
        similar_pairs is a list of tuples containing the names of the preprocessing methods that are considered as similar.
    """
    preprocessing_list = list(preprocessings.values())
    preprocessing_names = list(preprocessings.keys())
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

        # Determine the similar preprocessing pairs based on the threshold
        if threshold[0] == "value":
            similar_pairs = [pair for pair, dist in dict.items() if dist < threshold[1]]
        else:
            similar_pairs = [pair for pair, dist in dict.items() if dist < np.percentile(list(dict.values()), threshold[1]*100)]

    return dict, similar_pairs









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

        # Test if one of the variables is constant, which would lead to NaN values for kurtosis and skewness
        isnan1 = any(np.isnan(x) for x in meta_attributes_x1_i.values())
        isnan2 = any(np.isnan(x) for x in meta_attributes_x2_i.values())
        if isnan1 or isnan2:
            # Delete the kurtosis and skewness attributes for both variables
            del meta_attributes_x1_i['kurtosis'], meta_attributes_x1_i['skewness']
            del meta_attributes_x2_i['kurtosis'], meta_attributes_x2_i['skewness']

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





# Function that shows the preprocessing methods to remove in an optimal way

def preprocessings_to_remove_approx(preprocessings, Xcal, distance_fn=dissim_KS, normalize_dissim=False, threshold=("value",0.1), verbose=False):
    """
    Determines the preprocessing methods that might be removed so that it keeps the maximum number of them in the calibration phase.
    
    Parameters:
        preprocessings (dict): - key: preprocessing name (string)
        - value: preprocessing method (object).
        Xcal (pandas dataframe): A dataset containing the calibration spectra.
        distance_fn (function): A function to compute the distance between two datasets.
        normalize_dissim (bool): If True, normalize the dissimilarity values by their maximum value.
        threshold (string, float): - string: whether "value" or "proportion",
        - float: if string is "value", the threshold is the maximum dissimilarity value to consider two preprocessing methods as similar.
        IF string is "proportion", the threshold gives the proportion of preprocessing pairs to consider as similar.
        verbose (bool): If True, print the names of the preprocessing methods that are removed.
    Returns:
        A list of preprocessing names to remove with an approximation of the minimum vertex cover problem.
    """
    # Declare a graph structure
    G = nx.Graph()

    # Determine the close preprocessing pairs
    _, close_preprocessings = compare_preprocessings(preprocessings, Xcal, distance_fn, normalize_dissim, threshold)

    # Fill the graph with the close preprocessing pairs
    G.add_edges_from(close_preprocessings)

    # Find an approximate minimum vertex cover of the graph
    list_removal = nx.algorithms.approximation.min_weighted_vertex_cover(G)

    if verbose:
        print("List of preprocessing methods to remove for %s:" % distance_fn.__name__, list_removal)
        print("-"*(60 + 20*len(list_removal)))
    return  list_removal