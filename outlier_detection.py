### Libraries importation

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from itertools import combinations
from collections import defaultdict
from scipy.spatial.distance import mahalanobis

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import chi2
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Input, Bidirectional, LSTM, RepeatVector, TimeDistributed, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action="ignore", category=RuntimeWarning)





### Functions to compute the mean score of Jaccard between outliers detected by different methods

def jaccard_score(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

def jaccard_mean_scores(outlier_dict):
    methods = list(outlier_dict.keys())
    sets = {m: set(outlier_dict[m]) for m in methods}
    scores = {}

    for m in methods:
        jaccards = [
            jaccard_score(sets[m], sets[other])
            for other in methods if other != m
        ]
        scores[m] = sum(jaccards) / len(jaccards) if jaccards else 0.0

    return scores






### Function to save results in a json file

def save_results_to_json(name_method, data_source, exec_time, scores, dataset_size, epochs, name_cost, dict_outliers):
    """
    Save results to a JSON file.
    """
    file_path = "Outputs/outliers_detection/%s/%s/results_outlier_detection.json" % (name_method, data_source)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Convert numpy arrays to lists to ensure JSON compatibility
    dict_outliers_serializable = {}
    for key, value in dict_outliers.items():
        if len(value) == 0:
            dict_outliers_serializable[key] = []
        elif isinstance(value, np.ndarray):
            dict_outliers_serializable[key] = list(set(value.tolist()))
        else:
            dict_outliers_serializable[key] = list(set(value))
    # Compute Jaccard mean scores
    dict_jaccard = jaccard_mean_scores(dict_outliers_serializable)
    
    results = {
        "Method": name_method,
        "Name Dataset": data_source,
        "Execution time (s)": exec_time,
        "Reconstruction scores": scores,
        "Dataset size": dataset_size,
        "Epochs": epochs,
        "Cost func": name_cost,
        "Outliers detected": dict_outliers_serializable,
        "Jaccard mean scores": dict_jaccard,
    }
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {file_path}")


### Function to plot outliers that are given as a list of indices

def plot_spectra_outliers(X, dict_outliers, names, data_source, title='Visualization of the spectra with outliers', save_fig=False, name_method = "PCA", save_results=True, exec_time=0, scores={}, dataset_size=0, epochs=0, name_loss="MSE"):
    """
    Plots the spectra with outliers highlighted in red for each preprocessing method used.
    
    Parameters :
        X (list of arrays or pandas.dataframe or numpy.ndarray): List of spectra for each preprocessing method. 
                                                                 If a single dataframe or array is provided, it is included in a one-component list.
        dict_outliers (dict): - Keys: names of the preprocessing methods, 
                              - Values: lists of indices of outliers for each method.
        names (list): List of names for each preprocessing method.
        data_source (str): Name of the dataset used for the analysis.
        title (str): Title of the plot.
        save_fig (bool): If True, saves the figure in the Figures/outliers_detection folder.
        name_method (str): Name of the method used for outlier detection for the title of the saved figure.
    """

    if isinstance(X, pd.DataFrame):
        X = [X.values]
    elif isinstance(X, np.ndarray):
            X = [X]

    fig, axs = plt.subplots((len(X)+1)//2, 2, figsize=(15, 4*len(X)//2), dpi=100)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i >= len(X):
            ax.axis('off')
        else:
            name = names[i]
            dataset = X[i]
            if isinstance(dataset, pd.DataFrame):
                 dataset = dataset.values
            outliers = dict_outliers[name]
            ax.set_title(name)
            ax.set_xlabel('Wavelength')
            ax.set_ylabel('Absorbance')
            ax.set_xticks(np.arange(0, dataset.shape[1], dataset.shape[1]//10))
            ax.plot(dataset.T, color='blue', alpha = 0.1)
            ax.plot([], [], color='blue', label=name)

            if len(outliers) > 0:
                ax.plot(dataset[outliers,:].T, color='red', linewidth=0.5)
                ax.plot([], [], color='red', label='outliers', linewidth=0.5)
            
            ax.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(title, fontweight='bold')

    # Save results if required in the folder Outputs/outliers_detection
    if save_results:
        save_results_to_json(name_method, data_source, exec_time, scores, dataset_size, epochs, name_loss, dict_outliers)
    
    # Save figure if required in the folder Figures/outliers_detection
    if save_fig:
        file_path = "Figures/outliers_detection/%s/%s/see_outliers.png" % (name_method, data_source)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        fig.savefig(file_path, dpi=300)
    plt.show()


### Function to compute the dynamic threshold for the outliers detection
def compute_dynamic_threshold(scores, n_test):
        mean = np.mean(scores)
        std = np.std(scores)
        coeff_threshold = np.sqrt(n_test) / np.log(n_test+2)
        return mean + coeff_threshold * std


###  Function to find the outliers with the PCA method

def outlier_detection_PCA(X):
    """
    Detects outliers in the dataset using PCA method.

    Parameters:
        X (numpy.ndarray or pandas.DataFrame): The spectral data set from which outliers must be detected.
    """
    scaler = StandardScaler(with_std=False)
    X_scaled = scaler.fit_transform(X)

    # Perform PCA on the scaled data
    X_pca = PCA(n_components=1).fit_transform(X_scaled)

    # Store the PCA scores in a DataFrame for easier manipulation
    scores_df = pd.DataFrame(X_pca, columns=['PC1'])

    # Compute the bounds for extreme individuals
    mean_PC1 = scores_df['PC1'].mean()
    std_PC1 = scores_df['PC1'].std()
    lower_bound = mean_PC1 - 2 * std_PC1
    upper_bound = mean_PC1 + 2 * std_PC1

    # Filter the extreme individuals based on the bounds
    outliers = scores_df[
        (scores_df['PC1'] < lower_bound) | (scores_df['PC1'] > upper_bound)
    ]

    return outliers.index.tolist()








### Functions to find the outliers with the Data Depth Theory method

def diff_abs_j(X,j):
    N_T, n_x = X.shape

    # Compute the differences of a given absorbance by all other absorbances in the same column of the spectral dataset
    Q = np.tile(X[j,:], (N_T,1))
    X_2 = X - Q

    # Compute the norm 2 of each line of this matrix and then square it
    X_fin = np.square(np.linalg.norm(X_2, axis=0))

    return X_fin

def detect_outliers_DDT_ED(X, kM=2.0):
    """
    Detects outliers in the dataset using the Data Depth Theory method.

    Parameters:
        X (numpy.ndarray or pandas.DataFrame): The spectral data set from which outliers must be detected.
        kM (float): Multiplicative coefficient used to fix the threshold value to decide wether a spectrum is an outlier or not.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    N_T, n_x = X.shape

    # Compute the ED for each spectrum
    distances = np.array([diff_abs_j(X,j) for j in range(N_T)])
    R = np.linalg.norm(distances, axis=1, ord=1)
    ED = 1/N_T * np.sqrt(R)

    # Define a threshold and detect outliers
    threshold = kM * np.median(ED)
    mask = ED > threshold
    outliers_ind = np.where(mask)
    return list(outliers_ind[0])


def other_outlier_detection(X):
    """
    Detects outliers in the dataset by finding the spectra above or below every all spectra.

    Parameters:
        X (numpy.ndarray or pandas.DataFrame): The spectral data set from which outliers must be detected.
    """
    N_T, n_x = X.shape
    outlier_mask = np.zeros(N_T, dtype=bool)

    for j in range(N_T):
        above_all = np.all(X[j, :] > X[np.arange(N_T) != j], axis=0).all()
        below_all = np.all(X[j, :] < X[np.arange(N_T) != j], axis=0).all()
        if above_all or below_all:
            outlier_mask[j] = True
    outliers_ind = list(np.where(outlier_mask)[0])
    return outliers_ind














### Functions to find the outliers with the Robust DDT method

def identify_mrs(X):
    """
    Identifies the Most Representative Spectrum (MRS) from a matrix of spectra
    using the data depth approach described in Section 2.2.1, Step 2.

    Parameters:
        X (numpy.ndarray): Matrix of shape (NR, nx), where NR is the number of
                                 spectra and nx is the number of wavelengths.

    Returns:
        mrs_index (int): Index of the most representative spectrum.
    """
    if isinstance(X, pd.DataFrame): X = X.values # make sure X is a numpy.ndarray

    NR, nx = X.shape
    sdiff = np.zeros(NR)
    sequal = np.zeros(NR)

    for j in range(NR):
        diff = np.zeros(nx)
        equal = np.zeros(nx)
        for i in range(nx):
            # Absorbance for spectrum j at wavelength i
            val_j = X[j, i]
            # Comparison across all other spectra
            higher = np.sum(X[:, i] > val_j)
            equal_i = np.sum(X[:, i] == val_j)
            lower = np.sum(X[:, i] < val_j)
            diff[i] = abs(higher - lower)
            equal[i] = equal_i
        sdiff[j] = np.sum(diff)
        sequal[j] = np.sum(equal)

    # Find the spectrum with minimum SDIFF
    min_diff = np.min(sdiff)
    candidates = np.where(sdiff == min_diff)[0]

    # If tie, use the one with maximum SEQUAL
    if len(candidates) > 1:
        mrs_idx = candidates[np.argmax(sequal[candidates])]
    else:
        mrs_idx = candidates[0]

    return mrs_idx




def compute_mahalanobis_distances(X, ref_index):
    """
    Computes Mahalanobis distance of each spectrum to the reference spectrum.

    Parameters:
        X (np.ndarray): Matrix of shape (N_T, n_x), with N_T spectra and n_x wavelengths.
        ref_index (int): Index of the reference spectrum in the dataset.

    Returns:
        distances (np.ndarray): Array of Mahalanobis distances of shape (N_T,).
    """
    if isinstance(X, pd.DataFrame): X = X.values # make sure X is a numpy.ndarray

    N_T, n_x = X.shape

    # Reference spectrum
    ref_spectrum = X[ref_index]

    # Covariance matrix of the dataset
    cov_matrix = np.cov(X, rowvar=False)
    
    # Regularization in case of singular matrix
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Add a small diagonal value to regularize
        inv_cov_matrix = np.linalg.pinv(cov_matrix + np.eye(n_x) * 1e-10)

    # Compute Mahalanobis distance for each spectrum
    distances = np.array([
        mahalanobis(X[i], ref_spectrum, inv_cov_matrix)
        for i in range(N_T)
    ])

    return distances




def detect_outliers_robust_ddt(X, coeffs, scale = False):
    """
    Detects the outliers in a spectral dataset with a more robust method than proposed above with Data Depth Theory.

    Parameters:
        X (numpy.ndarray): Matrix of shape (NT, nx), where NT is the number of spectra and nx is the number of wavelengths.
        coeffs (list of floats): List of kM coefficients used to estimate the first outliers and to find the MRS.
        scale (bool): If True, scales the data to the range [0, 1] before processing.

    Returns:
        outliers_ind (list): Indices of outliers in the original data set.
    """
    if isinstance(X, pd.DataFrame): X = X.values # make sure X is a numpy.ndarray

    if scale: 
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    
    ### Estimate the MRS with several values of kM

    mrs_list = []
    for kM in coeffs:
        # Perform outliers detection with the DDT method
        outliers_ind = detect_outliers_DDT_ED(X, kM=kM)

        # Filter the data set by removing these first outliers
        x = np.delete(X, outliers_ind, axis=0)
        outliers_ind_else = other_outlier_detection(x)

        # Filter another set of spectra outliers
        x = np.delete(x, outliers_ind_else, axis=0)

        # Identify the MRS with the filtered data set
        mrs_idx = identify_mrs(x)

        # Store the MRS index in a list
        mrs_list.append(mrs_idx)
    
    # Find the most recurrent spectrum index in this list
    counter = Counter(mrs_list)
    final_mrs, _ = counter.most_common(1)[0]

    ### Identify outliers with an abnormal Mahalanobis distance of a spectrum from the MRS
    
    distances = compute_mahalanobis_distances(X, final_mrs)

    # Define a threshold based on the mean and standard deviation of computed distances
    mean_dist = np.nanmean(distances)
    std_dist = np.nanstd(distances)
    threshold = mean_dist + 3*std_dist
    
    # Detect outliers based on this threshold
    outliers_ind = np.where(distances > threshold)[0]

    return outliers_ind



















### Functions to detect the outliers with the outdst method based on space transformation and spectral analysis

def gaussian_kernel(x, y, sigma, D):
    div = (2 * np.pi * sigma)**(D/2)
    if div == 0: return 0 # justified with the comparison theorem on the exponential function when sigma tends to zero
    return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))/div

def local_quadratic_entropy(X, neighbors, sigma):
    N, D = X.shape
    QE = np.zeros(N)
    for i in range(N):
        neigh_idx = list(neighbors[i])
        neighs = X[neigh_idx,:]
        total = 0
        for p in neighs:
            for q in neighs:
                total += gaussian_kernel(p, q, np.sqrt(2) * sigma, D)
        if len(neighs) == 0:
            QE[i] = 0
        else:
            QE[i] = -np.log(total / (len(neighs)**2))
    return QE

def compute_affinity_matrix(X, neighbors, QE, sigma, gamma):
    N = X.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in neighbors[i]:
            if max(QE[i], QE[j]) == 0: 
                K[i, j] = 1
            else : 
                beta = min(QE[i], QE[j]) / max(QE[i], QE[j])
                if beta > gamma and max(QE[i], QE[j]) < np.mean(QE):
                    beta = np.mean(QE) / max(QE[i], QE[j])
                dist = np.linalg.norm(X[i] - X[j])
                coeff = (beta * sigma)**2
                if coeff == 0: # avoid errors with a division by zero
                    K[i, j] = 0
                else:
                    K[i, j] = np.exp(-dist**2 / (2 * coeff))
            K[j, i] = K[i, j]
    return K

def outdst(X, l_neighbors=10, gamma=0.7, c=0.05, d=3, with_distance=False, scale=False):
    """Finds the outliers in a spectral dataset using the outdst method.
    
    Parameters:
        X (numpy.ndarray): Matrix of shape (N, D), where N is the number of spectra and D is the number of wavelengths.
        l_neighbors (int): Number of nearest neighbors to consider for local quadratic entropy.
        gamma (float): Threshold for the ratio of local quadratic entropy above which two instances are considered from the same distribution.
        c (float): Coefficient to determine the sparsity of non-zero components in eigenvectors.
        d (int or str): Number of dimensions to keep after spectral decomposition. If "Kaiser", uses Kaiser criterion.
        with_distance (bool): If True, considers distance from origin in reduced space for outlier detection.
        scale (bool): If True, scales the data to the range [0, 1] before processing.
    """
    N, D = X.shape

    if scale: 
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    sigma = np.mean(np.std(X, axis=0))  # robust estimate of sd

    # Find the l nearest neighbors of each individual
    nn = NearestNeighbors(n_neighbors=l_neighbors).fit(X)
    _, indices = nn.kneighbors(X)
    neighbors = [set(neigh) - {i} for i, neigh in enumerate(indices)]

    # Compute the local quadratic entropy
    QE = local_quadratic_entropy(X, neighbors, sigma)

    # Build the matrix giving weights to the edges of the graph
    K = compute_affinity_matrix(X, neighbors, QE, sigma, gamma)

    # Compute the Laplacian of the graph
    L = laplacian(K, normed=False)

    # Spectral decomposition
    vals, vecs = eigh(L)

    ### Keep the first dimensions of the span of eigenvectors
    if d == "Kaiser":
        vecs = vecs[:, 1:] # the first eigenvector is zero, we dump it
        mean_var = np.mean(np.var(vecs, axis=0)) # Compute the mean variance from all dimensions
        Y = vecs[:, np.var(vecs, axis=0) > mean_var] # Each dimension whose variance is over the mean variance is kept
    else:
        Y = vecs[:, 1:d+1]  # the first eigenvector is zero, we dump it
    
    scores = np.linalg.norm(Y, axis=1) 

    ### Detection of outliers based on the number of non zero components in eigenvectors
    non_zero_components = np.sum(Y!=0.0, axis=0) # number of non zero components per dimension
    dimensions_candidates = non_zero_components <= c * N # condition to verify if non zero components are sparse in the eigenvector

    # if non zero components are sparse in the eigenvector, they are considered outliers
    outliers = np.array([])
    for dimension, is_candidate in enumerate(dimensions_candidates):
        if is_candidate: 
            pos = np.where(Y[:, dimension] != 0.0)[0]
            outliers = np.concatenate((outliers, pos), axis=0)
    
    ### Detection of outliers based on their distance from the origin in the reduced space
    if with_distance:
        n_outliers = int(c * N)
    
        # Outliers are then the most remote vectors from the origin
        outliers_2 = np.argsort(scores)[-n_outliers:]

        # merge the outliers detected with both appraoches
        outliers = np.concatenate((outliers_2, outliers), axis=0)
    outliers = np.unique(outliers).astype(int)

    # If we spectra have been detected as outliers because they are equal to zero, remove it from the list
    outliers = [int(idx) for idx in outliers if not np.all(X[idx] == 0)]

    return outliers


















### Function to find the normal spectra to train the Deep Learning model

def pipeline_find_normal_spectra(X, method = 'kNN', k = 10, percentile = 95):
    """
    Filters the spectra to keep only the normal ones based on the specified method.
    
    Parameters:
        X (numpy.ndarray): The spectral data set from which to find the normal spectra.
        method (str): The method to use for filtering ('kNN' or 'Mahalanobis').
        k (int): The number of nearest neighbors to consider for the kNN method.
        percentile (float): The percentile threshold for filtering outliers.

    Returns:
        X_filtered (numpy.ndarray): The filtered spectral data set containing only the normal spectra.
    """

    if method == 'kNN':
        # --- Compute kNN distances (mean distance to k nearest neighbors) ---
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')  # +1 because the point itself is included
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        knn_distances = distances[:, 1:].mean(axis=1)  # exclude self-distance at index 0

        # --- Determine distance threshold based on percentile of the flattened version of knn_distances ---
        threshold = np.percentile(knn_distances, percentile)

        # --- Filter inliers ---
        inliers = knn_distances <= threshold
        X_filtered = X[inliers]

    elif method == 'Mahalanobis':
        # --- Dimensionality Reduction with PCA ---
        # Keep enough components to retain ~99% of variance
        pca = PCA(n_components=0.99, svd_solver='full')
        X_reduced = pca.fit_transform(X)
        print(f"Reduced to {X_reduced.shape[1]} dimensions.")

        # --- Compute Mahalanobis distance in reduced space ---
        cov = EmpiricalCovariance().fit(X_reduced)
        mahal_dist = cov.mahalanobis(X_reduced)

        # --- Determine cutoff threshold ---
        threshold = chi2.ppf(percentile, df=X_reduced.shape[1])

        # --- Filter inliers ---
        inliers = mahal_dist < threshold
        X_filtered = X[inliers]

    return X_filtered

















### Functions to detect the outliers based on the LSTM AutoEncoder method

def pipeline_lstm_outliers(X, X_normal, latent_dim=64, epochs=100, batch_size=32, validation_split=0.2, lr=1e-4, verbose=False):
    """
    Detects outliers in the dataset using a LSTM AutoEncoder architecture.
    
    Parameters:
        X (numpy.ndarray): The spectral data set from which outliers must be detected.
        X_normal (numpy.ndarray): The normal spectra used to train the LSTM AutoEncoder.
        time_steps (int): Number of time steps for LSTM input.
        latent_dim (int): Dimensionality of the latent space.
        epochs (int): Number of epochs for training the model.
        batch_size (int): Batch size for training the model.
        threshold_percentile (float): Percentile for determining the anomaly threshold.
    """

    # --- Normalize all data ---
    scaler = MinMaxScaler()
    X_all_scaled = scaler.fit_transform(X)
    X_normal_scaled = scaler.transform(X_normal)

    # --- Dimensions for LSTM: (samples, features) ---
    
    X_train = X_normal_scaled
    X_full = X_all_scaled

    # --- Build LSTM Autoencoder ---
    input_dim = X_train.shape[1]  # Number of features (wavelengths)
    model = Sequential([
        LSTM(latent_dim, activation='relu', input_shape=(input_dim), return_sequences=False), # output shape (latent_dim)
        LSTM(latent_dim, activation='relu', return_sequences=False),
        Dense(input_dim) # shape
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    # --- Train the autoencoder ---
    model.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=validation_split,
                        verbose=verbose)

    # --- Compute reconstruction error on all spectra ---
    X_pred = model.predict(X_full, verbose=verbose)
    reconstruction_scores = np.mean((X_full - X_pred)**2, axis=-1)

    # Dynamic thresholding
    threshold = compute_dynamic_threshold(reconstruction_scores, X.shape[0])

    anomalies = reconstruction_scores > threshold
    outliers_indices = np.where(anomalies)[0]

    return outliers_indices, reconstruction_scores















### Function to detect the outliers with the bi-LSTM AutoEncoder method

def pipeline_bilstm_autoencoder(X, X_normal, latent_dim=64, epochs=100, batch_size=32, lr=1e-4, verbose=False, validation_split=0.2):
    """
    Detects outliers in the dataset using a Bi-LSTM AutoEncoder architecture.
    
    Parameters:
        X (numpy.ndarray): The spectral data set from which outliers must be detected.
        X_normal (numpy.ndarray): The normal spectra used to train the Bi-LSTM AutoEncoder.
        time_steps (int): Number of time steps for LSTM input.
        latent_dim (int): Dimensionality of the latent space.
        epochs (int): Number of epochs for training the model.
        batch_size (int): Batch size for training the model.
        threshold_percentile (float): Percentile for determining the anomaly threshold.
    """

    # Normalization
    scaler = MinMaxScaler()
    X_all_scaled = scaler.fit_transform(X)
    X_normal_scaled = scaler.fit_transform(X_normal)

    # Reshape to fit LSTM (samples, time_steps)
    X_train = X_normal_scaled.reshape((X_normal_scaled.shape[0], X_normal_scaled.shape[1]))
    X_all_reshaped = X_all_scaled.reshape((X_all_scaled.shape[0], X_all_scaled.shape[1]))

    # ---- Bi-LSTM Autoencoder ----
    input_dim = X_train.shape[1]  # Number of features (wavelengths)
    # Architecture of the model
    input_layer = Input(shape=(input_dim))
    encoded = Bidirectional(LSTM(latent_dim, return_sequences=False))(input_layer)
    decoded = Bidirectional(LSTM(latent_dim, return_sequences=False))(encoded)
    output_layer = Dense(input_dim)(decoded)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    # Training phase
    history = model.fit(X_train, X_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        shuffle=True,
                        verbose=verbose)

    # Reconstruction of all spectra
    X_pred = model.predict(X_all_reshaped, verbose=verbose)
    reconstruction_scores = np.linalg.norm(X_all_reshaped - X_pred, axis=-1) # norm 2 between the real spectrum and its reconstruction

    # Dynamic thresholding
    threshold = compute_dynamic_threshold(reconstruction_scores, X.shape[0])

    anomalies = reconstruction_scores > threshold
    outliers_indices = np.where(anomalies)[0]

    return outliers_indices, reconstruction_scores





### Function that computes the reconstruction score of a decoder
def compute_reconstruction_scores(model, data, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(data, dtype=torch.float32).to(device)
        try:
            pred = model(x_tensor).to(device).numpy()
        except:
            pred = model(x_tensor)[0]
            pred = pred.to(device).numpy()
    errors = np.linalg.norm(data - pred, axis=-1) # norm 2 between the real spectrum and its reconstruction
    return errors