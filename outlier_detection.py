### Libraries importation

import math
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.fft as fft

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




def outlier_detection_DDT(X, kM=2.0):
    """
    Detects outliers in the dataset using the Data Depth Theory method.

    Parameters:
        X (numpy.ndarray or pandas.DataFrame): The spectral data set from which outliers must be detected.
        kM (float): Multiplicative coefficient used to fix the threshold value to decide wether a spectrum is an outlier or not.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    N_T = X.shape[0]

    # Compute the ED for each spectrum
    distances = np.array([diff_abs_j(X,j) for j in range(N_T)])
    R = np.linalg.norm(distances, axis=1, ord=1)
    ED = 1/N_T * np.sqrt(R)

    # Define a threshold and detect outliers
    threshold = kM * np.median(ED)
    mask = ED > threshold
    outliers_ind = np.where(mask)
    outliers_ind = list(outliers_ind[0])

    ### Detect other outliers by finding the spectra that are above or below every all spectra
    x = np.delete(X, outliers_ind, axis=0)
    outliers_ind_else = other_outlier_detection(x)

    # Store the extreme individuals in the dictionary
    outliers_ind_all = list(np.concatenate((outliers_ind, outliers_ind_else), axis=0))

    return outliers_ind_all

















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




def outlier_detection_DDT_robust(X, coeffs = [1.0, 10.0, 100.0], scale = False):
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
        outliers_ind = outlier_detection_DDT(X, kM=kM)

        # Filter the data set by removing these first outliers
        outliers_ind = np.array(outliers_ind, dtype=int)
        x = np.delete(X, outliers_ind, axis=0)

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
    threshold = mean_dist + 2*std_dist
    
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
            denom = len(neighs)**2
            if denom == 0 or total == 0:
                QE[i] = 0  # ou np.nan si vous voulez marquer explicitement
            else:
                QE[i] = -np.log(total / denom)

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
            if np.isnan(K[i, j]) or np.isinf(K[i, j]):
                K[i, j] = 0
            K[j, i] = K[i, j]
    return K

def outlier_detection_Space_transformation(X, l_neighbors=10, gamma=0.7, c=0.05, d=3, with_distance=False, scale=False):
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

    stds = np.std(X, axis=0)
    stds[stds == 0] = 1e-8
    sigma = np.mean(stds) # robust estimate of sd

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

    if np.isnan(L).any() or np.isinf(L).any():
        raise ValueError("Laplacian matrix contains NaN or Inf values.")


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

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=latent_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoded = self.decoder(hidden[-1])
        return decoded


def outlier_detection_LSTM(X, latent_dim=64, epochs=10, batch_size=32, lr=1e-4, verbose=False, get_normal=True, normal_method='kNN', k=10, percentile=95, return_scores=False):
    """
    Detects outliers in the dataset using a LSTM AutoEncoder architecture with PyTorch.
    
    Parameters:
        X (numpy.ndarray): The spectral data set from which outliers must be detected.
        X_normal (numpy.ndarray): The normal spectra used to train the LSTM AutoEncoder.
        time_steps (int): Number of time steps for LSTM input.
        latent_dim (int): Dimensionality of the latent space.
        epochs (int): Number of epochs for training the model.
        batch_size (int): Batch size for training the model.
        threshold_percentile (float): Percentile for determining the anomaly threshold.
    """

    # --- Find the normal spectra to train the model ---
    if get_normal:
        X_normal = pipeline_find_normal_spectra(X, method=normal_method, k=k, percentile=percentile)
    else:
        X_normal = X

    # --- Preprocess all data ---
    scaler = MinMaxScaler()
    X_all_scaled = scaler.fit_transform(X)
    X_normal_scaled = scaler.transform(X_normal)

    # --- Dimensions for LSTM: (samples, wavelengths, 1) ---
    
    X_train = X_normal_scaled.reshape((X_normal_scaled.shape[0], X_normal_scaled.shape[1], 1)) # just one feature per time step
    X_full = X_all_scaled.reshape((X_all_scaled.shape[0], X_all_scaled.shape[1], 1)) 

    # --- Build LSTM Autoencoder ---
    n_wavelengths = X_train.shape[1]  # number of wavelengths
    input_dim = n_wavelengths  # shape (wavelengths,1)
    
    model = LSTMAutoEncoder(input_dim, latent_dim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Prepare DataLoader ---
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Train the autoencoder ---
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs.squeeze(-1))
            loss.backward()

    # --- Compute reconstruction error on all spectra ---
    model.eval()
    X_full_tensor = torch.tensor(X_full, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_pred = model(X_full_tensor).cpu()

    reconstruction_errors = torch.mean((X_pred - X_full_tensor.squeeze(2))**2, dim=1).cpu().numpy()


    # Dynamic thresholding
    threshold = compute_dynamic_threshold(reconstruction_errors, X.shape[0])

    anomalies = reconstruction_errors > threshold
    outliers_indices = np.where(anomalies)[0]

    if return_scores:
    # Return the indices of outliers and their reconstruction scores
        return outliers_indices, reconstruction_errors

    return outliers_indices








### Function to detect the outliers with the bi-LSTM AutoEncoder method


class BiLSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(BiLSTMAutoEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=latent_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, input_dim)
        )

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden_concat = torch.cat((hidden[0], hidden[1]), dim=-1)
        decoded = self.decoder(hidden_concat)
        return decoded

def outlier_detection_BiLSTM(X, latent_dim=64, epochs=10, batch_size=32, lr=1e-4, verbose=False, validation_split=0.2, get_normal=True, normal_method='kNN', k=10, percentile=95, return_scores=False):
    """
    Detects outliers in the dataset using a Bi-LSTM AutoEncoder architecture with PyTorch.
    
    Parameters:
        X (numpy.ndarray): The spectral data set from which outliers must be detected.
        X_normal (numpy.ndarray): The normal spectra used to train the Bi-LSTM AutoEncoder.
        time_steps (int): Number of time steps for LSTM input.
        latent_dim (int): Dimensionality of the latent space.
        epochs (int): Number of epochs for training the model.
        batch_size (int): Batch size for training the model.
        threshold_percentile (float): Percentile for determining the anomaly threshold.
    """
    # --- Find the normal spectra to train the model ---
    if get_normal:
        X_normal = pipeline_find_normal_spectra(X, method=normal_method, k=k, percentile=percentile)
    else:
        X_normal = X

    # --- Preprocess data ---
    scaler = MinMaxScaler()
    X_all_scaled = scaler.fit_transform(X)
    X_normal_scaled = scaler.transform(X_normal)

    # Reshape to fit LSTM (samples, time_steps)
    X_train = X_normal_scaled.reshape((X_normal_scaled.shape[0], X_normal_scaled.shape[1], 1))
    X_all_reshaped = X_all_scaled.reshape((X_all_scaled.shape[0], X_all_scaled.shape[1], 1))

    # ---- Bi-LSTM Autoencoder ----
    n_wavelengths = X_train.shape[1]  # Number of wavelengths
    input_dim = n_wavelengths  # Number of features (wavelengths)
    
    model = BiLSTMAutoEncoder(input_dim, latent_dim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Prepare DataLoader ---
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Train the autoencoder ---
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs.squeeze(-1))
            loss.backward()

    # --- Compute reconstruction error on all spectra ---
    model.eval()
    X_full_tensor = torch.tensor(X_all_reshaped, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_pred = model(X_full_tensor).cpu()

    reconstruction_errors = torch.mean((X_pred - X_full_tensor.squeeze(2))**2, dim=1).cpu().numpy()

    # Dynamic thresholding
    threshold = compute_dynamic_threshold(reconstruction_errors, X.shape[0])

    anomalies = reconstruction_errors > threshold
    outliers_indices = np.where(anomalies)[0]

    if return_scores:
        return outliers_indices, reconstruction_errors

    return outliers_indices



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









###################################### TRANSFORMER BASED METHODS ######################################






### Anomaly Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# === Anomaly Attention & Transformer ===

class AnomalyAttention1D(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_sigma = nn.Linear(d_model, n_heads)

    def forward(self, x):
        B, _ = x.shape      # [B, d_model] where B is the batch size

        ### Prior attention branch
        W_sigma = nn.Linear(self.d_model, self.n_heads)
        sigma = torch.abs(W_sigma(x)) + 1e-5       # [B, H]
        positions = torch.arange(B, device=x.device).unsqueeze(0)       # [1, B]
        prior = []
        for h in range(self.n_heads):
            dists = (positions.T - positions) ** 2      # [B, B]
            gauss = torch.exp(-dists[None, :, :] / (2 * sigma[:, h].unsqueeze(1) ** 2))
            gauss = gauss / (torch.sqrt(2 * math.pi * sigma[:, h].unsqueeze(1) ** 2))
            gauss = gauss / gauss.sum(dim=-1, keepdim=True)
            prior.append(gauss)
        P = torch.stack(prior, dim=1).squeeze(0)       # [H, B, B]

        ### Series attention branch
        Q = self.W_q(x).view(B, self.n_heads, self.head_dim).transpose(0, 1)        # [H, B, d_model]
        K = self.W_k(x).view(B, self.n_heads, self.head_dim).transpose(0, 1)        # [H, B, d_model]
        V = self.W_v(x).view(B, self.n_heads, self.head_dim).transpose(0, 1)        # [H, B, d_model]

        S = torch.softmax(Q @ K.transpose(1,2) / (self.head_dim ** 0.5), dim=-1)       # [H, B, B]
        out = S @ V       # [H, B, d_model]

        # Concatenate the heads along the last dimension
        out = out.transpose(1, 2).reshape(B, self.d_model)

        return out, S, P


class AnomalyTransformer1D(nn.Module):
    def __init__(self, seq_len, d_model=512, num_layers=3, n_heads=8):
        super().__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.input_proj = nn.Linear(seq_len, d_model)
        # Stacked Transformer encoder
        self.anomaly_layers = nn.ModuleList([
            AnomalyAttention1D(d_model=d_model, n_heads=n_heads, seq_len=seq_len)
            for _ in range(num_layers)
        ])
        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
            ) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        self.reconstruction = nn.Linear(d_model, seq_len)

    def forward(self, x):
        x = self.input_proj(x)  # [B, L] → [B, d_model]
        
        list_P, list_S = [], []
        for ind in range(self.num_layers):
            # Anomaly-Attention + Skip + Norm
            attn_layer = self.anomaly_layers[ind]
            attn_out, S, P = attn_layer(x)
            norm1_layer = self.norm1_layers[ind]
            z = norm1_layer(attn_out + x)
            
            # FeedForward + Skip + Norm
            ff_layer = self.ff_layers[ind]
            z2 = ff_layer(z)
            norm2_layer = self.norm2_layers[ind]
            x = norm2_layer(z2 + z)

            list_P.append(P)
            list_S.append(S)

        P = torch.cat(list_P, dim=0)        # [H*num_layers, B, B]
        S = torch.cat(list_S, dim=0)        # [H*num_layers, B, B]

        x_hat = self.reconstruction(x)  # [B, L]
        return x_hat, P, S



def association_discrepancy(P, S, eps=1e-8):
    P_safe = P + eps
    S_safe = S + eps

    # KL divergences par point et par head : [H, B, B] -> [H, B]
    kl_1 = F.kl_div(P_safe.log(), S_safe, reduction='none').sum(dim=1)
    kl_2 = F.kl_div(S_safe.log(), P_safe, reduction='none').sum(dim=1)

    # Moyenne sur les heads et les layers → [B]
    return (kl_1 + kl_2).mean(dim=0)


def compute_loss(x, x_hat, P, S, lam=3.0):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    ass_dis = association_discrepancy(P, S).mean()
    return recon_loss - lam * ass_dis, recon_loss, ass_dis


# === Training phase ===

def train_minimax_model(model, train_loader, epochs=10, lr= 1e-4, lam=3.0, device=torch.device('cpu'), verbose=True):
    model.train()
    model.to(device)
    if lr is None: optimizer = torch.optim.Adam(model.parameters())
    else : optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x in train_loader:
            batch_x = batch_x[0].to(device)  # [B, L]
            
            # Min phase
            x_hat, S, P = model(batch_x)
            loss_min, _, _ = compute_loss(batch_x, x_hat, P, S.detach(), lam=-lam)
            optimizer.zero_grad()
            loss_min.backward()
            optimizer.step()

            # Max phase
            x_hat, S, P = model(batch_x)
            loss_max, _, _ = compute_loss(batch_x, x_hat, P.detach(), S, lam=lam)
            optimizer.zero_grad()
            loss_max.backward()
            optimizer.step()

            total_loss += loss_max.item()

        if verbose: print(f"Epoch {epoch + 1}/{epochs} — Loss: {total_loss / len(train_loader):.4f}")


# === Definition of the score function ===

def compute_anomaly_scores(model, np_data, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(np_data, dtype=torch.float32).to(device)
        x_hat, S, P = model(x)
        recon_error = ((x - x_hat) ** 2).sum(dim=1).detach()
        ass_dis = association_discrepancy(P, S).detach()
        weights = torch.softmax(-ass_dis, dim=0)
        anomaly_scores = weights * recon_error
        return anomaly_scores.cpu().numpy()


def compute_dynamic_threshold(scores, n_test):
        mean = np.mean(scores)
        std = np.std(scores)
        coeff_threshold = np.sqrt(n_test) / np.log(n_test+2)
        return mean + 2 * std
    
    

def outlier_detection_Anomaly_transformer(X_train, batch_size=32, epochs=10, lr=1e-4, lam=3.0, d_model=512, num_layers=3, n_heads=8, verbose=False, return_scores=False):
    if isinstance(X_train, pd.DataFrame): X_train = X_train.values
    X_test = X_train
    if isinstance(X_test, pd.DataFrame): X_test = X_test.values
    p = X_train.shape[1]

    # Create DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyTransformer1D(seq_len=p, d_model=d_model, num_layers=num_layers, n_heads=n_heads).to(device)

    # Train the model
    train_minimax_model(model, train_loader, epochs=epochs, lr=lr, lam=lam, device=device, verbose=verbose)
    
    # Compute anomaly scores
    test_scores = compute_anomaly_scores(model, X_test, device=device)

    # Compute reconstruction scores
    reconstruction_scores = compute_reconstruction_scores(model, X_test, device)

    # Dynamic thresholding
    threshold = compute_dynamic_threshold(test_scores, len(X_test))

    anomalies = test_scores > threshold
    outliers_indices = np.where(anomalies)[0]

    if return_scores:
        return outliers_indices, reconstruction_scores
    
    return outliers_indices





### STOC

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.exceptions import TrialPruned

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, length):
        super().__init__()
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# -----------------------------
# STOC Model
# -----------------------------
class STOC(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, length=input_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])

        self.conv1d = nn.Conv1d(d_model * num_layers, d_model, kernel_size=3, padding=1)
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        h_list = []
        for layer in self.transformer_layers:
            x = layer(x)
            h_list.append(x)

        h_stack = torch.cat(h_list, dim=2)
        h_conv = self.conv1d(h_stack.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.output_layer(h_conv).squeeze(1)
        return out

# -----------------------------
# Training
# -----------------------------
def train_model_STOC(model, train_loader, val_loader, trial=None, epochs=50, lr=1e-4, device=torch.device("cpu"), patience=10):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch in train_loader:
            x_batch = x_batch[0].to(device)
            pred = model(x_batch)
            loss = criterion(pred, x_batch.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val in val_loader:
                x_val = x_val[0].to(device)
                val_pred = model(x_val)
                loss = criterion(val_pred, x_val.squeeze(1))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Pruning
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise TrialPruned()

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return model


# -----------------------------
# Score reconstruction
# -----------------------------
def compute_reconstruction_scores_STOC(model, X_tensor, device):
    model.eval()
    scores = []
    criterion = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for x in X_tensor:
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            loss = criterion(pred, x.squeeze(1)).mean().item()
            scores.append(loss)
    return np.array(scores)

# -----------------------------
# Seuil dynamique
# -----------------------------
def compute_dynamic_threshold(scores, n_test):
    mean = np.mean(scores)
    std = np.std(scores)
    coeff_threshold = np.sqrt(n_test) / np.log(n_test + 2)
    return mean + coeff_threshold * std

# -----------------------------
# Fonction finale
# -----------------------------
def outlier_detection_STOC(X_train, optuna_trials=30, timeout=600, return_scores=False):
    if isinstance(X_train, pd.DataFrame): X_train = X_train.values
    X_train = X_train.astype(np.float32)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = trial.suggest_int("epochs", 30, 100)

        # Train / Val split
        idx = np.random.permutation(len(X_train))
        split = int(0.8 * len(X_train))
        X_tr, X_val = X_train[idx[:split]], X_train[idx[split:]]

        train_loader = DataLoader(TensorDataset(torch.tensor(X_tr).unsqueeze(1)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val).unsqueeze(1)), batch_size=batch_size)

        model = STOC(input_dim, d_model=d_model, nhead=nhead, num_layers=num_layers)
        model = train_model_STOC(
                    model,
                    train_loader,
                    val_loader=val_loader,
                    trial=trial,
                    epochs=epochs,
                    lr=lr,
                    device=device
                )


        # Scores sur validation
        X_val_tensor = torch.tensor(X_val).unsqueeze(1).to(device)
        scores = compute_reconstruction_scores_STOC(model, X_val_tensor, device)
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=optuna_trials, timeout=timeout)

    # Récupérer les meilleurs hyperparamètres
    best_params = study.best_trial.params

    # Réentraîner avec les meilleurs paramètres sur TOUT X_train
    model = STOC(
        input_dim=input_dim,
        d_model=best_params["d_model"],
        nhead=best_params["nhead"],
        num_layers=best_params["num_layers"]
    ).to(device)

    full_loader = DataLoader(TensorDataset(torch.tensor(X_train).unsqueeze(1)), batch_size=best_params["batch_size"], shuffle=True)
    dummy_val_loader = DataLoader(TensorDataset(torch.tensor(X_train).unsqueeze(1)), batch_size=best_params["batch_size"])
    model = train_model_STOC(model, full_loader, dummy_val_loader, trial=study.best_trial,
                        epochs=best_params["epochs"], lr=best_params["lr"], device=device)

    # Reconstruction scores finaux
    X_tensor = torch.tensor(X_train).unsqueeze(1).to(device)
    scores = compute_reconstruction_scores_STOC(model, X_tensor, device)

    threshold = compute_dynamic_threshold(scores, len(X_train))
    anomalies = scores > threshold
    indices = np.where(anomalies)[0]

    if return_scores:
        return indices, scores, study
    return indices





















### DATN

class SeriesDecomposition(nn.Module):
    """
    Series decomposition block using moving average to split a time series into trend and seasonal components.
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.padding)

    def forward(self, x):
        # x: [B, d_model]
        trend = self.avg_pool(x)
        seasonal = x - trend
        return seasonal, trend


class AutoAttention(nn.Module):
    """
    Auto-attention mechanism using FFT to extract dominant periodic components.
    """
    def __init__(self, d_model, n_heads, window_size=192, c=4, device=torch.device('cpu')):
        super().__init__()
        self.d_model = d_model
        self.num_heads = n_heads
        self.device = device
        self.window_size = window_size

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        
        self.top_k_factor = int(c * np.log(window_size))
        self.attn = nn.MultiheadAttention(embed_dim=d_model*3, num_heads=n_heads, batch_first=True)
        self.linear = nn.Linear(d_model*3, d_model)

    def forward(self, x):
        B, _ = x.shape     
        x = x.to(self.device)       # [B, d_model]

        # Linear projections
        Q = self.to_q(x)       # [B, d_model]
        K = self.to_k(x)       # [B, d_model]
        V = self.to_v(x)       # [B, d_model]

        # Apply FFT on each
        FQ = fft.fft(Q, n=self.window_size, dim=1)       # [B, d_model]
        FK = fft.fft(K, n=self.window_size, dim=1)       # [B, d_model]
        FV = fft.fft(V, n=self.window_size, dim=1)       # [B, d_model]

        # Concatenate in feature dimension
        F_cat = torch.cat([FQ, FK, FV], dim=-1)  # [B, 3*d_model]
        amplitude = torch.abs(F_cat)

        # Select top-K frequency locations globally
        _, indices = torch.topk(amplitude, self.top_k_factor, dim=1)

        # Mask everything except top-K positions
        F_masked = torch.zeros_like(F_cat)
        for b in range(B):
            vect = F_cat[b]
            selected = torch.zeros_like(vect)
            selected[indices[b]] = vect[indices[b]]
            F_masked[b] = selected
        
        # Inverse FFT to time domain (retain dominant periods)
        periodic_component = fft.ifft(F_masked, n=self.window_size, dim=1).real      # [B, 3*d_model]

        # Apply standard multi-head self-attention
        attended, _ = self.attn(periodic_component, periodic_component, periodic_component)       # [B, 3*d_model]

        return self.linear(attended)       # [B, d_model]


class EncoderLayer(nn.Module):
    """
    A single encoder layer with decomposition and dual-path attention.
    """
    def __init__(self, d_model, n_heads, kernel_size, device, c=4, window_size=192):
        super().__init__()
        self.device = device
        self.decomp = SeriesDecomposition(kernel_size).to(device)
        self.auto_attn = AutoAttention(d_model, n_heads, device=device, c=c, window_size=window_size).to(device)
        self.mhsa = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True).to(device)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        ).to(device)
        self.norm = nn.LayerNorm(d_model).to(device)

    def forward(self, x):
        x = x.to(self.device)
        # Decompose into seasonal and trend
        seasonal, trend = self.decomp(x)        # [B, d_model] X 2
        seasonal = seasonal.to(self.device)
        trend = trend.to(self.device)

        # Auto-attention on both components
        s_seasonal = self.auto_attn(seasonal)
        s_trend = self.auto_attn(trend)

        # Multi-head self-attention and feedforward
        attn_s, _ = self.mhsa(s_seasonal, s_seasonal, s_seasonal)
        attn_t, _ = self.mhsa(s_trend, s_trend, s_trend)

        out_s = self.ffn(attn_s)
        out_t = self.ffn(attn_t)

        # Add the results
        return self.norm(out_s + out_t)


class DATN(nn.Module):
    """
    Complete DATN model with stacked encoder layers and linear decoder.
    """
    def __init__(self, input_dim, d_model, n_heads, num_layers, kernel_size, device, c=4, window_size=192):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.input_proj = nn.Linear(input_dim, d_model).to(device)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, kernel_size, device, c=c, window_size=window_size) for _ in range(num_layers)
        ]).to(device)
        self.output_proj = nn.Linear(d_model, input_dim).to(device)

    def forward(self, x):
        x = x.to(self.device)
        # x shape: [B, input_dim] => apply projection
        x = self.input_proj(x)      # [B, input_dim] → [B, d_model]
        
        # Each encoder layer depends on the previous one
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Decode
        x_hat = self.output_proj(x)        # → [B, input_dim]

        return x_hat


def compute_anomaly_reconstruction(original, reconstructed, device):
    """
    Compute anomaly scores as L2 norm between original and reconstructed signals.
    """
    return torch.norm(original.to(device) - reconstructed.to(device), dim=-1)  # shape: [B, T]





def train_model(model, train_loader, epochs=10, lr=1e-4, device=torch.device("cpu"), verbose=True):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # ---- Training Loop ----
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch[0].to(device)  # shape: [B, T]
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose: print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.6f}")


def compute_anomaly_scores(model, np_data, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(np_data, dtype=torch.float32).to(device)
        x_hat, S, P = model(x)
        recon_error = ((x - x_hat) ** 2).sum(dim=1).detach()
        ass_dis = association_discrepancy(P, S).detach()
        weights = torch.softmax(-ass_dis, dim=0)
        anomaly_scores = weights * recon_error
        return anomaly_scores.cpu().numpy()

def compute_dynamic_threshold(scores, n_test):
        mean = np.mean(scores)
        std = np.std(scores)
        coeff_threshold = np.sqrt(n_test) / np.log(n_test+2)
        return mean + coeff_threshold * std


def outlier_detection_DATN(X_train, batch_size=32, epochs=10, lr=1e-4, d_model=64, n_heads=4, num_layers=4, kernel_size=5, c=4, window_size=192, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=False, return_scores=False):
    if isinstance(X_train, pd.DataFrame): X_train = X_train.values
    n, p = X_train.shape

    # ---- Create DataLoader ----
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---- Initialize Model ----
    model = DATN(input_dim=p, d_model=d_model, n_heads=n_heads, num_layers=num_layers, kernel_size=kernel_size, c=c, window_size=window_size, device=device).to(device)

    # ---- Training Loop ----
    train_model(model, train_loader, epochs=epochs, lr=lr, device=device, verbose=verbose)

    # ---- Compute Reconstruction scores ---
    reconstruction_scores = compute_reconstruction_scores(model, X_train, device)

    # ---- Dynamic Thresholding ----
    threshold = compute_dynamic_threshold(reconstruction_scores, n)
    anomalies = reconstruction_scores > threshold
    outliers_indices = np.where(anomalies)[0]
    
    if return_scores:
        return outliers_indices, reconstruction_scores
    
    return outliers_indices

























### RINAT


# === Anomaly Attention & Transformer ===

class ReversibleInstanceNorm(nn.Module):
    """Applies reversible instance normalization (RevIN) to the input."""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = None  # as per the paper, not used

    def forward(self, x, reverse=False, stats=None):
        if not reverse:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + self.eps
            x_norm = (x - mean) / std
            return self.gamma.view(1, -1) * x_norm, (mean, std)
        else:
            mean, std = stats
            return x / self.gamma.view(1, -1) * std + mean



class RINAT_layer(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_sigma = nn.Linear(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x_norm, x_raw):
        B, _ = x_norm.shape      # [B, d_model] where B is the batch size

        ### Prior attention branch
        W_sigma = nn.Linear(self.d_model, self.n_heads)
        sigma = torch.abs(W_sigma(x_raw)) + 1e-5       # [B, H]
        positions = torch.arange(B, device=x_norm.device).unsqueeze(0)       # [1, B]
        prior = []
        for h in range(self.n_heads):
            dists = (positions.T - positions) ** 2      # [B, B]
            gauss = torch.exp(-dists[None, :, :] / (2 * sigma[:, h].unsqueeze(1) ** 2))
            gauss = gauss / (torch.sqrt(2 * math.pi * sigma[:, h].unsqueeze(1) ** 2))
            gauss = gauss / gauss.sum(dim=-1, keepdim=True)
            prior.append(gauss)
        P = torch.stack(prior, dim=1).squeeze(0)       # [H, B, B]

        ### Series attention branch
        Q = self.W_q(x_norm).view(B, self.n_heads, self.head_dim).transpose(0, 1)        # [H, B, d_model]
        K = self.W_k(x_norm).view(B, self.n_heads, self.head_dim).transpose(0, 1)        # [H, B, d_model]
        V = self.W_v(x_norm).view(B, self.n_heads, self.head_dim).transpose(0, 1)        # [H, B, d_model]

        S = torch.softmax(Q @ K.transpose(1,2) / (self.head_dim ** 0.5), dim=-1)       # [H, B, B]
        out = S @ V       # [H, B, d_model]

        ### Remaining transformations in the layer
        # Concatenate the heads along the last dimension
        out = out.transpose(1, 2).reshape(B, self.d_model)

        # Skip connection + Norm
        z = self.norm1(out + x_norm)
        
        # FeedForward + Skip + Norm
        z2 = self.ff(z)
        x_final = self.norm2(z2 + z)

        return x_final, S, P


class RINAT(nn.Module):
    def __init__(self, seq_len, d_model=512, num_layers=3, n_heads=8):
        super().__init__()
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.input_proj = nn.Linear(seq_len, d_model)
        self.revin = ReversibleInstanceNorm(d_model)
        # Stacked Transformer encoder
        self.rinat_layers = nn.ModuleList([
            RINAT_layer(d_model=d_model, n_heads=n_heads, seq_len=seq_len)
            for _ in range(num_layers)
        ])
        self.concat_layers = nn.Linear(d_model * num_layers, d_model)
        self.reconstruction = nn.Linear(d_model, seq_len)

    def forward(self, x):
        x = self.input_proj(x)  # [B, L] → [B, d_model]

        # Apply reversible instance normalization
        x_norm, stats = self.revin(x)

        rinat_outputs = [layer(x_norm, x) for layer in self.rinat_layers]

        list_x, list_P, list_S = zip(*rinat_outputs)      # [num_layers, B, d_model] X 3
        
        # Concatenate the outputs of all layers
        x_concat = torch.cat(list_x, dim=1)        # [B, num_layers*d_model]
        P = torch.cat(list_P, dim=0)        # [H*num_layers, B, B]
        S = torch.cat(list_S, dim=0)        # [H*num_layers, B, B]

        # Concatenate the outputs of all layers
        x_concat = self.concat_layers(x_concat)         # [B, d_model]

        # Apply inverse normalization
        x_concat_norm = self.revin(x_concat, reverse=True, stats=stats)

        # Reconstruction of the spectra
        x_hat = self.reconstruction(x_concat_norm)      # [B, seq_len]

        return x_hat, P, S



def association_discrepancy(P, S, eps=1e-8):
    P_safe = P + eps
    S_safe = S + eps

    # KL divergences par point et par head : [H, B, B] -> [H, B]
    kl_1 = F.kl_div(P_safe.log(), S_safe, reduction='none').sum(dim=1)
    kl_2 = F.kl_div(S_safe.log(), P_safe, reduction='none').sum(dim=1)

    # Moyenne sur les heads et les layers → [B]
    return (kl_1 + kl_2).mean(dim=0)


def compute_loss(x, x_hat, P, S, lam=3.0):
    recon_loss = F.mse_loss(x_hat, x, reduction='mean')
    ass_dis = association_discrepancy(P, S).mean()
    return recon_loss - lam * ass_dis, recon_loss, ass_dis


# === Training phase ===

def train_minimax_model(model, train_loader, epochs=10, lr= 1e-4, lam=3.0, device=torch.device('cpu'), verbose=True):
    model.train()
    model.to(device)
    if lr is None: optimizer = torch.optim.Adam(model.parameters())
    else : optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x in train_loader:
            batch_x = batch_x[0].to(device)  # [B, L]
            
            # Min phase
            x_hat, S, P = model(batch_x)
            loss_min, _, _ = compute_loss(batch_x, x_hat, P, S.detach(), lam=-lam)
            optimizer.zero_grad()
            loss_min.backward()
            optimizer.step()

            # Max phase
            x_hat, S, P = model(batch_x)
            loss_max, _, _ = compute_loss(batch_x, x_hat, P.detach(), S, lam=lam)
            optimizer.zero_grad()
            loss_max.backward()
            optimizer.step()

            total_loss += loss_max.item()

        if verbose: print(f"Epoch {epoch + 1}/{epochs} — Loss: {total_loss / len(train_loader):.4f}")


# === Definition of the score function ===

def compute_anomaly_scores(model, np_data, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(np_data, dtype=torch.float32).to(device)
        x_hat, S, P = model(x)
        recon_error = ((x - x_hat) ** 2).sum(dim=1).detach()
        ass_dis = association_discrepancy(P, S).detach()
        weights = torch.softmax(-ass_dis, dim=0)
        anomaly_scores = weights * recon_error
        return anomaly_scores.cpu().numpy()


def compute_dynamic_threshold(scores, n_test):
        mean = np.mean(scores)
        std = np.std(scores)
        coeff_threshold = np.sqrt(n_test) / np.log(n_test+2)
        return mean + coeff_threshold * std
    
    

def outlier_detection_RINAT(X_train, batch_size=32, epochs=10, lr=1e-4, lam=3.0, d_model=512, num_layers=3, n_heads=8, verbose=False, return_scores=False):
    if isinstance(X_train, pd.DataFrame): X_train = X_train.values
    X_test = X_train
    if isinstance(X_test, pd.DataFrame): X_test = X_test.values
    p = X_train.shape[1]

    # Create DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RINAT(seq_len=p, d_model=d_model, num_layers=num_layers, n_heads=n_heads).to(device)

    # Train the model
    train_minimax_model(model, train_loader, epochs=epochs, lr=lr, lam=lam, device=device, verbose=verbose)
    
    # Compute anomaly scores
    test_scores = compute_anomaly_scores(model, X_test, device=device)

    # Compute reconstruction scores
    reconstruction_scores = compute_reconstruction_scores(model, X_test, device)

    # Dynamic thresholding
    threshold = compute_dynamic_threshold(test_scores, len(X_test))

    anomalies = test_scores > threshold
    outliers_indices = np.where(anomalies)[0]

    if return_scores:
        return outliers_indices, reconstruction_scores
    
    return outliers_indices