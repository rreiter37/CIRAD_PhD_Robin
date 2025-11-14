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