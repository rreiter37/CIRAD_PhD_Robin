from sklearn.preprocessing import OneHotEncoder
import numpy as np

def correct_class_unbalances(X_train, Y_train, type_correction="duplicate", random_state=42):
    """
    Corrects class imbalance for PLS-DA preprocessing.

    Parameters
    ----------
    X_train : ndarray
        Training features.
    Y_train : ndarray
        Training labels.
    type_correction : str, default="duplicate"
        "ponderate" -> applies class weighting
        "duplicate" -> duplicates minority class samples to balance
    random_state : int, default=42
        Random seed for reproducibility (used in duplication)
    """

    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    Y_train_oh = encoder.fit_transform(Y_train.reshape(-1, 1))

    if type_correction == "ponderate":
        # ----- Weight-based correction -----
        classes, counts = np.unique(Y_train, return_counts=True)
        N = len(Y_train)
        class_weights = {cls: np.sqrt(N / count) for cls, count in zip(classes, counts)}

        sample_weights_train = np.array([class_weights[cls] for cls in Y_train])
        X_train = X_train * sample_weights_train[:, np.newaxis]
        Y_train = Y_train_oh * sample_weights_train[:, np.newaxis]

        return X_train, Y_train

    elif type_correction == "duplicate":
        # ----- Duplication-based correction -----
        rng = np.random.default_rng(random_state)
        classes, counts = np.unique(Y_train, return_counts=True)
        max_count = np.max(counts)

        X_train_balanced = []
        Y_train_balanced = []

        for cls in classes:
            X_cls = X_train[Y_train == cls]
            y_cls = Y_train_oh[Y_train == cls]

            if len(X_cls) < max_count:
                # Number of extra samples needed
                n_to_add = max_count - len(X_cls)
                idx_dup = rng.choice(len(X_cls), size=n_to_add, replace=True)
                X_cls = np.vstack([X_cls, X_cls[idx_dup]])
                y_cls = np.vstack([y_cls, y_cls[idx_dup]])

            X_train_balanced.append(X_cls)
            Y_train_balanced.append(y_cls)

        X_train_balanced = np.vstack(X_train_balanced)
        Y_train_balanced = np.vstack(Y_train_balanced)

        return X_train_balanced, Y_train_balanced

    else:
        raise ValueError("type_correction must be either 'ponderate' or 'duplicate'")
