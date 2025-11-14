from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from joblib import Parallel, delayed
import numpy as np

class AutoPLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cv=5, scale=True, 
                 scoring=None, seed=42, candidate_components=None, parallelism=False):
        self.cv = cv
        self.scale = scale
        self.scoring = scoring
        self.seed = seed
        self.candidate_components = candidate_components
        self.parallelism = parallelism

    def fit(self, X, y):
        n_wavelengths = X.shape[-1]

        # Déterminer le nombre max de composantes possibles
        if self.candidate_components is None:
            nb_spectra_cv = int(X.shape[0] * (self.cv - 1) / self.cv)
            global_max = min(n_wavelengths, nb_spectra_cv)
            self.candidate_components = np.linspace(1, global_max, global_max, dtype=int)

        self.candidate_components = list(self.candidate_components)

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

        # Détecter si y est déjà one-hot ou pas
        if y.ndim == 2 and y.shape[1] > 1:
            # One-hot → on retrouve les labels
            y_labels = np.argmax(y, axis=1)
            classes_ = np.arange(y.shape[1])
            n_classes = y.shape[1]
            y_is_onehot = True
        else:
            y_labels = y
            classes_ = np.unique(y_labels)
            n_classes = len(classes_)
            y_is_onehot = False

        def run_fold(train_idx, test_idx, n_components):
            try:
                model = PLSRegression(n_components=int(n_components), scale=self.scale)
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                y_train_labels = y_labels[train_idx]
                y_test_labels = y_labels[test_idx]

                # Si y est déjà one-hot → pas besoin de réencoder
                if y_is_onehot:
                    y_train_pls = y_train
                else:
                    if n_classes == 2:
                        y_train_pls = (y_train_labels == classes_[1]).astype(float).reshape(-1, 1)
                    else:
                        y_train_pls = np.eye(n_classes)[
                            np.searchsorted(classes_, y_train_labels)
                        ]

                model.fit(X_train, y_train_pls)
                y_scores = model.predict(X_test)

                # Décodage des prédictions
                if n_classes == 2:
                    y_pred = (y_scores[:, 0] > 0.5).astype(int)
                    y_true = (y_test_labels == classes_[1]).astype(int)
                else:
                    y_pred = np.argmax(y_scores, axis=1)
                    y_true = y_test_labels

                return accuracy_score(y_true, y_pred)
            except Exception:
                return 0.0  # penalisation en cas d'erreur

        def objective(n_components):
            if self.parallelism:
                accs = Parallel(n_jobs=-1)(
                    delayed(run_fold)(train_idx, test_idx, n_components)
                    for train_idx, test_idx in kf.split(X)
                )
            else:
                accs = [run_fold(train_idx, test_idx, n_components)
                        for train_idx, test_idx in kf.split(X)]
            avg_acc = np.mean(accs)
            return {'loss': -avg_acc, 'status': STATUS_OK}

        statut_parallel = "in parallel" if self.parallelism else "sequentially"
        print(f"[INFO] Execution of PLSDA {self.cv}-fold CV {statut_parallel}.")

        space = hp.choice('n_components', self.candidate_components)
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=len(self.candidate_components),
            rstate=np.random.default_rng(self.seed)
        )

        best_idx = best['n_components']
        self.best_n_component_ = int(self.candidate_components[best_idx])
        print(f"[AutoPLSDAClassifier] Optimal number of components: {self.best_n_component_}")

        # Entraînement final sur tout le jeu
        if y_is_onehot:
            y_pls = y
        else:
            if n_classes == 2:
                y_pls = (y_labels == classes_[1]).astype(float).reshape(-1, 1)
            else:
                y_pls = np.eye(n_classes)[np.searchsorted(classes_, y_labels)]

        self.classes_ = classes_
        self.best_model_ = PLSRegression(n_components=self.best_n_component_, scale=self.scale)
        self.best_model_.fit(X, y_pls)
        return self

    def predict(self, X):
        if not hasattr(self, "best_model_"):
            raise NotFittedError("Model is not fitted yet.")
        y_scores = self.best_model_.predict(X)

        # Classification binaire
        if len(self.classes_) == 2:
            if y_scores.ndim == 2 and y_scores.shape[1] > 1:
                # Prendre la classe avec la plus grande valeur
                y_pred_idx = np.argmax(y_scores, axis=1)
            else:
                # Seuil à 0.5 sur la première colonne
                y_pred_idx = (y_scores[:, 0] > 0.5).astype(int)
        else:
            # Multi-classes
            y_pred_idx = np.argmax(y_scores, axis=1)

        # Retourne un vecteur 1D d'étiquettes (pas de multi-output)
        return np.array([self.classes_[i] for i in y_pred_idx])

    def predict_proba(self, X):
        if not hasattr(self, "best_model_"):
            raise NotFittedError("Model is not fitted yet.")
        y_scores = self.best_model_.predict(X)
        if len(self.classes_) == 2:
            y_prob = np.clip(y_scores[:, 0], 0, 1)
            return np.vstack([1 - y_prob, y_prob]).T
        else:
            y_scores = np.clip(y_scores, 0, None)
            row_sums = y_scores.sum(axis=1, keepdims=True)
            return y_scores / np.maximum(row_sums, 1e-8)

    def score(self, X, y):
        y_pred = self.predict(X)
        score_value = accuracy_score(y, y_pred)
        return score_value