from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss
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
        # Determine valid upper limit for components
        if self.candidate_components is None:
            nb_spectra_cv = int(X.shape[0] * (self.cv - 1) / self.cv)
            global_max = min(n_wavelengths, nb_spectra_cv)
            self.candidate_components = np.linspace(1, global_max, global_max, dtype=int)

        self.label_binarizer_ = LabelBinarizer()
        y_bin = self.label_binarizer_.fit_transform(y)

        # Always force y_bin to have shape (n_samples, n_classes)
        n_classes = len(self.label_binarizer_.classes_)
        if y_bin.ndim == 1:
            y_bin = y_bin.reshape(-1, 1)
        if y_bin.shape[1] < n_classes:
            missing_cols = n_classes - y_bin.shape[1]
            y_bin = np.hstack([y_bin, np.zeros((y_bin.shape[0], missing_cols))])

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

        def run_fold(train_idx, test_idx, model):
            try:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_bin[train_idx], y_bin[test_idx]  # log_loss expects binarized y
                model.fit(X_train, y_train)
                y_scores = model.predict(X_test)
                # Clip and normalize to get valid probabilities
                if y_scores.ndim == 1 or y_scores.shape[1] == 1:
                    y_prob = np.clip(y_scores.reshape(-1, 1), 0, 1)
                    y_prob = np.hstack([1 - y_prob, y_prob])
                else:
                    y_scores = np.clip(y_scores, 0, None)
                    row_sums = y_scores.sum(axis=1, keepdims=True)
                    y_prob = y_scores / np.maximum(row_sums, 1e-8)
                return log_loss(y_test, y_prob)
            except Exception:
                return np.inf  # strong penalty to avoid interfering with actual loss values

        def objective(n_components):
            model = PLSRegression(n_components=int(n_components), scale=self.scale)

            if self.parallelism:
                losses = Parallel(n_jobs=-1)(
                    delayed(run_fold)(train_idx, test_idx, model)
                    for train_idx, test_idx in kf.split(X)
                )
            else:
                losses = [run_fold(train_idx, test_idx, model) for train_idx, test_idx in kf.split(X)]

            avg_loss = np.mean(losses)
            return {'loss': avg_loss, 'status': STATUS_OK}
        statut_parallel = "in parallel" if self.parallelism else "sequentially"
        print(f"[INFO] Execution of PLSDA {self.cv}-fold cv in {statut_parallel}.")
        space = hp.choice('n_components', self.candidate_components)
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            rstate=np.random.default_rng(self.seed)
        )
        self.best_n_components_ = int(best)
        print(f"[AutoPLSDAClassifier] Optimal number of components: {self.best_n_components_}")

        self.best_model_ = PLSRegression(n_components=self.best_n_components_, scale=self.scale)
        self.best_model_.fit(X, y_bin)
        return self

    def predict(self, X):
        if not hasattr(self, "best_model_"):
            raise NotFittedError("Model is not fitted yet.")

        y_scores = self.best_model_.predict(X)

        # Ensure 2D
        if y_scores.ndim == 1:
            y_scores = y_scores.reshape(-1, 1)

        n_classes = len(self.label_binarizer_.classes_)
        if y_scores.shape[1] < n_classes:
            missing_cols = n_classes - y_scores.shape[1]
            y_scores = np.hstack([y_scores, np.zeros((y_scores.shape[0], missing_cols))])

        # Binary classification
        if n_classes == 2:
            return self.label_binarizer_.inverse_transform(y_scores > 0.5)

        # Multi-class: avoid inverse_transform errors, use argmax
        return self.label_binarizer_.classes_[np.argmax(y_scores, axis=1)]

    def predict_proba(self, X):
        if not hasattr(self, "best_model_"):
            raise NotFittedError("Model is not fitted yet.")
        y_scores = self.best_model_.predict(X)
        if y_scores.ndim == 1 or y_scores.shape[1] == 1:
            y_prob = np.clip(y_scores.reshape(-1, 1), 0, 1)
            return np.hstack([1 - y_prob, y_prob])
        else:
            y_scores = np.clip(y_scores, 0, None)
            row_sums = y_scores.sum(axis=1, keepdims=True)
            return y_scores / np.maximum(row_sums, 1e-8)

    def score(self, X, y):
        y_bin = self.label_binarizer_.transform(y)
        y_prob = self.predict_proba(X)
        return -log_loss(y_bin, y_prob)  # positive score (to maximize)