from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
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
        
        # Determine valid upper limit for components
        if self.candidate_components is None:
            nb_spectra_cv = int(X.shape[0] * (self.cv - 1) / self.cv)
            global_max = min(n_wavelengths, nb_spectra_cv)
            self.candidate_components = np.linspace(1, global_max, global_max, dtype=int)
        
        self.candidate_components = list(self.candidate_components)

        self.label_binarizer_ = LabelBinarizer()
        y_bin = self.label_binarizer_.fit_transform(y)
        n_classes = len(self.label_binarizer_.classes_)

        # Ensure shape (n_samples, n_classes)
        if y_bin.ndim == 1:
            y_bin = y_bin.reshape(-1, 1)
        if y_bin.shape[1] < n_classes:
            missing_cols = n_classes - y_bin.shape[1]
            y_bin = np.hstack([y_bin, np.zeros((y_bin.shape[0], missing_cols))])

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

        def run_fold(train_idx, test_idx, n_components):
            try:
                model = PLSRegression(n_components=int(n_components), scale=self.scale)
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y_bin[train_idx], y_bin[test_idx]
                
                model.fit(X_train, y_train)
                y_scores = model.predict(X_test)

                # Convert scores to predicted classes
                if n_classes == 2:
                    y_pred = (y_scores[:, 0] > 0.5).astype(int)
                    y_true = y_test[:, 0]
                else:
                    y_pred = np.argmax(y_scores, axis=1)
                    y_true = np.argmax(y_test, axis=1)

                return accuracy_score(y_true, y_pred)
            except Exception:
                return 0.0  # Penalize failures

        def objective(n_components):
            if self.parallelism:
                accs = Parallel(n_jobs=-1)(
                    delayed(run_fold)(train_idx, test_idx, n_components)
                    for train_idx, test_idx in kf.split(X)
                )
            else:
                accs = [run_fold(train_idx, test_idx, n_components) for train_idx, test_idx in kf.split(X)]

            avg_acc = np.mean(accs)

            return {'loss': -avg_acc, 'status': STATUS_OK}  # negative for maximization
        
        statut_parallel = "in parallel" if self.parallelism else "sequentially"
        print(f"[INFO] Execution of PLSDA {self.cv}-fold cv in {statut_parallel}.")

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

        self.best_model_ = PLSRegression(n_components=self.best_n_component_, scale=self.scale)
        self.best_model_.fit(X, y_bin)
        return self

    def predict(self, X):
        if not hasattr(self, "best_model_"):
            raise NotFittedError("Model is not fitted yet.")
        y_scores = self.best_model_.predict(X)
        if y_scores.ndim == 1:
            y_scores = y_scores.reshape(-1, 1)

        n_classes = len(self.label_binarizer_.classes_)
        if n_classes == 2:
            y_pred = (y_scores[:, 0] > 0.5).astype(int)
            return self.label_binarizer_.inverse_transform(y_pred)
        else:
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
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)