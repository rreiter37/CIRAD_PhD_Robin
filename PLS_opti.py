from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.exceptions import NotFittedError
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np

class ConstantTargetPLSError(Exception):
    """Exception raised if y is too constant for the PLS."""
    pass

class AutoPLSRegression(BaseEstimator, RegressorMixin):
    def __init__(self, cv=5, scale=True,
                 scoring=None, seed=42, candidate_components=None):
        self.cv = cv
        self.scale = scale
        self.scoring = scoring
        self.seed = seed
        self.candidate_components = candidate_components

    def fit(self, X, y):
        y_arr = np.asarray(y)

        # Check for NaNs in y
        if np.isnan(y_arr).any():
            raise ValueError("y contains NaN values.")

        # Raise exception if y has near-zero variance
        if np.all(np.std(y_arr, axis=0) < 1e-15):
            raise ConstantTargetPLSError("y is constant (null variance). Cannot fit a PLS model.")
        
        n_wavelengths = X.shape[-1]
        # Determine valid upper limit for components
        if self.candidate_components is None:
            nb_spectra_cv = int(X.shape[0] * (self.cv - 1) / self.cv)
            global_max = min(n_wavelengths, nb_spectra_cv)
            self.candidate_components = np.linspace(1, global_max, global_max, dtype=int)

        # Define the scoring function (default: negative MSE)
        if self.scoring is None:
            scorer = make_scorer(mean_squared_error, greater_is_better=False)
        else:
            scorer = self.scoring

        # KFold CV (deterministic)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

        # Objective function for Hyperopt
        def objective(params):
            n_components = int(params['n_components'])
            model = PLSRegression(n_components=n_components, scale=self.scale)

            scores = cross_val_score(model, X, y, cv=kf, scoring=scorer, n_jobs=-1)
            avg_loss = -np.mean(scores)  # Convert to positive MSE
            return {'loss': avg_loss, 'status': STATUS_OK}

        # Search space: quantized uniform integer in component_range
        space = hp.choice('n_components', self.candidate_components)

        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            rstate=np.random.default_rng(self.seed)
        )

        self.best_n_components_ = int(best['n_components'])
        print(f"Optimal number of components found for PLS: {self.best_n_components_}")

        self.best_model_ = PLSRegression(n_components=self.best_n_components_, scale=self.scale)
        self.best_model_.fit(X, y)

        return self

    def predict(self, X):
        if not hasattr(self, "best_model_"):
            raise NotFittedError("The model is still not trained.")
        return self.best_model_.predict(X)

    def score(self, X, y):
        if not hasattr(self, "best_model_"):
            raise NotFittedError("The model is still not trained.")
        return self.best_model_.score(X, y)
