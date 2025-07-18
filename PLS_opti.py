from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.exceptions import NotFittedError
import numpy as np

class ConstantTargetPLSError(Exception):
    """Exception levée si y est trop constant pour la PLS."""
    pass

class AutoPLSRegression(BaseEstimator, RegressorMixin):
    def __init__(self, max_components=100, cv=5, scale=True, scoring=None):
        self.max_components = max_components
        self.cv = cv
        self.scale = scale
        self.scoring = scoring

    def fit(self, X, y):
        y_arr = np.asarray(y)
        if np.isnan(y_arr).any():
            raise ValueError("y contains NaN values.")

        if np.all(np.std(y_arr, axis=0) < 1e-15):
            raise ConstantTargetPLSError("y is constant (null variance). Impossible to adjust a PLS model.")

        nb_spectra_cv = int(X.shape[0] * (self.cv - 1) / self.cv)
        max_comp = min(self.max_components, X.shape[1], nb_spectra_cv)
        param_grid = {'n_components': list(range(1, max_comp + 1))}

        pls = PLSRegression(scale=self.scale)
        scorer = self.scoring
        if scorer is None:
            scorer = make_scorer(mean_squared_error, greater_is_better=False)

        self.grid_ = GridSearchCV(pls, param_grid, scoring=scorer, cv=self.cv)
        self.grid_.fit(X, y)
        self.best_n_components_ = self.grid_.best_params_['n_components']
        print("Optimal number of components found for PLS : ", self.best_n_components_)
        self.best_model_ = self.grid_.best_estimator_
        return self

    def predict(self, X):
        if not hasattr(self, "best_model_"):
            raise NotFittedError("The model is still not trained.")
        return self.best_model_.predict(X)

    def score(self, X, y):
        if not hasattr(self, "best_model_"):
            raise NotFittedError("The model is still not trained.")
        return self.best_model_.score(X, y)
