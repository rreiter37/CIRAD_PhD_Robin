import numpy as np
import random
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

class RidgeCVRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alphas=None, cv=5, scoring='neg_mean_squared_error', random_state=42):
        self.alphas = alphas if alphas is not None else np.logspace(-4, 2, 50)
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_model_ = None

    def fit(self, X, y):
        set_global_seed(self.random_state)

        # Définir un KFold déterministe
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        self.best_model_ = RidgeCV(alphas=self.alphas, cv=kf, scoring=self.scoring)
        self.best_model_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def get_params(self, deep=True):
        return {
            'alphas': self.alphas,
            'cv': self.cv,
            'scoring': self.scoring,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
