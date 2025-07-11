import optuna
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

class RidgeOptuna(BaseEstimator, RegressorMixin):
    def __init__(self, cv=5, n_trials=30, random_state=None, scoring='neg_mean_squared_error'):
        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.scoring = scoring
        self.best_alpha_ = None
        self.best_model_ = None

    def fit(self, X, y):
        def objective(trial):
            alpha = trial.suggest_loguniform('alpha', 1e-4, 1e2)
            model = Ridge(alpha=alpha, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_alpha_ = study.best_params['alpha']
        self.best_model_ = Ridge(alpha=self.best_alpha_, random_state=self.random_state)
        self.best_model_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def get_params(self, deep=True):
        return {'cv': self.cv, 'n_trials': self.n_trials, 'random_state': self.random_state, 'scoring': self.scoring}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
