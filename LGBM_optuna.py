import optuna
import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score, KFold
from lightgbm import LGBMRegressor

class LGBMOptuna(BaseEstimator, RegressorMixin):
    def __init__(self, cv=5, n_trials=50, random_state=42, scoring='neg_mean_squared_error', verbose=0, verbose_optuna=False):
        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose
        self.verbose_optuna = verbose_optuna
        self.best_params_ = None
        self.best_model_ = None

    def fit(self, X, y):
        # Fix all randomness globally
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        if not self.verbose_optuna:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': 1,  # to avoid non deterministic parallelism
                'verbose': -1,  # to suppress warnings
                'force_col_wise': True,  # to force the column mode (more stable)
            }

            model = LGBMRegressor(**params)

            # Set a deterministic cross-validation
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=kf, scoring=self.scoring, n_jobs=1)

            return np.mean(scores)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, timeout=600, show_progress_bar=self.verbose > 0)

        self.best_params_ = study.best_params
        self.best_params_.update({
            'random_state': self.random_state,
            'n_jobs': 1,
            'force_col_wise': True,
            'verbose': -1,  # to suppress warnings
        })
        self.best_model_ = LGBMRegressor(**self.best_params_)
        self.best_model_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def get_params(self, deep=True):
        return {
            'cv': self.cv,
            'n_trials': self.n_trials,
            'random_state': self.random_state,
            'scoring': self.scoring,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
