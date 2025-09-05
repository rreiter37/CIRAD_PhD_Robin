import optuna
import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping


class LGBMOptuna(BaseEstimator, RegressorMixin):
    def __init__(self, cv=5, n_trials=50, random_state=42,
                 scoring='neg_mean_squared_error', verbose=0, verbose_optuna=False, best_trials=None, name_pp=None):
        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose
        self.verbose_optuna = verbose_optuna
        self.best_trials = best_trials
        self.name_pp = name_pp
        self.best_params_ = None
        self.best_model_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
            
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        if not self.verbose_optuna:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
                #'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 100, 1000),
                #'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 10),
                #'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                #'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                #'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1e-4, log=True),
                'random_state': self.random_state,
                'n_jobs': 1,
                'deterministic': True,
                'verbose': -1,
                'force_col_wise': True,
                'device':'gpu',
                'gpu_use_dp': True
            }

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            scores = []

            try:
                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    model = LGBMRegressor(**params)

                    model.fit(
                        X[train_idx], y[train_idx],
                        eval_set=[(X[val_idx], y[val_idx])],
                        eval_metric='l2',
                        callbacks=[early_stopping(stopping_rounds=10, verbose=False)],
                    )

                    preds = model.predict(X[val_idx])
                    score = -mean_squared_error(y[val_idx], preds)  # neg MSE

                    if np.isnan(score) or np.isinf(score):
                        raise ValueError(f"Invalid score ({score}) at fold {fold}")

                    scores.append(score)

                mean_score = np.mean(scores)
                return mean_score

            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return -np.inf

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name="LGBM_optuna",
            load_if_exists=False,
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=600, show_progress_bar=self.verbose > 0)

        self.best_params_ = study.best_params
        print('best parameters found : ', self.best_params_)
        self.best_params_.update({
            'random_state': self.random_state,
            'n_jobs': -1,
            'force_col_wise': True,
            'verbose': -1,
            'deterministic': True,
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
            'verbose_optuna': self.verbose_optuna,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
