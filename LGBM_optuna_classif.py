import optuna
import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, KFold
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from optuna.integration import LightGBMPruningCallback

class LGBMOptunaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cv=5, n_trials=50, random_state=42, scoring='accuracy', verbose=0, timeout=600):
        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose
        self.timeout = timeout
        self.best_params_ = None
        self.best_model_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        def objective(trial):
            params = {
                'n_estimators': 10000,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': 1,
                'force_col_wise': True,
                'verbose': -1
            }

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            scores = []

            for train_idx, valid_idx in kf.split(X):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                model = LGBMClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric=self.scoring,
                    early_stopping_rounds=50,
                    callbacks=[LightGBMPruningCallback(trial, self.scoring)],
                )
                score = accuracy_score(y_valid, model.predict(X_valid))
                scores.append(score)

            return np.mean(scores)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=self.verbose > 0)

        self.best_params_ = study.best_params
        self.best_params_.update({
            'n_estimators': 10000,
            'random_state': self.random_state,
            'n_jobs': 1,
            'force_col_wise': True,
            'verbose': -1,
        })

        self.best_model_ = LGBMClassifier(**self.best_params_)
        self.best_model_.fit(
            X, y,
            eval_metric=self.scoring,
            early_stopping_rounds=50,
            eval_set=[(X, y)],
        )
        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        return self.best_model_.predict_proba(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            'cv': self.cv,
            'n_trials': self.n_trials,
            'random_state': self.random_state,
            'scoring': self.scoring,
            'verbose': self.verbose,
            'timeout': self.timeout,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
