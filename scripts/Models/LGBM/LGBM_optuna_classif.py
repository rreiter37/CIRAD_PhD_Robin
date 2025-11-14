import optuna
import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
from optuna.integration import LightGBMPruningCallback


class LGBMOptunaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cv=5, n_trials=50, random_state=42, verbose=0, verbose_optuna=False, timeout=600,
                 scoring="accuracy", direction_opt="minimize", subsampling_rate=None,
                 best_trials=None, name_pp=None):
        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.verbose = verbose
        self.verbose_optuna = verbose_optuna
        self.timeout = timeout
        self.scoring = scoring
        self.direction_opt = direction_opt
        self.subsampling_rate = subsampling_rate
        self.best_params_ = None
        self.best_model_ = None
        self.best_trials = best_trials  # previous trials for progressive optimization
        self.name_pp = name_pp  # preprocessing name (optional, for logging)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        # Determine the number of classes
        n_classes = len(np.unique(y))

        # Set the appropriate name of the evaluation metric, before training models
        if self.scoring=="accuracy":
            self.scoring = "binary_error" if n_classes==2 else "multi_error"
        elif self.scoring=="log_loss":
            self.scoring = "binary_logloss" if n_classes==2 else "multi_logloss"

        # === Subsample a given proportion of the data for Optuna optimization ===
        if self.subsampling_rate is not None:
            X_optuna, _, y_optuna, _ = train_test_split(X, y, train_size=self.subsampling_rate, stratify=y, random_state=self.random_state)

        # === If best_trials is provided, narrow the search space ===
        if self.best_trials is not None and len(self.best_trials) > 0:
            # Compute median of best params to define reduced bounds
            df = pd.DataFrame([t.params for t in self.best_trials])

            def median_range(col, low_factor=0.5, high_factor=1.5):
                val = df[col].median()
                if df[col].dtype.kind in 'i':  # integer hyperparameter
                    return max(1, int(val * low_factor)), int(val * high_factor)
                else:  # float hyperparameter
                    return float(val * low_factor), float(val * high_factor)

            search_space = {
                'learning_rate': median_range('learning_rate', 0.7, 1.3),
                'max_depth': median_range('max_depth', 0.7, 1.3),
                'num_leaves': median_range('num_leaves', 0.7, 1.3),
                'min_child_samples': median_range('min_child_samples', 0.7, 1.3),
                'subsample': (max(0.5, df['subsample'].median() * 0.8), min(1.0, df['subsample'].median() * 1.2)),
                'colsample_bytree': (max(0.5, df['colsample_bytree'].median() * 0.8), min(1.0, df['colsample_bytree'].median() * 1.2)),
                'reg_alpha': median_range('reg_alpha', 0.5, 2.0),
                'reg_lambda': median_range('reg_lambda', 0.5, 2.0),
            }
        else:
            search_space = None  # full search

        # === Define objective function ===
        def objective(trial):
            if search_space is None:
                # Full space search
                params = {
                    'n_estimators': 100,
                    'early_stopping_rounds': 20,
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'num_leaves': trial.suggest_int('num_leaves', 16, 128),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
                }
            else:
                # Narrowed space search
                params = {
                    'n_estimators': 100,
                    'early_stopping_rounds': 20,
                    'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate']),
                    'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                    'num_leaves': trial.suggest_int('num_leaves', *search_space['num_leaves']),
                    'min_child_samples': trial.suggest_int('min_child_samples', *search_space['min_child_samples']),
                    'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
                    'reg_alpha': trial.suggest_float('reg_alpha', *search_space['reg_alpha']),
                    'reg_lambda': trial.suggest_float('reg_lambda', *search_space['reg_lambda']),
                }

            params.update({
                'random_state': self.random_state,
                'n_jobs': 1,
                'force_col_wise': True,
                'verbose': -1
            })

            # Stratified K-Fold CV
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            scores = []

            X_data, y_data = (X, y) if self.subsampling_rate is None else (X_optuna, y_optuna)

            for train_idx, val_idx in kf.split(X_data, y_data):
                X_train, X_val = X_data[train_idx], X_data[val_idx]
                y_train, y_val = y_data[train_idx], y_data[val_idx]

                model = LGBMClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric=self.scoring,
                    callbacks=[LightGBMPruningCallback(trial, self.scoring)],
                )
                if self.scoring == "binary_logloss":
                    score = log_loss(y_val, model.predict_proba(X_val))
                else:
                    score = 1 - accuracy_score(y_val, model.predict(X_val))
                scores.append(score)

            return np.mean(scores)

        # === Run Optuna optimization ===
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=1000, reduction_factor=3)

        # Define and launch the optuna phase
        study = optuna.create_study(
            direction=self.direction_opt,
            sampler=sampler,
            pruner=pruner
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=self.verbose_optuna)

        # Save best params and best trials
        self.best_params_ = study.best_params

        # Store the best hyperameters for next pp-model associations
        if self.best_trials is None:
            self.best_trials = [study.best_trial]
        else:
            self.best_trials.append(study.best_trial)

        # Update final model params to complete necessary information
        self.best_params_.update({
            'n_estimators': 2000,
            'early_stopping_rounds': 100,
            'random_state': self.random_state,
            'n_jobs': 1,
            'force_col_wise': True,
            'verbose': -1,
        })

        # Define the model with optimal hyperparameters
        self.best_model_ = LGBMClassifier(**self.best_params_)

        # Train the model
        self.best_model_.fit(
            X, y,
            eval_metric=self.scoring,
            eval_set=[(X, y)],
        )

        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        return self.best_model_.predict_proba(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        if self.scoring == "binary_logloss":
            return log_loss(y, self.predict_proba(X))
        else:
            return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {
            'cv': self.cv,
            'n_trials': self.n_trials,
            'random_state': self.random_state,
            'scoring': self.scoring,
            'verbose': self.verbose,
            'timeout': self.timeout,
            'best_trials': self.best_trials,
            'name_pp': self.name_pp
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self