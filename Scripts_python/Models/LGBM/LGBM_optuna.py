import optuna
import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping


class LGBMOptuna(BaseEstimator, RegressorMixin):
    def __init__(self, cv=5, n_trials=50, random_state=42,
                 scoring='neg_mean_squared_error', verbose=0, verbose_optuna=False, best_trials=None, name_pp=None, subsampling_rate=None):
        self.cv = cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose
        self.verbose_optuna = verbose_optuna
        self.best_trials = best_trials
        self.name_pp = name_pp
        self.subsampling_rate = subsampling_rate
        self.best_params_ = None
        self.best_model_ = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values.ravel()
            
        np.random.seed(self.random_state)
        random.seed(self.random_state)

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
                #'learning_rate': median_range('learning_rate', 0.7, 1.3),
                'max_depth': median_range('max_depth', 0.7, 1.3),
                #'num_leaves': median_range('num_leaves', 0.7, 1.3),
                'min_child_samples': median_range('min_child_samples', 0.7, 1.3),
                #'subsample': (max(0.5, df['subsample'].median() * 0.8), min(1.0, df['subsample'].median() * 1.2)),
                #'colsample_bytree': (max(0.5, df['colsample_bytree'].median() * 0.8), min(1.0, df['colsample_bytree'].median() * 1.2)),
                'reg_alpha': median_range('reg_alpha', 0.5, 2.0),
                #'reg_lambda': median_range('reg_lambda', 0.5, 2.0),
            }
        else:
            search_space = None  # full search

        if not self.verbose_optuna:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            if search_space is None:
                params = {
                    'n_estimators': 100,
                    'early_stopping_rounds': 20,
                    #'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                    'max_depth': trial.suggest_int('max_depth', 100, 1000),
                    #'num_leaves': trial.suggest_int('num_leaves', 1, 1000),
                    'min_child_samples': trial.suggest_int('min_child_samples', 1, 10),
                    #'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    #'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                    #'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1e-4, log=True),
                }
            else:
                # Narrowed space search
                params = {
                    'n_estimators': 100,
                    'early_stopping_rounds': 20,
                    #'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate']),
                    'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
                    #'num_leaves': trial.suggest_int('num_leaves', *search_space['num_leaves']),
                    'min_child_samples': trial.suggest_int('min_child_samples', *search_space['min_child_samples']),
                    #'subsample': trial.suggest_float('subsample', *search_space['subsample']),
                    #'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
                    'reg_alpha': trial.suggest_float('reg_alpha', *search_space['reg_alpha']),
                    #'reg_lambda': trial.suggest_float('reg_lambda', *search_space['reg_lambda']),
                }
            
            params.update({
                'random_state': self.random_state,
                'n_jobs': 1,
                'deterministic': True,
                'force_col_wise': True,                
                'verbose': -1,
                'device':'gpu',
                'gpu_use_dp': True
            })

            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            scores = []

            X_data, y_data = (X, y) if self.subsampling_rate is None else (X_optuna, y_optuna)

            try:
                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    X_train, X_val = X_data[train_idx], X_data[val_idx]
                    y_train, y_val = y_data[train_idx], y_data[val_idx]

                    model = LGBMRegressor(**params)

                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric='l2',
                        callbacks=[early_stopping(stopping_rounds=params['early_stopping_rounds'], verbose=False)],
                    )

                    preds = model.predict(X_val)
                    score = -mean_squared_error(y_val, preds)  # neg MSE

                    if np.isnan(score) or np.isinf(score):
                        raise ValueError(f"Invalid score ({score}) at fold {fold}")

                    scores.append(score)

                return np.mean(scores)

            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return -np.inf
            
        # === Run Optuna optimization ===
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=1000, reduction_factor=3)

        # Define and launch the optuna phase
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name="LGBM_optuna",
            load_if_exists=False,
        )
        study.optimize(objective, n_trials=self.n_trials, timeout=600, show_progress_bar=self.verbose_optuna)

        # Save best params and best trials
        self.best_params_ = study.best_params

        # Store the best hyperameters for next pp-model associations
        if self.best_trials is None:
            self.best_trials = [study.best_trial]
        else:
            self.best_trials.append(study.best_trial)

        self.best_params_.update({
            'n_estimators': 2000,
            'early_stopping_rounds': 100,
            'random_state': self.random_state,
            'n_jobs': -1,
            'force_col_wise': True,
            'verbose': -1,
            'deterministic': True,
        })

        # Define the model with optimal hyperparameters
        self.best_model_ = LGBMRegressor(**self.best_params_)

        # Train the model
        self.best_model_.fit(X, y,
                             eval_metric="neg_mean_squared_error",
                             eval_set=[(X, y)],
                             )
        
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
            'best_trials' : self.best_trials,
            'name_pp' : self.name_pp,
            'subsampling_rate' : self.subsampling_rate,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
