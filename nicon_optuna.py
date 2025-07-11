from sklearn.base import BaseEstimator, RegressorMixin
import optuna
from optuna.integration import TFKerasPruningCallback
from keras.callbacks import EarlyStopping 
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from nirs4all.presets.ref_models import nicon

from sklearn.base import BaseEstimator, RegressorMixin
from keras.callbacks import EarlyStopping
from optuna.integration import TFKerasPruningCallback
import optuna
import tensorflow as tf
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class NiconOptunaRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_shape=None,
        n_trials=20,
        epochs=100,
        patience=10,
        verbose=0,
        random_state=42,
    ):
        self.input_shape = input_shape
        self.n_trials = n_trials
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        self.study_ = None
        self.model_ = None
        self.best_params_ = None

    def _reshape(self, X):
        """Ajoute une dimension pour la compatibilité avec nicon."""
        if len(X.shape) == 2:
            if isinstance(X, pd.DataFrame):
                X_np = X.values
            else:
                X_np = np.array(X)
            return X_np[..., np.newaxis]
        return X

    def _build_and_train(self, trial, X, y):
        # Hyperparameter suggestions
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        # Build model
        params = {"learning_rate": learning_rate, "batch_size": batch_size}
        model = nicon(input_shape=self.input_shape, params=params)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="mse", metrics=["mae"])

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        ]
        try:
            callbacks.append(TFKerasPruningCallback(trial, "val_loss"))
        except Exception as e:
            print(f"[WARNING] Pruning callback skipped: {e}")

        history = model.fit(
            self._reshape(X), y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )

        return history.history["val_loss"][-1]

    def fit(self, X, y):
        if self.input_shape is None:
            self.input_shape = (X.shape[1], 1) if len(X.shape) == 2 else X.shape[1:]

        def objective(trial):
            return self._build_and_train(trial, X, y)

        self.study_ = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        self.study_.optimize(objective, n_trials=self.n_trials)

        # Final model with best hyperparameters
        self.best_params_ = self.study_.best_params
        best_lr = self.best_params_["learning_rate"]
        best_bs = self.best_params_["batch_size"]

        self.model_ = nicon(input_shape=self.input_shape, params=self.best_params_)
        self.model_.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
                            loss="mse")

        self.model_.fit(
            self._reshape(X), y,
            epochs=self.epochs,
            batch_size=best_bs,
            verbose=self.verbose,
            callbacks=[
                EarlyStopping(monitor="loss", patience=self.patience, restore_best_weights=True)
            ],
        )
        return self

    def predict(self, X):
        return self.model_.predict(self._reshape(X)).flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        return -np.mean((y - y_pred) ** 2)  # Score compatible avec sklearn (mse négatif)

    def get_best_params(self):
        return self.best_params_

