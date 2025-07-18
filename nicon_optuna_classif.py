import os
import optuna
import numpy as np
import random
import tensorflow as tf

from nirs4all.presets.ref_models import customizable_nicon_classification

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical # type: ignore
from optuna.integration import TFKerasPruningCallback
from optuna.pruners import HyperbandPruner

# Empêche TensorFlow d'utiliser le GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

# Importe ta fonction classification prête à l'emploi, ex:
# from ton_module import customizable_nicon_classification

class NiconOptunaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape=None, num_classes=2, n_trials=20, epochs=100, patience=10, epochs_optuna=100, verbose=0, random_state=42):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_trials = n_trials
        self.epochs = epochs
        self.patience = patience
        self.epochs_optuna = epochs_optuna
        self.verbose = verbose
        self.random_state = random_state
        self.study_ = None
        self.model_ = None
        self.best_params_ = None

    def _reshape(self, X):
        if len(X.shape) == 2:
            return np.array(X)[..., np.newaxis]
        return X

    def _suggest_params(self, trial, n_samples):
        max_batch = 2**(np.ceil(np.log2(n_samples)).astype(int) - 1)
        return {
            "batch_size": trial.suggest_categorical("batch_size", [max(1, k) for k in [max_batch // 32, max_batch // 16, max_batch // 8, max_batch // 4, max_batch // 2, max_batch]]),
            "spatial_dropout": trial.suggest_float("spatial_dropout", 0.0, 0.5),
            "filters1": trial.suggest_categorical("filters1", [4, 8, 16, 32, 64, 128, 256]),
            "kernel_size1": trial.suggest_categorical("kernel_size1", [3, 5, 7, 9, 11, 13, 15]),
            "strides1": trial.suggest_categorical("strides1", [1, 2, 3, 4, 5]),
            "activation1": trial.suggest_categorical("activation1", ['relu', 'selu', 'elu', 'swish']),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "filters2": trial.suggest_categorical("filters2", [4, 8, 16, 32, 64, 128, 256]),
            "kernel_size2": trial.suggest_categorical("kernel_size2", [3, 5, 7, 9, 11, 13, 15]),
            "strides2": trial.suggest_categorical("strides2", [1, 2, 3, 4, 5]),
            "activation2": trial.suggest_categorical("activation2", ['relu', 'selu', 'elu', 'swish']),
            "normalization_method1": trial.suggest_categorical("normalization_method1", ['BatchNormalization', 'LayerNormalization']),
            "filters3": trial.suggest_categorical("filters3", [4, 8, 16, 32, 64, 128, 256]),
            "kernel_size3": trial.suggest_categorical("kernel_size3", [3, 5, 7, 9, 11, 13, 15]),
            "strides3": trial.suggest_categorical("strides3", [1, 2, 3, 4, 5]),
            "activation3": trial.suggest_categorical("activation3", ['relu', 'selu', 'elu', 'swish']),
            "normalization_method2": trial.suggest_categorical("normalization_method2", ['BatchNormalization', 'LayerNormalization']),
            "dense_units": trial.suggest_categorical("dense_units", [4, 8, 16, 32, 64, 128, 256]),
            "dense_activation": trial.suggest_categorical("dense_activation", ['relu', 'selu', 'elu', 'swish']),
        }

    def _build_and_train(self, trial, X, y):
        set_global_seed(self.random_state)
        n_samples = X.shape[0]
        params = self._suggest_params(trial, n_samples)

        # Calculate total_steps for learning rate scheduling
        train_size = int(X.shape[0] * 0.8)
        batch_size = params["batch_size"]
        steps_per_epoch = int(np.ceil(train_size / batch_size))
        total_steps = self.epochs_optuna * steps_per_epoch


        model = customizable_nicon_classification(input_shape=self.input_shape, num_classes=self.num_classes, params=params)

        # Choix de la loss selon binaire ou multi-classes
        if self.num_classes == 2:
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            loss = "categorical_crossentropy"
            metrics = ["accuracy"]
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3,
            first_decay_steps=total_steps // 4,
            t_mul=1.0,
            m_mul=1.0,
            alpha=1e-6
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer,
                      loss=loss, metrics=metrics)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True, verbose=1),
            TFKerasPruningCallback(trial, "val_loss")
        ]

        history = model.fit(
            self._reshape(X), y,
            validation_split=0.2,
            epochs=self.epochs_optuna,
            batch_size=params["batch_size"],
            verbose=0,
            callbacks=callbacks,
        )

        return history.history["val_loss"][-1]

    def fit(self, X, y):
        set_global_seed(self.random_state)

        if self.input_shape is None:
            self.input_shape = (X.shape[1], 1) if len(X.shape) == 2 else X.shape[1:]

        # Pour multi-classes, s’assurer que y est one-hot encodé
        if self.num_classes > 2:
            y = to_categorical(y, num_classes=self.num_classes)

        def objective(trial):
            return self._build_and_train(trial, X, y)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = HyperbandPruner(min_resource=1, max_resource=self.epochs_optuna, reduction_factor=3)
        self.study_ = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        self.study_.optimize(objective, n_trials=self.n_trials, timeout=600)

        self.best_params_ = self.study_.best_params

        # Calculate total_steps for learning rate scheduling
        train_size = int(X.shape[0] * 0.8)
        batch_size = self.best_params_["batch_size"]
        steps_per_epoch = int(np.ceil(train_size / batch_size))
        total_steps = self.epochs * steps_per_epoch

        set_global_seed(self.random_state)
        self.model_ = customizable_nicon_classification(input_shape=self.input_shape, num_classes=self.num_classes, params=self.best_params_)

        if self.num_classes == 2:
            loss = "binary_crossentropy"
        else:
            loss = "categorical_crossentropy"

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3,
            first_decay_steps=total_steps // 4,  # 1 cycle = 1/4 of the total
            t_mul=1.0,        # number of double steps double in each cycle? No : Stays constant
            m_mul=1.0,        # constant minimum learning rate
            alpha=1e-6        
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model_.compile(optimizer=optimizer, loss=loss)

        self.model_.fit(
            self._reshape(X), y,
            epochs=self.epochs,
            batch_size=self.best_params_["batch_size"],
            verbose=self.verbose,
            callbacks=[EarlyStopping(monitor="loss", patience=self.patience, restore_best_weights=True, verbose=1)]
        )
        return self

    def predict(self, X):
        preds = self.model_.predict(self._reshape(X))
        if self.num_classes == 2:
            return (preds.flatten() > 0.5).astype(int)
        else:
            return np.argmax(preds, axis=1)

    def predict_proba(self, X):
        preds = self.model_.predict(self._reshape(X))
        if self.num_classes == 2:
            return np.concatenate([1 - preds, preds], axis=1)
        else:
            return preds

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_best_params(self):
        return self.best_params_
