import os
import optuna
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, log_loss
from Scripts_python.Models.nicon_classif_pytorch import customizable_nicon_classification

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _reshape(self, X):
        if len(X.shape) == 2:
            return X[:, :, np.newaxis]
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

    def _train_model(self, model, train_loader, val_loader, loss_fn, optimizer, trial):
        best_loss = np.inf
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs_optuna):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        model.load_state_dict(best_state)
        return best_loss

    def _build_and_train(self, trial, X, y):
        set_global_seed(self.random_state)
        params = self._suggest_params(trial, X.shape[0])
        model = customizable_nicon_classification(input_shape=self.input_shape, num_classes=self.num_classes, params=params).to(self.device)

        y_tensor = torch.tensor(y, dtype=torch.long if self.num_classes > 2 else torch.float32)
        X_tensor = torch.tensor(self._reshape(X), dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)

        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(self.random_state))

        train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=False)

        if self.num_classes == 2:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        return self._train_model(model, train_loader, val_loader, loss_fn, optimizer, trial)

    def fit(self, X, y):
        set_global_seed(self.random_state)
        if self.input_shape is None:
            self.input_shape = (X.shape[1], 1) if len(X.shape) == 2 else X.shape[1:]

        def objective(trial):
            return self._build_and_train(trial, X, y)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=self.epochs_optuna, reduction_factor=3)
        self.study_ = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        self.study_.optimize(objective, n_trials=self.n_trials, timeout=600)
        self.best_params_ = self.study_.best_params

        # Entraînement final
        X_tensor = torch.tensor(self._reshape(X), dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long if self.num_classes > 2 else torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.best_params_["batch_size"], shuffle=True)

        self.model_ = customizable_nicon_classification(input_shape=self.input_shape, num_classes=self.num_classes, params=self.best_params_).to(self.device)
        optimizer = optim.Adam(self.model_.parameters(), lr=1e-3)
        if self.num_classes == 2:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        self._train_model(self.model_, train_loader, train_loader, loss_fn, optimizer, trial=optuna.trial.FixedTrial({}))

        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_tensor = torch.tensor(self._reshape(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            if self.num_classes == 2:
                probs = torch.sigmoid(logits).cpu().numpy()
                return np.stack([1 - probs, probs], axis=1)
            else:
                return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_best_params(self):
        return self.best_params_
