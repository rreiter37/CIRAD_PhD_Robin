import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from Scripts_python.Models.nicon_classif_pytorch import customizable_nicon_classification
from Scripts_python.utils.checkpointing_logger import CheckpointLoggerCallback
from max_batch_size import find_max_batch_size


# Custom Optuna pruning callback for PyTorch Lightning
class CustomOptunaPruningCallback(Callback):
    def __init__(self, trial, monitor="val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()
        self.trial.report(current_score, step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

# Set global seed for reproducibility
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NiconPLClassifier(pl.LightningModule):
    def __init__(self, input_shape, num_classes, params, lr_max=1e-3, lr_min=1e-6, epochs=100, t0_steps=None, cyclic_learning=True):
        super().__init__()
        self.save_hyperparameters(ignore=["t0_steps", "cyclic_learning"])
        self.params = params
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epochs = epochs
        self.t0_steps = t0_steps
        self.cyclic_learning = cyclic_learning
        self.num_classes = num_classes

        self.model = customizable_nicon_classification(
            input_shape=input_shape,
            num_classes=num_classes,
            params=params
        )
        self.criterion = nn.CrossEntropyLoss() if num_classes > 2 else nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_max)

        if self.cyclic_learning:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.t0_steps if self.t0_steps else max(1, self.epochs // 4),
                T_mult=1,
                eta_min=self.lr_min
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }
        else:
            return optimizer


# Main Optuna classifier class
class NiconOptunaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape=None, num_classes=2, n_trials=20, epochs=100, patience=10, epochs_optuna=10, batch_size=None,
                 verbose=0, verbose_optuna=False, timeout=600, random_state=42, cyclic_learning=True, lr_max=1e-3, lr_min=1e-6, 
                 device=None, get_logger=True, get_logger_optuna=False, best_trials=None, name_pp=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_trials = n_trials
        self.epochs = epochs
        self.patience = patience
        self.epochs_optuna = epochs_optuna
        self.batch_size = batch_size
        self.verbose = verbose
        self.verbose_optuna = verbose_optuna
        self.timeout = timeout
        self.random_state = random_state
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.study_ = None
        self.best_params_ = None
        self.model_ = None
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger = get_logger
        self.get_logger_optuna = get_logger_optuna
        self.cyclic_learning = cyclic_learning
        self.best_trials = best_trials
        self.name_pp = name_pp

    def _reshape(self, X):
        if len(X.shape) == 2:
            return X[:, :, np.newaxis]
        return X

    def _suggest_params(self, trial):
        best_trials = self.best_trials
        def median_and_range(param_name, type, low, high, log=False, step=2, scale=0.4):
            if best_trials is None:
                if type=='int':
                    return trial.suggest_int(param_name, low, high, step=step)
                else:
                    return trial.suggest_float(param_name, low, high, log=log)
                
            values = [t[param_name] for t in best_trials if param_name in t.keys()]

            if not values:
                if type=='int':
                    return trial.suggest_int(param_name, low, high, step=step)
                else:
                    return trial.suggest_float(param_name, low, high, log=log)
                
            median = np.median(values)

            if type=='int':
                median = int(median)
                delta = int((high - low) * scale/2)
                bounded_low = max(low, median - delta)
                bounded_high = min(high, median + delta)
                return trial.suggest_int(param_name, bounded_low, bounded_high, step = max(1, step//2))
            else:
                delta = (high - low) * scale/2
                bounded_low = max(low, median - delta)
                bounded_high = min(high, median + delta)
                return trial.suggest_float(param_name, bounded_low, bounded_high, log=log)
            
        return {
            "spatial_dropout": median_and_range("spatial_dropout", "float", 0.0, 0.5, scale=0.2),
            "filters1": 2** median_and_range("filters1_power", "int", 2, 8, step=1, scale=0.4),
            "kernel_size1": median_and_range("kernel_size1", "int", 3, 25, step=2, scale=0.4),
            "strides1": median_and_range("strides1", "int", 1, 5, step=1, scale=0.4),
            "activation1": trial.suggest_categorical("activation1", ['relu', 'selu', 'elu', 'swish']),
            "dropout_rate": median_and_range("dropout_rate", "float", 0.01, 0.5, log=True, scale=0.2),
            "filters2": 2** median_and_range("filters2_power", "int", 2, 8, step=1, scale=0.4),
            "kernel_size2": median_and_range("kernel_size2", "int", 3, 25, step=2, scale=0.4),
            "strides2": median_and_range("strides2", "int", 1, 5, step=1, scale=0.4),
            "activation2": trial.suggest_categorical("activation2", ['relu', 'selu', 'elu', 'swish']),
            "normalization_method1": trial.suggest_categorical("normalization_method1", ['BatchNormalization', 'LayerNormalization']),
            "filters3": 2** median_and_range("filters3_power", "int", 2, 8, step=1, scale=0.4),
            "kernel_size3": median_and_range("kernel_size3", "int", 3, 25, step=2, scale=0.4),
            "strides3": median_and_range("strides3", "int", 1, 5, step=1, scale=0.4),
            "activation3": trial.suggest_categorical("activation3", ['relu', 'selu', 'elu', 'swish']),
            "normalization_method2": trial.suggest_categorical("normalization_method2", ['BatchNormalization', 'LayerNormalization']),
            "dense_units": 2** median_and_range("dense_units_power", "int", 2, 8, step=1, scale=0.4),
            "dense_activation": trial.suggest_categorical("dense_activation", ['relu', 'selu', 'elu', 'swish']),
            }

    def _build_and_train(self, trial, X, y):
        set_global_seed(self.random_state)
        params = self._suggest_params(trial, X.shape[0])
        X_tensor = torch.tensor(self._reshape(X), dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(self.random_state))
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        model = NiconPLClassifier(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            params=params,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=self.epochs_optuna,
            t0_steps=len(train_loader),
            cyclic_learning=self.cyclic_learning
        )

        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=self.patience),
        ]

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            enable_progress_bar=False,
            logger=False,
            deterministic=True,
            enable_model_summary=False
        )

        trainer.fit(model, train_loader, val_loader)
        val_loss = trainer.callback_metrics["val_loss"].item()

        trial.set_user_attr("best_model_state_dict", model.state_dict())
        return val_loss

    def fit(self, X, y):
        set_global_seed(self.random_state)
        if self.input_shape is None:
            self.input_shape = (X.shape[1], 1) if len(X.shape) == 2 else X.shape[1:]
        
        n_samples = len(X)

        # Find the maximum batch size accepted by the GPU
        if self.batch_size is None:
            params = {"kernel_size1": 3, "kernel_size2": 3, "kernel_size3": 3, "spatial_dropout": 0.01, "dropout_rate": 0.01}
            params["output_dim"] = 1
            model = NiconPLClassifier(
                input_shape=self.input_shape[0],
                num_classes=self.num_classes,
                params=params,
                lr_max=self.lr_max,
                lr_min=self.lr_min,
                epochs=1,
                t0_steps=n_samples,
                cyclic_learning=self.cyclic_learning
            )
            self.batch_size = find_max_batch_size(model=model, input_shape=self.input_shape, device=self.device, max_batch=X.shape[-1], min_batch=1)
            print("Maximum batch size found : ", self.batch_size)

        def objective(trial):
            return self._build_and_train(trial, X, y)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=self.epochs, reduction_factor=3)
        self.study_ = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        self.study_.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        self.best_params_ = self.study_.best_trial.user_attrs["best_model_state_dict"]
        best_trial_params = self.study_.best_trial.params.copy()

        # Train final model
        self.model_ = NiconPLClassifier(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            params=best_trial_params,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=self.epochs,
            t0_steps=None,
            cyclic_learning=self.cyclic_learning
        )
        self.model_.load_state_dict(self.best_params_)
        self.model_.to(self.device)
        self.model_.eval()
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
        return self.study_.best_trial.params
