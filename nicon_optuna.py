import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

import sys
import os

# Désactiver Fabric
os.environ["PL_DISABLE_FABRIC"] = "1"

# Capturer la sortie standard et erreur pendant l'import de pytorch_lightning
class DummyFile(object):
    def write(self, x): pass
    def flush(self): pass

sys.stdout = DummyFile()
sys.stderr = DummyFile()

import pytorch_lightning as pl

# Remettre la sortie standard normale
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Supprimer rank_zero_info au cas où (pour la suite)
import pytorch_lightning.utilities.rank_zero as rank_zero
rank_zero.rank_zero_info = lambda *args, **kwargs: None
rank_zero.rank_zero_warn = lambda *args, **kwargs: None
rank_zero.rank_zero_debug = lambda *args, **kwargs: None


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import optuna
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error


from Models.nicon_custom_pytorch import CustomizableNicon

from pytorch_lightning.callbacks import Callback


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

# Fix seed for reproducibility
def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class NiconPLModule(pl.LightningModule):
    def __init__(self, input_channels, params, lr_max, lr_min, epochs):
        super().__init__()
        self.model = CustomizableNicon(input_channels=input_channels, params=params)
        self.criterion = nn.MSELoss()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epochs = epochs

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x).squeeze(1)  # s'assurer que sortie = (batch,)

    def training_step(self, batch, batch_idx):
        x, y = batch
        try:
            y_pred = self(x)
            loss = self.criterion(y_pred, y)
        except RuntimeError as e:
            print("\n--- RuntimeError during training_step ---")
            print(f"Exception: {e}")
            print(f"x device: {x.device}, y device: {y.device}")
            print(f"x shape: {x.shape}, y shape: {y.shape}")
            print(f"Model is on device: {next(self.model.parameters()).device}")
            for name, param in self.model.named_parameters():
                print(f"{name} -> {param.device}")
            raise e
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        try:
            y_pred = self(x)
            loss = self.criterion(y_pred, y)
        except RuntimeError as e:
            print("\n--- RuntimeError during validation_step ---")
            print(f"Exception: {e}")
            print(f"x device: {x.device}, y device: {y.device}")
            print(f"x shape: {x.shape}, y shape: {y.shape}")
            print(f"Model is on device: {next(self.model.parameters()).device}")
            for name, param in self.model.named_parameters():
                print(f"{name} -> {param.device}")
            raise e
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_max)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.epochs // 4,
            T_mult=1,
            eta_min=self.lr_min,
        )
        return [optimizer], [scheduler]


class NiconOptunaRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape=None, n_trials=20, epochs=100, patience=10,
                 epochs_optuna=100, verbose=0, verbose_optuna=False, random_state=42, device=None):
        self.input_shape = input_shape
        self.n_trials = n_trials
        self.epochs = epochs
        self.patience = patience
        self.epochs_optuna = epochs_optuna
        self.verbose = verbose
        self.verbose_optuna = verbose_optuna
        self.random_state = random_state
        self.study_ = None
        self.model_ = None
        self.best_params_ = None
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    def _reshape(self, X):
        X = np.array(X)
        if len(X.shape) == 2:
            # (samples, features) -> (samples, channels=1, features)
            X = X[:, np.newaxis, :]
        return torch.tensor(X, dtype=torch.float32)

    def _suggest_params(self, trial, n_samples):
        max_batch = int(2**(int(np.ceil(np.log2(n_samples))) - 1))
        if self.device == "cuda":
            params_batch = [max_batch // 4, max_batch // 2, max_batch, max_batch * 2]
        else:
            params_batch = [max_batch // 32, max_batch // 16, max_batch // 8, max_batch // 4] 
        return {
            "batch_size": trial.suggest_categorical("batch_size", [int(max(1, k)) for k in params_batch]),
            "kernel_size1": trial.suggest_categorical("kernel_size1", [3, 5, 7, 9, 11, 13, 15]),
            "kernel_size2": trial.suggest_categorical("kernel_size2", [3, 5, 7, 9, 11, 13, 15]),
            "kernel_size3": trial.suggest_categorical("kernel_size3", [3, 5, 7, 9, 11, 13, 15]),
        }

    def _train_model(self, params, train_loader, val_loader, trial=None):
        set_global_seed(self.random_state)
        input_channels = self.input_shape[0] if self.input_shape else 1
        model = NiconPLModule(
            input_channels=input_channels,
            params=params,
            lr_max=1e-3,
            lr_min=1e-6,
            epochs=self.epochs_optuna,
        )
        model.to(self.device)

        trainer_callbacks = []
        if trial is not None:
            trainer_callbacks.append(CustomOptunaPruningCallback(trial, monitor="val_loss"))

        early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, verbose=self.verbose_optuna)
        trainer_callbacks.append(early_stop)

        trainer = pl.Trainer(
            max_epochs=self.epochs_optuna,
            enable_progress_bar=False,
            logger=False,
            callbacks=trainer_callbacks,
            enable_model_summary=False,
            devices=1 if self.device == "cuda" else None,
            accelerator=self.device if self.device == "cuda" else "cpu",
            deterministic=True,
            enable_checkpointing=False
        )

        trainer.fit(model, train_loader, val_loader)

        val_loss = trainer.callback_metrics.get("val_loss")
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        return val_loss, model


    def fit(self, X, y):
        set_global_seed(self.random_state)

        if not self.verbose_optuna:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.input_shape is None:
            # Input channels, length
            self.input_shape = (1, X.shape[1]) if len(X.shape) == 2 else X.shape[1:]

        X_tensor = self._reshape(X).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        def objective(trial):
            params = self._suggest_params(trial, n_samples=len(X))
            dataset = TensorDataset(X_tensor, y_tensor)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, len(dataset)))
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

            train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

            val_loss, _ = self._train_model(params, train_loader, val_loader, trial)
            return val_loss


        self.study_ = optuna.create_study(direction="minimize")
        self.study_.optimize(objective, n_trials=self.n_trials)

        self.best_params_ = self.study_.best_params

        # Train final model on all data
        final_params = self.best_params_.copy()
        batch_size = final_params.pop("batch_size")

        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = NiconPLModule(
            input_channels=self.input_shape[0],
            params=final_params,
            lr_max=1e-3,
            lr_min=1e-6,
            epochs=self.epochs,
        )
        self.model_ = model.to(self.device)

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            enable_progress_bar=self.verbose,
            logger=False,
            callbacks=[
                pl.callbacks.EarlyStopping(monitor="train_loss", patience=self.patience, verbose=self.verbose)
            ],
            enable_model_summary=False,
            devices=1 if self.device == "cuda" else None,
            accelerator=self.device if self.device == "cuda" else "cpu",
            deterministic=True,
            enable_checkpointing=False
        )

        trainer.fit(self.model_, train_loader)

    def predict(self, X):
        self.model_.eval()
        X_tensor = self._reshape(X).to(self.device)
        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy()
        return preds
