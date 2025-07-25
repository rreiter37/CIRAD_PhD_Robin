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
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

from max_batch_size import find_max_batch_size
from Models.nicon_custom_pytorch import CustomizableNicon

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger



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
    def __init__(self, input_channels, params, lr_max, lr_min, epochs, t0_steps=None, cyclic_learning=True):
        super().__init__()
        self.model = CustomizableNicon(input_channels=input_channels, params=params)
        self.criterion = nn.MSELoss()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epochs = epochs
        self.t0_steps = t0_steps  # Scheduler restart cycle length
        self.cyclic_learning = cyclic_learning

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x).squeeze(1)  # Ensure output shape is (batch,)

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

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_epoch=True, on_step=False)
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

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        # Manually log train loss and learning rate using epoch as the X-axis
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            writer = self.logger.experiment
            epoch = self.current_epoch

            train_loss = self.trainer.callback_metrics.get("train_loss")
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
            writer.add_scalar("epoch/train_loss", train_loss, epoch)

            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            writer.add_scalar("epoch/lr", lr, epoch)

    def on_validation_epoch_end(self):
        # Manually log validation loss using epoch as the X-axis
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            writer = self.logger.experiment
            epoch = self.current_epoch

            val_loss = self.trainer.callback_metrics.get("val_loss")
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            writer.add_scalar("epoch/val_loss", val_loss, epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_max)

        if self.cyclic_learning:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.t0_steps if self.t0_steps is not None else max(1, self.epochs // 4),
                T_mult=1,
                eta_min=self.lr_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # Scheduler updates every training step
                    "frequency": 1,
                }
            }
        else:
            return optimizer



class NiconOptunaRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape=None, n_trials=20, batch_size=None, epochs=100, patience=10,
                 lr_min=1e-6, lr_max=1e-3, epochs_optuna=100, verbose=0, verbose_optuna=False,
                 random_state=42, device=None, get_logger=True, get_logger_optuna=False,
                 cyclic_learning=True, best_trials=None):
        self.input_shape = input_shape
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.epochs_optuna = epochs_optuna
        self.verbose = verbose
        self.verbose_optuna = verbose_optuna
        self.random_state = random_state
        self.study_ = None
        self.model_ = None
        self.best_params_ = None
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger = get_logger
        self.get_logger_optuna = get_logger_optuna
        self.cyclic_learning = cyclic_learning
        self.best_trials = best_trials

    def _reshape(self, X):
        X = np.array(X)
        if len(X.shape) == 2:
            X = X[:, np.newaxis, :]
        return torch.tensor(X, dtype=torch.float32)

    def _suggest_params(self, trial, best_trials=None):
        def median_and_range(param_name, type, low, high, log=False, step=2):
            if best_trials is None:
                if type=='int':
                    return trial.suggest_int(param_name, low, high, step=step)
                else:
                    return trial.suggest_float(param_name, low, high, log=log)
            values = [t.params[param_name] for t in best_trials if param_name in t.params]
            if not values:
                if type=='int':
                    return trial.suggest_int(param_name, low, high, step=step)
                else:
                    return trial.suggest_float(param_name, low, high, log=log)
            median = np.median(values)
            if type=='int':
                median = int(median)
                delta = int((high - low) * 0.4)
                bounded_low = max(low, median - delta)
                bounded_high = min(high, median + delta)
                return trial.suggest_int(param_name, bounded_low, bounded_high, step = max(1, step//2))
            else:
                delta = (high - low) * 0.1
                bounded_low = max(low, median - delta)
                bounded_high = min(high, median + delta)
                return trial.suggest_float(param_name, bounded_low, bounded_high, log=log)

        return {
            "kernel_size1": median_and_range("kernel_size1", 'int', 3, 25, step=2),
            "kernel_size2": median_and_range("kernel_size2", 'int', 3, 25, step=2),
            "kernel_size3": median_and_range("kernel_size3", 'int', 3, 25, step=2),
            "spatial_dropout": median_and_range("spatial_dropout", 'float', 0.01, 0.5),
            "dropout_rate": median_and_range("dropout_rate", 'float', 0.01, 0.5),
        }

    def _train_model(self, params, train_loader, val_loader, trial=None):
        set_global_seed(self.random_state)
        input_channels = self.input_shape[0] if self.input_shape else 1

        steps_per_epoch = len(train_loader)
        t0_steps = steps_per_epoch * (self.epochs_optuna // 4) or 1

        model = NiconPLModule(
            input_channels=input_channels,
            params=params,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=self.epochs_optuna,
            t0_steps=t0_steps,
            cyclic_learning=self.cyclic_learning
        )
        model.to(self.device)

        callbacks = [pl.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, verbose=self.verbose_optuna)]
        if trial is not None:
            callbacks.insert(0, CustomOptunaPruningCallback(trial, monitor="val_loss"))

        logger = TensorBoardLogger("lightning_logs", name="nicon_optuna", default_hp_metric=False) if self.get_logger_optuna else False

        trainer = pl.Trainer(
            max_epochs=self.epochs_optuna,
            enable_progress_bar=False,
            logger=logger,
            callbacks=callbacks,
            enable_model_summary=False,
            devices=1 if self.device == "cuda" else None,
            accelerator=self.device if self.device == "cuda" else "cpu",
            deterministic=True,
            enable_checkpointing=False
        )

        trainer.fit(model, train_loader, val_loader)

        val_loss = trainer.callback_metrics.get("val_loss")
        return model, val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

    def fit(self, X, y):
        X = self._reshape(X)
        y = torch.tensor(y, dtype=torch.float32)

        if self.input_shape is None:
            self.input_shape = X.shape[1:]

        n_samples = len(X)
        set_global_seed(self.random_state)

        dataset = TensorDataset(X, y)
        n_val = max(1, int(n_samples * 0.2))
        n_train = n_samples - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(self.random_state))

        if self.batch_size is None:
            params = {"kernel_size1": 3, "kernel_size2": 3, "kernel_size3": 3, "spatial_dropout": 0.01, "dropout_rate": 0.01}
            model = NiconPLModule(
                input_channels=self.input_shape[0],
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
            params = self._suggest_params(trial, n_train, best_trials=self.best_trials)
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=self.batch_size)
            _, val_loss = self._train_model(params, train_loader, val_loader, trial=trial)
            return val_loss

        self.study_ = optuna.create_study(direction="minimize", sampler=TPESampler(seed=self.random_state), pruner=HyperbandPruner())
        self.study_.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose_optuna)

        self.best_params_ = self.study_.best_params

        if self.best_trials is None:
            self.best_trials = [self.best_params_]
        else:
            values = [t.params[param_name] for t in best_trials if param_name in t.params]
            self.best_trials = 

        # Final training phase
        train_final, val_final = random_split(dataset, [n_samples - n_val, n_val], generator=torch.Generator().manual_seed(self.random_state))
        train_loader_final = DataLoader(train_final, batch_size=self.batch_size, shuffle=True)
        val_loader_final = DataLoader(val_final, batch_size=self.batch_size)

        t0_steps = len(train_loader_final) * (self.epochs // 4) or 1

        final_model = NiconPLModule(
            input_channels=self.input_shape[0],
            params=self.best_params_,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=self.epochs,
            t0_steps=t0_steps,
            cyclic_learning=self.cyclic_learning
        )
        final_model.to(self.device)

        callbacks = [pl.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, verbose=self.verbose)]
        logger = TensorBoardLogger("lightning_logs", name="nicon_final", default_hp_metric=False) if self.get_logger else False

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            enable_progress_bar=self.verbose > 0,
            logger=logger,
            callbacks=callbacks,
            enable_model_summary=False,
            devices=1 if self.device == "cuda" else None,
            accelerator=self.device if self.device == "cuda" else "cpu",
            deterministic=True,
            enable_checkpointing=False
        )

        trainer.fit(final_model, train_loader_final, val_loader_final)

        self.model_ = final_model
        return self

    def predict(self, X):
        self.model_.eval()
        X = self._reshape(X).to(self.device)
        with torch.no_grad():
            preds = self.model_(X)
        return preds.cpu().numpy()
