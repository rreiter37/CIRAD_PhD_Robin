import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
import torch.nn as nn

from Models.simple_feed_forward import SimpleFeedForward
from nicon_optuna import CustomOptunaPruningCallback

class FeedForwardPLModule(pl.LightningModule):
    def __init__(self, input_dim, params, lr_max, lr_min, epochs, t0_steps=None, cyclic_learning=True):
        super().__init__()
        self.model = SimpleFeedForward(input_dim=input_dim, params=params)
        self.criterion = nn.MSELoss()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epochs = epochs
        self.t0_steps = t0_steps
        self.cyclic_learning = cyclic_learning

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_max)
        if self.cyclic_learning:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.t0_steps if self.t0_steps else max(1, self.epochs // 4),
                T_mult=1,
                eta_min=self.lr_min,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        else:
            return optimizer

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
import optuna
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class FeedForwardOptunaRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape=None, n_trials=20, epochs=100, patience=10, lr_min=1e-6, lr_max=1e-3,
                 epochs_optuna=100, verbose=0, verbose_optuna=False, random_state=42, device=None,
                 get_logger=True, get_logger_optuna=False, cyclic_learning=True):
        self.input_shape = input_shape
        self.n_trials = n_trials
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

    def _reshape(self, X):
        return torch.tensor(X, dtype=torch.float32)

    def _suggest_params(self, trial):
        return {
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128]),
            "hidden1": trial.suggest_categorical("hidden1", [16, 32, 64, 128]),
            "hidden2": trial.suggest_categorical("hidden2", [8, 16, 32, 64]),
            "hidden3": trial.suggest_categorical("hidden3", [4, 8, 16, 32]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh", "elu", "selu", "swish"])
        }

    def _train_model(self, params, train_loader, val_loader, trial=None):
        input_dim = self.input_shape[0] if self.input_shape else train_loader.dataset.tensors[0].shape[1]
        steps_per_epoch = len(train_loader)
        t0_steps = max(1, steps_per_epoch * (self.epochs_optuna // 4))

        model = FeedForwardPLModule(
            input_dim=input_dim,
            params=params,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=self.epochs_optuna,
            t0_steps=t0_steps,
            cyclic_learning=self.cyclic_learning
        ).to(self.device)

        callbacks = []
        if trial is not None:
            callbacks.append(CustomOptunaPruningCallback(trial, monitor="val_loss"))
        callbacks.append(EarlyStopping(monitor="val_loss", patience=self.patience, verbose=self.verbose_optuna))

        logger = TensorBoardLogger("lightning_logs", name="ff_optuna") if self.get_logger_optuna else False

        trainer = pl.Trainer(
            max_epochs=self.epochs_optuna,
            logger=logger,
            callbacks=callbacks,
            accelerator=self.device,
            devices=1,
            enable_model_summary=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            deterministic=True
        )
        trainer.fit(model, train_loader, val_loader)
        val_loss = trainer.callback_metrics.get("val_loss")
        return model, val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

    def fit(self, X, y):
        X = self._reshape(X)
        y = torch.tensor(y, dtype=torch.float32)
        if self.input_shape is None:
            self.input_shape = X.shape[1:]

        dataset = TensorDataset(X, y)
        n_samples = len(X)
        n_val = max(1, int(n_samples * 0.2))
        train_set, val_set = random_split(dataset, [n_samples - n_val, n_val])

        def objective(trial):
            params = self._suggest_params(trial)
            batch_size = params.pop("batch_size")
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size)
            _, val_loss = self._train_model(params, train_loader, val_loader, trial)
            return val_loss

        self.study_ = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_state), pruner=optuna.pruners.HyperbandPruner())
        self.study_.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose_optuna)
        self.best_params_ = self.study_.best_params

        batch_size = self.best_params_.pop("batch_size")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        t0_steps = max(1, len(train_loader) * (self.epochs // 4))

        model = FeedForwardPLModule(
            input_dim=self.input_shape[0],
            params=self.best_params_,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=self.epochs,
            t0_steps=t0_steps,
            cyclic_learning=self.cyclic_learning
        ).to(self.device)

        callbacks = [EarlyStopping(monitor="val_loss", patience=self.patience, verbose=self.verbose)]
        logger = TensorBoardLogger("lightning_logs", name="ff_final") if self.get_logger else False
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator=self.device,
            devices=1,
            enable_model_summary=False,
            enable_checkpointing=False,
            enable_progress_bar=self.verbose > 0,
            deterministic=True
        )

        # split again for early stopping
        train_set, val_set = random_split(dataset, [n_samples - n_val, n_val])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        trainer.fit(model, train_loader, val_loader)
        self.model_ = model
        return self

    def predict(self, X):
        X = self._reshape(X).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            return self.model_(X).cpu().numpy()
