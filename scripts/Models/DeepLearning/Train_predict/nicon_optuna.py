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

from scripts.utils.max_batch_size import find_max_batch_size
from scripts.Models.DeepLearning.Architectures.nicon_custom_pytorch import CustomizableNicon
from scripts.utils.checkpointing_logger import CheckpointLoggerCallback

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
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========================================
# Dynamic Batch and LR Scaling
# ==========================================
class DynamicBatchScalingCallback(Callback):
    """
    Dynamically adjust batch size and learning rate based on gradient noise scale
    following McCandlish et al. (2018). Also records and logs batch size history.
    """
    def __init__(self, model_ref, dataset, adaptive_factor=2.0, probe_batches=5, min_batch=16, max_batch=8192):
        super().__init__()
        self.model_ref = model_ref  # Reference to NiconOptunaRegressor
        self.dataset = dataset
        self.adaptive_factor = adaptive_factor
        self.probe_batches = probe_batches
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.current_batch = model_ref.batch_size
        self.prev_noise_scale = None

        # Initialize batch size history
        if not hasattr(self.model_ref, "batch_size_history"):
            self.model_ref.batch_size_history = [self.current_batch]

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch to adapt batch size and LR dynamically,
        record history, and log everything to TensorBoard.
        """
        # Estimate gradient noise scale using current model state
        S = self.model_ref._estimate_noise_scale(self.dataset, pl_module.hparams.input_channels)
        if S is None or S <= 0:
            if self.model_ref.verbose:
                print(f"[Dynamic] Epoch {pl_module.current_epoch}: failed to estimate noise scale.")
            # Still append last known batch to maintain alignment
            self.model_ref.batch_size_history.append(self.current_batch)
            return

        # Compute new batch size using the square-root heuristic
        new_batch = int(np.clip(np.sqrt(self.adaptive_factor * S), self.min_batch, self.max_batch))

        # Adjust learning rate to maintain constant noise ratio (η/B ≈ const)
        optimizer = trainer.optimizers[0]
        old_lr = optimizer.param_groups[0]["lr"]
        new_lr = old_lr * np.sqrt(new_batch / max(1, self.current_batch))

        # Apply updates
        self.current_batch = new_batch
        optimizer.param_groups[0]["lr"] = new_lr

        # ✅ Record batch size into the model's history
        self.model_ref.batch_size_history.append(new_batch)

        if self.model_ref.verbose:
            print(f"[Dynamic] Epoch {pl_module.current_epoch}: "
                  f"S={S:.4e}, batch={new_batch}, lr={new_lr:.2e}")

        # Update DataLoader for next epoch
        train_loader = trainer.train_dataloader
        if hasattr(train_loader, "batch_sampler"):
            train_loader.batch_sampler.batch_size = new_batch
        else:
            trainer.fit_loop._data_loader = DataLoader(
                self.dataset, batch_size=new_batch, shuffle=True, num_workers=0
            )

        # ──────────────────────────────
        # 🔵 TensorBoard logging
        # ──────────────────────────────
        if isinstance(trainer.logger, pl.loggers.TensorBoardLogger):
            writer = trainer.logger.experiment
            epoch = pl_module.current_epoch

            # Scalar metrics for this epoch
            writer.add_scalar("adaptive/S_noise", S, epoch)
            writer.add_scalar("adaptive/batch_size", new_batch, epoch)
            writer.add_scalar("adaptive/lr", new_lr, epoch)

            # Full batch history (for visualization curve)
            batch_hist = np.array(self.model_ref.batch_size_history, dtype=float)
            writer.add_scalars("adaptive/batch_history", {"batch_size": batch_hist[-1]}, epoch)





class NiconPLModule(pl.LightningModule):
    def __init__(self, input_channels, params, lr_max, lr_min, epochs, t0_steps=None, cyclic_learning=True):
        super().__init__()
        self.save_hyperparameters(ignore=["t0_steps", "cyclic_learning"])
        self.params = params
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epochs = epochs
        self.t0_steps = t0_steps
        self.cyclic_learning = cyclic_learning

        self.model = CustomizableNicon(
            input_channels=input_channels,
            params=self.params
        )
        self.criterion = nn.MSELoss()

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
                 cyclic_learning=True, best_trials=None, name_pp=None,
                 adaptive_batch_size=False, adaptive_factor=2.0, probe_batches=5):
        """
        adaptive_batch_size : if True, estimate batch size using gradient noise scale
        adaptive_factor : scaling factor applied to noise scale for safety margin
        probe_batches : number of probing batches used to estimate noise scale
        """
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
        self.pp = None
        self.best_params_ = None
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger = get_logger
        self.get_logger_optuna = get_logger_optuna
        self.cyclic_learning = cyclic_learning
        self.best_trials = best_trials
        self.name_pp = name_pp
        self.adaptive_batch_size = adaptive_batch_size
        self.adaptive_factor = adaptive_factor
        self.probe_batches = probe_batches
        self.batch_size_history = []  # Stores list of batch sizes when dynamic mode is active

    def _reshape(self, X):
        X = np.array(X)
        if len(X.shape) == 2:
            X = X[:, np.newaxis, :]
        return torch.tensor(X, dtype=torch.float32)

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
            "kernel_size1": median_and_range("kernel_size1", 'int', 3, 35, step=2, scale=0.4),
            "kernel_size2": median_and_range("kernel_size2", 'int', 3, 35, step=2, scale=0.4),
            "kernel_size3": median_and_range("kernel_size3", 'int', 3, 35, step=2, scale=0.4),
            "spatial_dropout": median_and_range("spatial_dropout", 'float', 0.01, 0.5, scale=0.2),
            "dropout_rate": median_and_range("dropout_rate", 'float', 0.01, 0.5, scale=0.2),
            "filters1": median_and_range("filters1", 'int', 8, 64, step=8),
            "filters2": median_and_range("filters2", 'int', 32, 256, step=32),
            "filters3": median_and_range("filters3", 'int', 13, 128, step=16),
            "dense_units":median_and_range("dense_units", 'int', 16, 128, step=8)
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

        # --- PRE-FLIGHT SHAPE CHECK ---
        with torch.no_grad():
            try:
                dummy = torch.zeros(2, input_channels, self.input_shape[-1], device=self.device)
                _ = model.model(dummy)  # forward of the internal architecture (CustomizableNicon)
            except RuntimeError as e:
                # If invalid shape (kernel size too big), trial is pruned
                if trial is not None:
                    trial.set_user_attr("shape_error", str(e))
                    raise optuna.TrialPruned()
                else:
                    raise
        # --- END PRE-FLIGHT ---

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
            verbose=self.verbose_optuna,
            filename=f"{self.name_pp or 'noprep'}-{{epoch:02d}}-{{val_loss:.4f}}"
        )

        early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, verbose=self.verbose_optuna)

        callbacks = [checkpoint_callback, early_stopping, CheckpointLoggerCallback()]
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
        )

        trainer.fit(model, train_loader, val_loader)

        if checkpoint_callback.best_model_path:
            model = NiconPLModule.load_from_checkpoint(checkpoint_callback.best_model_path)


        val_loss = trainer.callback_metrics.get("val_loss")
        return model, val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
    
    def _estimate_noise_scale(self, dataset, input_channels):
        """
        Estimate gradient noise scale (McCandlish et al., 2018) using small probe batches.
        """
        # Use small probing batch size (default 32)
        probe_batch = min(32, len(dataset))
        loader = DataLoader(dataset, batch_size=probe_batch, shuffle=True)
        params = {"kernel_size1": 3, "kernel_size2": 3, "kernel_size3": 3,
                  "spatial_dropout": 0.01, "dropout_rate": 0.01, "output_dim": 1}
        model = NiconPLModule(
            input_channels=input_channels,
            params=params,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=1
        ).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        grads = []
        # Collect gradients on a few probe batches
        for i, (x, y) in enumerate(loader):
            if i >= self.probe_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = nn.MSELoss()(y_pred, y)
            loss.backward()
            grad_vector = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            grads.append(grad_vector.detach().cpu())

        if len(grads) < 2:
            return None  # Not enough data

        grads = torch.stack(grads)
        g_mean = grads.mean(dim=0)
        var = grads.var(dim=0, unbiased=True).mean()  # mean variance across params
        norm_sq = g_mean.pow(2).sum()

        # Gradient noise scale S ≈ variance / ||g||^2
        if norm_sq.item() == 0:
            return None
        S = var.item() / norm_sq.item()
        return S

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
        train_set, val_set = random_split(dataset, [n_train, n_val],
                                          generator=torch.Generator().manual_seed(self.random_state))

        # --- Adaptive batch size estimation ---
        if self.batch_size is None:
            if self.adaptive_batch_size in [True, "static"]:
                try:
                    input_channels = self.input_shape if isinstance(self.input_shape, int) else self.input_shape[0]
                    S = self._estimate_noise_scale(train_set, input_channels)
                    if S is not None and S > 0:
                        # Use heuristic: batch ≈ S / factor, bounded by GPU max
                        max_hw_batch = find_max_batch_size(
                            model=CustomizableNicon(input_channels, {"kernel_size1": 3, "kernel_size2": 3,
                                                                    "kernel_size3": 3, "spatial_dropout": 0.01,
                                                                    "dropout_rate": 0.01, "output_dim": 1}),
                            input_shape=self.input_shape, device=self.device,
                            max_batch=X.shape[-1], min_batch=1
                        )
                        self.batch_size = int(min(max_hw_batch, max(32, S / self.adaptive_factor)))
                        if self.verbose:
                            print(f"[Static] Gradient noise scale={S:.4e}, chosen batch size={self.batch_size}")
                    else:
                        # Fallback if noise scale fails
                        raise RuntimeError("Noise scale estimation failed")
                except Exception as e:
                    if self.verbose:
                        print(f"[Adaptive] Fallback to max batch size due to error: {e}")
                    params = {"kernel_size1": 3, "kernel_size2": 3, "kernel_size3": 3,
                              "spatial_dropout": 0.01, "dropout_rate": 0.01, "output_dim": 1}
                    model = NiconPLModule(
                        input_channels=self.input_shape[0],
                        params=params,
                        lr_max=self.lr_max,
                        lr_min=self.lr_min,
                        epochs=1
                    )
                    self.batch_size = find_max_batch_size(model=model, input_shape=self.input_shape,
                                                          device=self.device, max_batch=X.shape[-1], min_batch=1)
                    print("Maximum batch size found (adaptive): ", self.batch_size)
            elif self.adaptive_batch_size == "dynamic":
                # Initialize batch size using small probe, then dynamically adapt
                self.batch_size = min(max(n_train // 10, 8), 32)
                if self.verbose:
                    print("[Dynamic] Starting with small batch (8 < B < 32) and adaptive scaling per epoch.")
            else:
                # Original max batch size strategy
                params = {"kernel_size1": 3, "kernel_size2": 3, "kernel_size3": 3,
                          "spatial_dropout": 0.01, "dropout_rate": 0.01, "output_dim": 1}
                model = NiconPLModule(
                    input_channels=self.input_shape[0],
                    params=params,
                    lr_max=self.lr_max,
                    lr_min=self.lr_min,
                    epochs=1
                )
                self.batch_size = find_max_batch_size(model=model, input_shape=self.input_shape,
                                                      device=self.device, max_batch=X.shape[-1], min_batch=1)
                print("Maximum batch size found (GPU-oriented): ", self.batch_size)

        def objective(trial):
            params = self._suggest_params(trial)
            params["output_dim"] = 1
            g = torch.Generator()
            g.manual_seed(self.random_state)
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker, num_workers=0)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, generator=g, worker_init_fn=seed_worker, num_workers=0)
            _, val_loss = self._train_model(params, train_loader, val_loader, trial=trial)
            return val_loss

        study_name=f"optuna_{self.pp}" if self.pp is not None else "optuna"

        self.study_ = optuna.create_study(direction="minimize", 
                                          sampler=TPESampler(seed=self.random_state), 
                                          pruner=HyperbandPruner(), 
                                          study_name=study_name
                                          )
        self.study_.optimize(objective, 
                             n_trials=self.n_trials,
                             show_progress_bar=self.verbose_optuna
                             )
        
        self.best_params_ = self.study_.best_params
        self.best_params_["output_dim"] = 1
        
        # Store the best hyperameters for next pp-model associations
        if self.best_trials is None:
            self.best_trials = [self.best_params_]
        else:
            self.best_trials.append(self.best_params_)

        # Final training phase
        g = torch.Generator()
        g.manual_seed(self.random_state)
        train_final, val_final = random_split(dataset, [n_samples - n_val, n_val], generator=g)
        train_loader_final = DataLoader(train_final, batch_size=self.batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker, num_workers=0)
        val_loader_final = DataLoader(val_final, batch_size=self.batch_size, generator=g, worker_init_fn=seed_worker, num_workers=0)

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

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            verbose=self.verbose,
            filename=f"{self.name_pp or 'noprep'}-{{epoch:02d}}-{{val_loss:.4f}}"
        )

        early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, verbose=self.verbose)

        callbacks = [checkpoint_callback, early_stopping, CheckpointLoggerCallback()]

        # Add dynamic adaptation callback if requested
        if self.adaptive_batch_size == "dynamic":
            dyn_callback = DynamicBatchScalingCallback(
                model_ref=self,
                dataset=train_final,
                adaptive_factor=self.adaptive_factor,
                probe_batches=self.probe_batches,
                max_batch=self.batch_size * 64  # safety cap
            )
            callbacks.append(dyn_callback)

        if self.name_pp is None:
            name = "cnn_final"
        else:
            name = "cnn_final_%s"%self.name_pp
        logger = TensorBoardLogger("lightning_logs", name=name, default_hp_metric=False) if self.get_logger else False

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            enable_progress_bar=self.verbose > 0,
            logger=logger,
            callbacks=callbacks,
            enable_model_summary=False,
            devices=1 if self.device == "cuda" else None,
            accelerator=self.device if self.device == "cuda" else "cpu",
            deterministic=True,
        )

        trainer.fit(final_model, train_loader_final, val_loader_final)

        if checkpoint_callback.best_model_path:
            final_model = NiconPLModule.load_from_checkpoint(checkpoint_callback.best_model_path)

        self.model_ = final_model

        # If dynamic mode was active but no history recorded, fallback to single batch value
        if self.adaptive_batch_size == "dynamic" and not hasattr(self, "batch_size_history"):
            self.batch_size_history = [self.batch_size]

        if self.get_logger and hasattr(self, "batch_size_history"):
            try:
                if isinstance(self.model_.logger, pl.loggers.TensorBoardLogger):
                    writer = self.model_.logger.experiment
                    for epoch, bsize in enumerate(self.batch_size_history):
                        writer.add_scalar("adaptive/batch_history_final", bsize, epoch)
            except Exception:
                pass

        return self

    def predict(self, X):
        self.model_.eval()
        X = self._reshape(X).to(self.device)
        with torch.no_grad():
            preds = self.model_(X)
        return preds.cpu().numpy()
