# scripts/models/nicon/regressor.py

from typing import Any, Dict, List, Optional

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from sklearn.base import BaseEstimator, RegressorMixin

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from scripts.utils.max_batch_size import find_max_batch_size
from scripts.utils.checkpointing_logger import CheckpointLoggerCallback

from .utils import set_global_seed, seed_worker, reshape_input
from .callbacks import CustomOptunaPruningCallback, DynamicBatchScalingCallback
from .lightning_module import NiconPLModule


class NiconOptunaRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn-style regressor that wraps a PyTorch Lightning CNN (CustomizableNicon)
    and performs hyperparameter optimization with Optuna.

    Features
    --------
    - Uses Optuna TPE + Hyperband for hyperparameter search.
    - Supports early stopping and pruning.
    - Can adapt batch size using gradient noise scale (static or dynamic).
    - Compatible with sklearn pipelines.
    """

    def __init__(
        self,
        input_shape: Optional[Any] = None,
        n_trials: int = 20,
        batch_size: Optional[int] = None,
        epochs: int = 100,
        patience: int = 10,
        lr_min: float = 1e-6,
        lr_max: float = 1e-3,
        epochs_optuna: int = 100,
        verbose: int = 0,
        verbose_optuna: bool = False,
        random_state: int = 42,
        device: Optional[str] = None,
        get_logger: bool = True,
        get_logger_optuna: bool = False,
        cyclic_learning: bool = True,
        best_trials: Optional[List[Dict[str, Any]]] = None,
        name_pp: Optional[str] = None,
        adaptive_batch_size: Any = False,
        adaptive_factor: float = 2.0,
        probe_batches: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        input_shape : tuple or None
            Shape of input data (C, L). If None, inferred from first fit call.
        n_trials : int
            Number of Optuna trials for hyperparameter search.
        batch_size : int or None
            If None, batch size is chosen via hardware-based search (or adaptive mode).
        epochs : int
            Number of epochs for the final training.
        patience : int
            Patience for early stopping.
        lr_min : float
            Minimum learning rate in cosine annealing.
        lr_max : float
            Maximum learning rate for the optimizer.
        epochs_optuna : int
            Number of epochs during Optuna hyperparameter search.
        verbose : int
            Verbosity level for final training.
        verbose_optuna : bool
            Verbosity level for Optuna phase.
        random_state : int
            Random seed used for reproducibility.
        device : str or None
            "cuda", "cpu", or None. If None, "cuda" is used if available.
        get_logger : bool
            Whether to use TensorBoard logger for the final training.
        get_logger_optuna : bool
            Whether to use TensorBoard logger for the Optuna phase.
        cyclic_learning : bool
            Whether to use cosine annealing with restarts.
        best_trials : list of dict or None
            Previous best hyperparameters used to narrow search space in progressive optimization.
        name_pp : str or None
            Optional name of preprocessing, used to name log directories and checkpoints.
        adaptive_batch_size : bool or str
            Controls batch size adaptation:
            - False / "False": no adaptation (hardware-based max batch).
            - "static": one-shot estimation via gradient noise scale.
            - "dynamic": adapt batch size per epoch using DynamicBatchScalingCallback.
        adaptive_factor : float
            Safety factor for noise scale based batch size heuristics.
        probe_batches : int
            Number of probe batches used to estimate noise scale.
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
        self.study_: Optional[optuna.Study] = None
        self.model_: Optional[NiconPLModule] = None
        self.pp: Optional[str] = None
        self.best_params_: Optional[Dict[str, Any]] = None

        # Prefer GPU if available
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.get_logger = get_logger
        self.get_logger_optuna = get_logger_optuna
        self.cyclic_learning = cyclic_learning
        self.best_trials = best_trials
        self.name_pp = name_pp
        self.adaptive_batch_size = adaptive_batch_size
        self.adaptive_factor = adaptive_factor
        self.probe_batches = probe_batches
        self.batch_size_history: List[int] = []

    # ------------------------------------------------------------------
    # Hyperparameter suggestion
    # ------------------------------------------------------------------
    def _suggest_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Suggest CNN hyperparameters, possibly narrowed around previous best trials."""

        best_trials = self.best_trials

        def median_and_range(
            param_name: str,
            type: str,
            low: float,
            high: float,
            log: bool = False,
            step: int = 2,
            scale: float = 0.4,
        ):
            if best_trials is None:
                if type == "int":
                    return trial.suggest_int(param_name, int(low), int(high), step=step)
                return trial.suggest_float(param_name, low, high, log=log)

            values = [t[param_name] for t in best_trials if param_name in t.keys()]
            if not values:
                if type == "int":
                    return trial.suggest_int(param_name, int(low), int(high), step=step)
                return trial.suggest_float(param_name, low, high, log=log)

            median = np.median(values)

            if type == "int":
                median = int(median)
                delta = int((high - low) * scale / 2)
                bounded_low = max(int(low), median - delta)
                bounded_high = min(int(high), median + delta)
                return trial.suggest_int(
                    param_name,
                    bounded_low,
                    bounded_high,
                    step=max(1, step // 2),
                )

            delta = (high - low) * scale / 2
            bounded_low = max(low, median - delta)
            bounded_high = min(high, median + delta)
            return trial.suggest_float(param_name, bounded_low, bounded_high, log=log)

        return {
            "kernel_size1": median_and_range("kernel_size1", "int", 3, 35, step=2, scale=0.4),
            "kernel_size2": median_and_range("kernel_size2", "int", 3, 35, step=2, scale=0.4),
            "kernel_size3": median_and_range("kernel_size3", "int", 3, 35, step=2, scale=0.4),
            "spatial_dropout": median_and_range(
                "spatial_dropout",
                "float",
                0.01,
                0.5,
                scale=0.2,
            ),
            "dropout_rate": median_and_range(
                "dropout_rate",
                "float",
                0.01,
                0.5,
                scale=0.2,
            ),
            "filters1": median_and_range("filters1", "int", 8, 64, step=8),
            "filters2": median_and_range("filters2", "int", 32, 256, step=32),
            "filters3": median_and_range("filters3", "int", 13, 128, step=16),
            "dense_units": median_and_range("dense_units", "int", 16, 128, step=8),
        }

    # ------------------------------------------------------------------
    # Internal training used by Optuna
    # ------------------------------------------------------------------
    def _train_model(
        self,
        params: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        trial: Optional[optuna.trial.Trial] = None,
    ):
        """
        Internal training loop used during Optuna optimization.

        This method uses GPU if available and can leverage mixed precision to speed up training.
        """
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
            cyclic_learning=self.cyclic_learning,
        )
        model.to(self.device)

        # Pre-flight shape check to catch dimension mismatches early
        with torch.no_grad():
            try:
                dummy = torch.zeros(2, input_channels, self.input_shape[-1], device=self.device)
                _ = model.model(dummy)
            except RuntimeError as e:
                if trial is not None:
                    trial.set_user_attr("shape_error", str(e))
                    raise optuna.TrialPruned()
                raise

        save_top_k = 0 if trial is not None else 1
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=save_top_k,
            mode="min",
            save_last=False,
            verbose=self.verbose_optuna,
            filename=f"{self.name_pp or 'noprep'}-{{epoch:02d}}-{{val_loss:.4f}}",
        )

        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            verbose=self.verbose_optuna,
        )

        callbacks = [checkpoint_callback, early_stopping, CheckpointLoggerCallback()]

        if trial is not None:
            callbacks.insert(0, CustomOptunaPruningCallback(trial, monitor="val_loss"))

        logger = (
            TensorBoardLogger(
                "lightning_logs",
                name="nicon_optuna",
                default_hp_metric=False,
            )
            if self.get_logger_optuna
            else False
        )

        use_gpu = (self.device == "cuda") and torch.cuda.is_available()
        accelerator = "gpu" if use_gpu else "cpu"
        devices = 1
        precision = 16 if use_gpu else 32

        trainer = pl.Trainer(
            max_epochs=self.epochs_optuna,
            enable_progress_bar=False,
            logger=logger,
            callbacks=callbacks,
            enable_model_summary=False,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            deterministic=True,
        )

        trainer.fit(model, train_loader, val_loader)

        if checkpoint_callback.best_model_path:
            model = NiconPLModule.load_from_checkpoint(checkpoint_callback.best_model_path)

        val_loss = trainer.callback_metrics.get("val_loss")
        return model, val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

    # ------------------------------------------------------------------
    # Gradient noise scale estimation (for adaptive batch size)
    # ------------------------------------------------------------------
    def _estimate_noise_scale(self, dataset, input_channels: int) -> Optional[float]:
        """
        Estimate gradient noise scale using small probe batches.

        This is based on McCandlish et al. (2018). It is only used when
        adaptive batch size is enabled.
        """
        probe_batch = min(32, len(dataset))
        loader = DataLoader(dataset, batch_size=probe_batch, shuffle=True)

        params = {
            "kernel_size1": 3,
            "kernel_size2": 3,
            "kernel_size3": 3,
            "spatial_dropout": 0.01,
            "dropout_rate": 0.01,
            "output_dim": 1,
        }
        model = NiconPLModule(
            input_channels=input_channels,
            params=params,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=1,
        ).to(self.device)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        grads = []
        for i, (x, y) in enumerate(loader):
            if i >= self.probe_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = nn.MSELoss()(y_pred, y)
            loss.backward()
            grad_vector = torch.cat(
                [p.grad.flatten() for p in model.parameters() if p.grad is not None]
            )
            grads.append(grad_vector.detach().cpu())

        if len(grads) < 2:
            return None

        grads = torch.stack(grads)
        g_mean = grads.mean(dim=0)
        var = grads.var(dim=0, unbiased=True).mean()
        norm_sq = g_mean.pow(2).sum()

        if norm_sq.item() == 0:
            return None

        S = var.item() / norm_sq.item()
        return S

    # ------------------------------------------------------------------
    # Public API: fit / predict
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit the Nicon model with Optuna hyperparameter search."""
        X = reshape_input(X)
        y = torch.tensor(y, dtype=torch.float32)

        if self.input_shape is None:
            self.input_shape = X.shape[1:]

        n_samples = len(X)
        set_global_seed(self.random_state)

        dataset = TensorDataset(X, y)
        n_val = max(1, int(n_samples * 0.2))
        n_train = n_samples - n_val

        train_set, val_set = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(self.random_state),
        )

        # -----------------------------
        # Batch size selection
        # -----------------------------
        if self.batch_size is None:
            if self.adaptive_batch_size in [True, "static"]:
                try:
                    input_channels = (
                        self.input_shape
                        if isinstance(self.input_shape, int)
                        else self.input_shape[0]
                    )
                    S = self._estimate_noise_scale(train_set, input_channels)
                    if S is not None and S > 0:
                        from scripts.Models.DeepLearning.Architectures.nicon_custom_pytorch import (
                            CustomizableNicon,
                        )

                        max_hw_batch = find_max_batch_size(
                            model=CustomizableNicon(
                                input_channels,
                                {
                                    "kernel_size1": 3,
                                    "kernel_size2": 3,
                                    "kernel_size3": 3,
                                    "spatial_dropout": 0.01,
                                    "dropout_rate": 0.01,
                                    "output_dim": 1,
                                },
                            ),
                            input_shape=self.input_shape,
                            device=self.device,
                            max_batch=X.shape[-1],
                            min_batch=1,
                        )
                        self.batch_size = int(
                            min(max_hw_batch, max(32, S / self.adaptive_factor))
                        )
                        if self.verbose:
                            print(
                                f"[Static] Gradient noise scale={S:.4e}, "
                                f"chosen batch size={self.batch_size}"
                            )
                    else:
                        raise RuntimeError("Noise scale estimation failed")
                except Exception as e:
                    if self.verbose:
                        print(
                            f"[Adaptive] Fallback to max batch size due to error: {e}"
                        )
                    from scripts.Models.DeepLearning.Architectures.nicon_custom_pytorch import (
                        CustomizableNicon,
                    )

                    model = CustomizableNicon(
                        self.input_shape[0],
                        {
                            "kernel_size1": 3,
                            "kernel_size2": 3,
                            "kernel_size3": 3,
                            "spatial_dropout": 0.01,
                            "dropout_rate": 0.01,
                            "output_dim": 1,
                        },
                    )
                    self.batch_size = find_max_batch_size(
                        model=model,
                        input_shape=self.input_shape,
                        device=self.device,
                        max_batch=X.shape[-1],
                        min_batch=1,
                    )
                    print("Maximum batch size found (adaptive): ", self.batch_size)

            elif self.adaptive_batch_size == "dynamic":
                self.batch_size = min(max(n_train // 10, 8), 32)
                if self.verbose:
                    print(
                        "[Dynamic] Starting with small batch (8 < B < 32) "
                        "and adaptive scaling per epoch."
                    )
            else:
                from scripts.Models.DeepLearning.Architectures.nicon_custom_pytorch import (
                    CustomizableNicon,
                )

                model = CustomizableNicon(
                    self.input_shape[0],
                    {
                        "kernel_size1": 3,
                        "kernel_size2": 3,
                        "kernel_size3": 3,
                        "spatial_dropout": 0.01,
                        "dropout_rate": 0.01,
                        "output_dim": 1,
                    },
                )
                self.batch_size = find_max_batch_size(
                    model=model,
                    input_shape=self.input_shape,
                    device=self.device,
                    max_batch=X.shape[-1],
                    min_batch=1,
                )
                print("Maximum batch size found (GPU-oriented): ", self.batch_size)

        # -----------------------------
        # Optuna objective definition
        # -----------------------------
        def objective(trial: optuna.trial.Trial) -> float:
            params = self._suggest_params(trial)
            params["output_dim"] = 1

            g = torch.Generator().manual_seed(self.random_state)
            train_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                generator=g,
                worker_init_fn=seed_worker,
                num_workers=0,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=self.batch_size,
                generator=g,
                worker_init_fn=seed_worker,
                num_workers=0,
            )

            _, val_loss = self._train_model(params, train_loader, val_loader, trial=trial)
            return val_loss

        study_name = f"optuna_{self.pp}" if self.pp is not None else "optuna"
        self.study_ = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.random_state),
            pruner=HyperbandPruner(),
            study_name=study_name,
        )
        self.study_.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose_optuna,
        )

        self.best_params_ = self.study_.best_params
        self.best_params_["output_dim"] = 1

        if self.best_trials is None:
            self.best_trials = [self.best_params_]
        else:
            self.best_trials.append(self.best_params_)

        # -----------------------------
        # Final training with best hyperparameters
        # -----------------------------
        g = torch.Generator().manual_seed(self.random_state)
        n_val = max(1, int(n_samples * 0.2))
        n_train = n_samples - n_val

        train_final, val_final = random_split(
            dataset,
            [n_train, n_val],
            generator=g,
        )

        train_loader_final = DataLoader(
            train_final,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
            num_workers=0,
        )
        val_loader_final = DataLoader(
            val_final,
            batch_size=self.batch_size,
            generator=g,
            worker_init_fn=seed_worker,
            num_workers=0,
        )

        t0_steps = len(train_loader_final) * (self.epochs // 4) or 1

        final_model = NiconPLModule(
            input_channels=self.input_shape[0],
            params=self.best_params_,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=self.epochs,
            t0_steps=t0_steps,
            cyclic_learning=self.cyclic_learning,
        )
        final_model.to(self.device)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            verbose=self.verbose,
            filename=f"{self.name_pp or 'noprep'}-{{epoch:02d}}-{{val_loss:.4f}}",
        )

        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            verbose=self.verbose,
        )

        callbacks = [checkpoint_callback, early_stopping, CheckpointLoggerCallback()]

        if self.adaptive_batch_size == "dynamic":
            dyn_callback = DynamicBatchScalingCallback(
                model_ref=self,
                dataset=train_final,
                adaptive_factor=self.adaptive_factor,
                probe_batches=self.probe_batches,
                max_batch=self.batch_size * 64,
            )
            callbacks.append(dyn_callback)

        if self.name_pp is None:
            name = "cnn_final"
        else:
            name = f"cnn_final_{self.name_pp}"

        logger = (
            TensorBoardLogger(
                "lightning_logs",
                name=name,
                default_hp_metric=False,
            )
            if self.get_logger
            else False
        )

        use_gpu = (self.device == "cuda") and torch.cuda.is_available()
        accelerator = "gpu" if use_gpu else "cpu"
        devices = 1
        precision = 16 if use_gpu else 32

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            enable_progress_bar=self.verbose > 0,
            logger=logger,
            callbacks=callbacks,
            enable_model_summary=False,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            deterministic=True,
        )

        trainer.fit(final_model, train_loader_final, val_loader_final)

        if checkpoint_callback.best_model_path:
            final_model = NiconPLModule.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )

        self.model_ = final_model

        if self.adaptive_batch_size == "dynamic" and not self.batch_size_history:
            self.batch_size_history = [self.batch_size]

        if self.get_logger and hasattr(self.model_, "logger"):
            try:
                if isinstance(self.model_.logger, pl.loggers.TensorBoardLogger):
                    writer = self.model_.logger.experiment
                    for epoch, bsize in enumerate(self.batch_size_history):
                        writer.add_scalar("adaptive/batch_history_final", bsize, epoch)
            except Exception:
                pass

        return self

    def predict(self, X):
        """Standard sklearn-style predict: returns a numpy array on CPU."""
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        self.model_.eval()
        X = reshape_input(X).to(self.device)
        with torch.no_grad():
            preds = self.model_(X)
        return preds.cpu().numpy()
