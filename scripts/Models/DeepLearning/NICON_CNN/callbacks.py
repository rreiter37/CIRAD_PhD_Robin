# scripts/models/nicon/callbacks.py

from typing import Optional

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class CustomOptunaPruningCallback(Callback):
    """Callback that reports validation loss to Optuna and prunes the trial if needed."""

    def __init__(self, trial: optuna.trial.Trial, monitor: str = "val_loss") -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """Report validation metric to Optuna and trigger pruning if appropriate."""
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()

        self.trial.report(current_score, step=trainer.current_epoch)

        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class DynamicBatchScalingCallback(Callback):
    """
    Dynamically adjust batch size and learning rate based on gradient noise scale.

    This follows the ideas from McCandlish et al. (2018), using a heuristic to
    scale the batch size and learning rate to maintain a roughly constant noise level.

    This is relatively expensive; only enable it when adaptive_batch_size == "dynamic".
    """

    def __init__(
        self,
        model_ref,
        dataset,
        adaptive_factor: float = 2.0,
        probe_batches: int = 5,
        min_batch: int = 16,
        max_batch: int = 8192,
    ) -> None:
        """
        Parameters
        ----------
        model_ref : object
            Reference to the NiconOptunaRegressor instance.
        dataset : Dataset
            Training dataset used for estimating the noise scale.
        adaptive_factor : float
            Safety factor applied to the estimated noise scale.
        probe_batches : int
            Number of probe batches used for estimating the noise scale.
        min_batch : int
            Minimum allowed batch size.
        max_batch : int
            Maximum allowed batch size.
        """
        super().__init__()
        self.model_ref = model_ref
        self.dataset = dataset
        self.adaptive_factor = adaptive_factor
        self.probe_batches = probe_batches
        self.min_batch = min_batch
        self.max_batch = max_batch

        self.current_batch = model_ref.batch_size
        self.prev_noise_scale: Optional[float] = None

        if not hasattr(self.model_ref, "batch_size_history"):
            self.model_ref.batch_size_history = [self.current_batch]

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """
        At the end of each training epoch, estimate the gradient noise scale,
        update the batch size and learning rate accordingly, and log the changes.
        """
        # Estimate gradient noise scale using current model state
        S = self.model_ref._estimate_noise_scale(
            self.dataset,
            pl_module.hparams.input_channels,
        )

        # Always append something to the history to keep it aligned with epochs
        if S is None or S <= 0:
            if self.model_ref.verbose:
                print(f"[Dynamic] Epoch {pl_module.current_epoch}: failed to estimate noise scale.")
            self.model_ref.batch_size_history.append(self.current_batch)
            return

        # Compute new batch size using a square-root heuristic
        new_batch = int(
            np.clip(np.sqrt(self.adaptive_factor * S), self.min_batch, self.max_batch)
        )

        # Adjust learning rate to keep roughly constant noise ratio (eta / B ≈ const)
        optimizer = trainer.optimizers[0]
        old_lr = optimizer.param_groups[0]["lr"]
        new_lr = old_lr * np.sqrt(new_batch / max(1, self.current_batch))

        # Apply updates
        self.current_batch = new_batch
        optimizer.param_groups[0]["lr"] = new_lr

        # Record batch size into the model's history
        self.model_ref.batch_size_history.append(new_batch)

        if self.model_ref.verbose:
            print(
                f"[Dynamic] Epoch {pl_module.current_epoch}: "
                f"S={S:.4e}, batch={new_batch}, lr={new_lr:.2e}"
            )

        # Update DataLoader for the next epoch
        train_loader = trainer.train_dataloader
        if hasattr(train_loader, "batch_sampler"):
            train_loader.batch_sampler.batch_size = new_batch
        else:
            trainer.fit_loop._data_loader = DataLoader(
                self.dataset,
                batch_size=new_batch,
                shuffle=True,
                num_workers=0,
            )

        # TensorBoard logging
        if isinstance(trainer.logger, pl.loggers.TensorBoardLogger):
            writer = trainer.logger.experiment
            epoch = pl_module.current_epoch
            writer.add_scalar("adaptive/S_noise", S, epoch)
            writer.add_scalar("adaptive/batch_size", new_batch, epoch)
            writer.add_scalar("adaptive/lr", new_lr, epoch)
