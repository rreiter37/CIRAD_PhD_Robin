# scripts/models/nicon/lightning_module.py

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl

from scripts.Models.DeepLearning.Architectures.nicon_custom_pytorch import CustomizableNicon


class NiconPLModule(pl.LightningModule):
    """
    LightningModule wrapper around the CustomizableNicon CNN.

    This module encapsulates:
    - Model definition
    - Loss function
    - Training and validation steps
    - Optimizer and scheduler configuration
    - TensorBoard logging for epoch-wise metrics
    """

    def __init__(
        self,
        input_channels: int,
        params: Dict[str, Any],
        lr_max: float,
        lr_min: float,
        epochs: int,
        t0_steps: Optional[int] = None,
        cyclic_learning: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        input_channels : int
            Number of input channels for the CNN.
        params : dict
            Dictionary of hyperparameters for CustomizableNicon.
        lr_max : float
            Maximum learning rate for the optimizer.
        lr_min : float
            Minimum learning rate used in cosine annealing.
        epochs : int
            Total number of epochs for training.
        t0_steps : int, optional
            Initial number of steps for CosineAnnealingWarmRestarts.
        cyclic_learning : bool
            Whether to use cosine annealing or a fixed learning rate.
        """
        super().__init__()

        # We keep some hyperparameters external (not saved) for flexibility
        self.save_hyperparameters(ignore=["t0_steps", "cyclic_learning"])

        self.params = params
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epochs = epochs
        self.t0_steps = t0_steps
        self.cyclic_learning = cyclic_learning

        self.model = CustomizableNicon(
            input_channels=input_channels,
            params=self.params,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that ensures the tensor is on the correct device."""
        x = x.to(self.device)
        return self.model(x).squeeze(1)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Single training step with logging of loss and learning rate."""
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
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Single validation step with logging of validation loss."""
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

    def on_train_epoch_end(self) -> None:
        """Manually log train loss and learning rate with epoch as x-axis."""
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            writer = self.logger.experiment
            epoch = self.current_epoch

            train_loss = self.trainer.callback_metrics.get("train_loss")
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
            writer.add_scalar("epoch/train_loss", train_loss, epoch)

            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            writer.add_scalar("epoch/lr", lr, epoch)

    def on_validation_epoch_end(self) -> None:
        """Manually log validation loss with epoch as x-axis."""
        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            writer = self.logger.experiment
            epoch = self.current_epoch

            val_loss = self.trainer.callback_metrics.get("val_loss")
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            writer.add_scalar("epoch/val_loss", val_loss, epoch)

    def configure_optimizers(self):
        """Configure optimizer and optional cosine annealing scheduler."""
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
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer
