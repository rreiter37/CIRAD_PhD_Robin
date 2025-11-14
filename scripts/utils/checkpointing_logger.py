from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl

class CheckpointLoggerCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if isinstance(trainer.logger, pl.loggers.TensorBoardLogger):
            writer = trainer.logger.experiment
            epoch = trainer.current_epoch
            writer.add_text("checkpoint", f"Checkpoint saved at epoch {epoch}", epoch)