import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import optuna
from optuna.exceptions import TrialPruned
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.storages import RDBStorage
from joblib import Parallel, delayed
import tempfile
import uuid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from scripts.Models.DeepLearning.Architectures.nicon_classif_pytorch import customizable_nicon_classification
from scripts.utils.checkpointing_logger import CheckpointLoggerCallback
from scripts.utils.max_batch_size import find_max_batch_size
from pytorch_lightning.loggers import TensorBoardLogger

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
    torch.use_deterministic_algorithms(True, warn_only=False)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    
    def check_device_consistency(self, batch, stage=""):
        x, y = batch
        model_device = next(self.model.parameters()).device
        if x.device != model_device:
            self.print(f"[DEVICE WARNING] {stage}: Input x is on {x.device}, model is on {model_device}")
        if y.device != model_device:
            self.print(f"[DEVICE WARNING] {stage}: Target y is on {y.device}, model is on {model_device}")

    def training_step(self, batch, batch_idx):
        self.check_device_consistency(batch, stage="Training")
        x, y = batch
        y_pred = self(x)
        if self.hparams.num_classes == 2:
            y = y.float()
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.check_device_consistency(batch, stage="Validation")
        x, y = batch
        y_pred = self(x)
        if self.hparams.num_classes == 2:
            y = y.float()
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
                 parallelize=False, device=None, get_logger=True, get_logger_optuna=False, best_trials=None, name_pp=None):
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
        self.parallelize = parallelize
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
            #"activation1": trial.suggest_categorical("activation1", ['relu', 'selu', 'elu', 'swish']),
            "dropout_rate": median_and_range("dropout_rate", "float", 0.01, 0.5, log=True, scale=0.2),
            "filters2": 2** median_and_range("filters2_power", "int", 2, 8, step=1, scale=0.4),
            "kernel_size2": median_and_range("kernel_size2", "int", 3, 25, step=2, scale=0.4),
            "strides2": median_and_range("strides2", "int", 1, 5, step=1, scale=0.4),
            #"activation2": trial.suggest_categorical("activation2", ['relu', 'selu', 'elu', 'swish']),
            #"normalization_method1": trial.suggest_categorical("normalization_method1", ['BatchNormalization', 'LayerNormalization']),
            "filters3": 2** median_and_range("filters3_power", "int", 2, 8, step=1, scale=0.4),
            "kernel_size3": median_and_range("kernel_size3", "int", 3, 25, step=2, scale=0.4),
            "strides3": median_and_range("strides3", "int", 1, 5, step=1, scale=0.4),
            #"activation3": trial.suggest_categorical("activation3", ['relu', 'selu', 'elu', 'swish']),
            #"normalization_method2": trial.suggest_categorical("normalization_method2", ['BatchNormalization', 'LayerNormalization']),
            "dense_units": 2** median_and_range("dense_units_power", "int", 2, 8, step=1, scale=0.4),
            #"dense_activation": trial.suggest_categorical("dense_activation", ['relu', 'selu', 'elu', 'swish']),
            }

    def _build_and_train(self, trial, X, y):
        set_global_seed(self.random_state)
        params = self._suggest_params(trial)
        X_tensor = torch.tensor(self._reshape(X), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        # Split the dataset into training and validation datasets
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(self.random_state))

        # Build the training and validation datasets
        g = torch.Generator()
        g.manual_seed(self.random_state)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, generator=g, worker_init_fn=seed_worker)

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
            EarlyStopping(monitor="val_loss", mode="min", patience=self.epochs_optuna//10),
            CustomOptunaPruningCallback(trial, monitor="val_loss"),
        ]

        # --- ADD LOGGER ---
        logger = TensorBoardLogger(
            save_dir="lightning_logs", 
            name=f"optuna_trial_{trial.number}"
        )

        trainer = pl.Trainer(
            detect_anomaly=True,
            max_epochs=self.epochs_optuna,
            callbacks=callbacks,
            enable_progress_bar=False,
            logger=logger if self.get_logger_optuna else False,
            deterministic=True,
            enable_model_summary=False,
            accelerator="gpu",
            devices=1,
            precision=32,
            profiler="simple"
        )

        trainer.fit(model, train_loader, val_loader)
        val_loss = trainer.callback_metrics["val_loss"].item()
        return val_loss

    def fit(self, X, y):
        set_global_seed(self.random_state)

        # Check if all values from the dataset are valid
        assert not np.isnan(X).any(), "NaN detected in input features"
        assert not np.isinf(X).any(), "Inf detected in input features"

        # Find the input shape to suit the architecture of the NICON classifier
        if self.input_shape is None:
            self.input_shape = (X.shape[1], 1) if len(X.shape) == 2 else X.shape[1:]
        
        n_samples = len(X)
        n_wavelengths = X.shape[-1]
        print("Number of wavelengths after preprocessing : ", n_wavelengths)
        # Find the maximum batch size accepted by the GPU
        if self.batch_size is None:
            params = {"kernel_size1": 3, "kernel_size2": 3, "kernel_size3": 3, "spatial_dropout": 0.01, "dropout_rate": 0.01}
            params["output_dim"] = 1
            model = NiconPLClassifier(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                params=params,
                lr_max=self.lr_max,
                lr_min=self.lr_min,
                epochs=1,
                t0_steps=n_samples,
                cyclic_learning=self.cyclic_learning
            )
            self.batch_size = find_max_batch_size(model=model, input_shape=self.input_shape, device=self.device, max_batch=min(X.shape[0], X.shape[-1]), min_batch=1)
            print("Maximum batch size found : ", self.batch_size)

        def objective(trial):
            try:
                mdl = self._build_and_train(trial, X, y)
            except ValueError as e:
                raise TrialPruned(str(e))
            return mdl
         
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=self.epochs, reduction_factor=3)
        if self.parallelize:
            # Create a temporary SQLite storage space
            study_id = str(uuid.uuid4())
            storage_path = f"sqlite:///{tempfile.gettempdir()}/optuna_study_{study_id}.db"
            storage = RDBStorage(url=storage_path)

            study_name = f"parallel_nicon_{study_id}"
            self.study_ = optuna.create_study(
                direction="minimize", 
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )

            def _objective_wrapper(trial_number):
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.load_study(study_name=study_name, storage=storage)
                study.optimize(objective, n_trials=1)

            # Lance n_trials essais en parallèle
            Parallel(n_jobs=-1)(
                delayed(_objective_wrapper)(i) for i in range(self.n_trials)
            )

        else:
            self.study_ = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
            self.study_.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_trial_params = self.study_.best_trial.params.copy()

        # Store the best hyperameters for next pp-model associations
        if self.best_trials is None:
            self.best_trials = [best_trial_params]
        else:
            self.best_trials.append(best_trial_params)

        X_tensor = torch.tensor(self._reshape(X), dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split train/val for final training too
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len

        g = torch.Generator()
        g.manual_seed(self.random_state)
        train_set, val_set = random_split(dataset, [train_len, val_len], generator=g)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, generator=g, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, generator=g, worker_init_fn=seed_worker)

        # Train final model
        self.model_ = NiconPLClassifier(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            params=best_trial_params,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            epochs=self.epochs,
            t0_steps=len(train_loader),
            cyclic_learning=self.cyclic_learning
        )

        # --- ADD LOGGER FOR FINAL TRAINING ---
        logger = TensorBoardLogger(
            save_dir="lightning_logs", 
            name=f"final_model_{self.name_pp or 'default'}"
        )

        # --- ADD EARLY STOPPING AND CHECKPOINTING FOR FINAL TRAINING ---
        checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", # Monitor validation loss
        mode="min", # We want the minimum validation loss
        save_top_k=1, # Save only the best model
        save_last=True, # Optionally also save the last model
        dirpath=f"checkpoints/{self.name_pp or 'default'}", # Directory where checkpoints will be saved
        filename="best_model-{epoch:02d}-{val_loss:.4f}", # Naming convention for checkpoints
        verbose=True
        )

        # --- ADD EARLY STOPPING FOR FINAL TRAINING ---
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=self.patience),
            checkpoint_callback
        ]

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator="gpu",
            devices=1,
            logger=logger if self.get_logger else False,
            callbacks=callbacks,
            enable_progress_bar=self.verbose,
            deterministic=True,
            precision=32,
            profiler="simple",
        )

        trainer.fit(self.model_, train_loader, val_loader)
        
        # --- LOAD BEST CHECKPOINTED MODEL ---
        best_path = checkpoint_callback.best_model_path
        if best_path:
            self.model_ = NiconPLClassifier.load_from_checkpoint(best_path)

        self.model_.to(self.device)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        self.model_.to(self.device)
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