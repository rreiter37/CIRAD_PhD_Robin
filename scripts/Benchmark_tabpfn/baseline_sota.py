from dotenv import load_dotenv
from pathlib import Path
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer

from nirs4all.data import DatasetConfigs
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.operators.splitters import SPXYGFold
from nirs4all.operators.transforms import SavitzkyGolay, ASLSBaseline

from huggingface_hub import login
from tabpfn import TabPFNClassifier, TabPFNRegressor
import torch
import argparse

# ===================== Fix TabPFN CUDA incompatibility (RTX 50xx) ===================== #

def get_safe_tabpfn_device():
    """
    Returns 'cuda' if the GPU architecture is supported by TabPFN v2.5,
    otherwise falls back to 'cpu' to prevent CUDA kernel crashes.
    """
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available → Using CPU for TabPFN.")
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)

    # TabPFN v2.5 kernels are NOT compiled for Blackwell (SM 90 / RTX 50xx).
    if major >= 9:
        print("⚠️ Detected GPU architecture incompatible with TabPFN CUDA kernels (SM >= 90).")
        print("➡️ Switching TabPFN to CPU mode to avoid CUDA errors.")
        return "cpu"

    # Otherwise safe to use CUDA
    return "cuda"


TABPFN_DEVICE = get_safe_tabpfn_device()

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

# Hugging Face login for TabPFN
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found in .env or environment. TabPFN may not work properly.")

# Clear GPU cache if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()

##========================== Main Pipeline Configuration =========================##

def parse_args():
    """
    Parse command line arguments allowing an external script to override dataset paths.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of dataset folders to use instead of the default DATA_PATH."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace",
        help="Workspace folder where predictions and logs will be stored."
    )
    return parser.parse_args()


args = parse_args()

# Configuration variables
# If --datasets is provided, override the default DATA_PATH
if args.datasets is not None:
    DATA_PATH = args.datasets
else:
    DATA_PATH = [
        'Data_nirs4all/Regression/Beer_OriginalExtract_60_KS',
    ]

AGGREGATION_KEY = "ID"  # None
TASK_TYPE = "regression"  # "classification" or "regression" or "auto" or "binary"

TabPFNModel = TabPFNRegressor if TASK_TYPE == "regression" else TabPFNClassifier
tabpfn_real_path = 'tabpfn-v2.5-regressor-v2.5_real.ckpt' if TASK_TYPE == "regression" else 'tabpfn-v2.5-classifier-v2.5_real.ckpt'
# Define the pipeline
pipeline = [
    ASLSBaseline(),
    {"split": SPXYGFold(n_splits=1, random_state=42), "group": AGGREGATION_KEY},  # COMMENT IF TRAIN AND TEST ARE PROVIDED
    {"split": SPXYGFold(n_splits=3, random_state=42), "group": AGGREGATION_KEY},
    PCA(n_components=0.99, random_state=42, whiten=True), # PCA(50)
    {
        'model': {
            'framework': 'autogluon',
            'params': {
                'presets': 'extreme_quality',
                'time_limit': 3600,
                'num_bag_folds': 5,
                'random_state': 42,
            }
        },
        "name": "AutoGluon",
    },
    {
        "model": TabPFNModel(n_estimators=16, device=TABPFN_DEVICE, random_state=42, ignore_pretraining_limits=True),
        "name": "TabPFN",
    },
    {
        "model": TabPFNModel(n_estimators=16, device=TABPFN_DEVICE, random_state=42, model_path=tabpfn_real_path, ignore_pretraining_limits=True),
        "name": "TabPFN-real",
    },
]

# Create configuration objects
pipeline_config = PipelineConfigs(pipeline, "SOTA")
dataset_config = DatasetConfigs(DATA_PATH, task_type=TASK_TYPE)

# Run the pipeline
runner = PipelineRunner(save_files=True, verbose=0, workspace_path=args.workspace)
predictions, predictions_per_dataset = runner.run(pipeline_config, dataset_config)

# Analyze and display top performing models
best_model_count = 5
rank_metric = 'rmse' if TASK_TYPE == "regression" else 'balanced_accuracy'
display_metrics = ['rmse', 'r2', 'mape', 'nrmse'] if TASK_TYPE == "regression" else ['accuracy', 'balanced_accuracy', 'f1', 'recall']

for dataset_name, dataset_prediction in predictions_per_dataset.items():
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 80}")

    dataset_predictions = dataset_prediction['run_predictions']

    # Display top performing models
    print("Top Predictions (per row):")
    print("-" * 80)

    print(f"\nTop {best_model_count} models based on validation {rank_metric.upper()}:")
    top_models_val = dataset_predictions.top(best_model_count, rank_metric, rank_partition='val', display_metrics=display_metrics)
    for idx, prediction in enumerate(top_models_val):
        print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=display_metrics)} - {prediction['preprocessings']}")

    print(f"\nTop {best_model_count} models based on test {rank_metric.upper()}:")
    top_models_test = dataset_predictions.top(best_model_count, rank_metric, rank_partition='test', display_metrics=display_metrics)
    for idx, prediction in enumerate(top_models_test):
        print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=display_metrics)} - {prediction['preprocessings']}")

    # Print aggregated results if aggregation_key is provided
    if AGGREGATION_KEY is not None:
        print("*" * 80)
        print(f"\n Top Predictions (aggregated by {AGGREGATION_KEY}):")
        print("-" * 80)

        print(f"\nTop {best_model_count} models based on validation {rank_metric.upper()}:")
        top_models_val = dataset_predictions.top(best_model_count, rank_metric, rank_partition='val', display_metrics=display_metrics, aggregate=AGGREGATION_KEY)
        for idx, prediction in enumerate(top_models_val):
            print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=display_metrics)} - {prediction['preprocessings']}")

        print(f"\nTop {best_model_count} models based on test {rank_metric.upper()}:")
        top_models_test = dataset_predictions.top(best_model_count, rank_metric, rank_partition='test', display_metrics=display_metrics, aggregate=AGGREGATION_KEY)
        for idx, prediction in enumerate(top_models_test):
            print(f"{idx + 1}. {Predictions.pred_short_string(prediction, metrics=display_metrics)} - {prediction['preprocessings']}")