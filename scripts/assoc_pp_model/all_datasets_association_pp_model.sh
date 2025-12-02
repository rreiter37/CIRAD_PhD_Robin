#!/bin/bash

# This script iterates over datasets and launches a Python pipeline.
# Update: --adaptive_batch_size now requires an explicit value: "dynamic" or "static".
# - If omitted entirely, no adaptive batch argument is forwarded to Python.
# - If provided, it must be followed by either "dynamic" or "static".
# - Any other usage results in an error and exit.

only_type="$1"  # Can be "Classification", "Regression" or empty
shift

# Optional model names and dataset range
model_names=()
dataset_names=()
between_datasets=()

# Will store "dynamic" or "static" if provided (empty means: not set)
adaptive_batch_value=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                model_names+=("$1")
                shift
            done
            ;;
        --dataset_names)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                dataset_names+=("$1")
                shift
            done
            ;;
        --between_datasets)
            shift
            if [[ $# -ge 2 ]]; then
                between_datasets=("$1" "$2")
                shift 2
            else
                echo "Error: --between_datasets requires two dataset names."
                exit 1
            fi
            ;;
        --adaptive_batch_size)
            # Expect a value: "dynamic" or "static"
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                if [[ "$1" == "dynamic" || "$1" == "static" ]]; then
                    adaptive_batch_value="$1"
                    shift
                else
                    echo "Error: --adaptive_batch_size must be 'dynamic' or 'static'."
                    exit 1
                fi
            else
                echo "Error: --adaptive_batch_size requires a value: 'dynamic' or 'static'."
                exit 1
            fi
            ;;
        *)
            # Ignore unknown tokens or move on
            shift
            ;;
    esac
done

# Build model_names argument string if provided
model_arg=""
if [[ ${#model_names[@]} -gt 0 ]]; then
    model_arg="--model_names ${model_names[*]}"
fi

# Build adaptive argument for Python (only if set)
adaptive_arg=""
if [[ -n "$adaptive_batch_value" ]]; then
    adaptive_arg="--adaptive_batch_size $adaptive_batch_value"
fi

# Base directories
classification_dir="Data/Classification"
regression_dir="Data/Regression"

# Function to run datasets
run_datasets() {
    local mode="$1"
    local base_dir="$2"

    if [[ -d "$base_dir" ]]; then
        local datasets=()
        if [[ ${#dataset_names[@]} -gt 0 ]]; then
            # Use explicitly provided dataset names
            datasets=("${dataset_names[@]}")
        else
            # Collect all datasets in alphabetical order
            for ds_path in "$base_dir"/*; do
                if [[ -d "$ds_path" ]]; then
                    datasets+=("$(basename "$ds_path")")
                fi
            done
        fi

        # Sort datasets to ensure consistent ordering
        IFS=$'\n' datasets=($(sort <<<"${datasets[*]}"))
        unset IFS

        # If between_datasets is specified, limit the range
        if [[ ${#between_datasets[@]} -eq 2 ]]; then
            local start="${between_datasets[0]}"
            local end="${between_datasets[1]}"
            local in_range=false
            local filtered_datasets=()

            for ds in "${datasets[@]}"; do
                if [[ "$ds" == "$start" ]]; then
                    in_range=true
                fi
                if [[ "$in_range" == true ]]; then
                    filtered_datasets+=("$ds")
                fi
                if [[ "$ds" == "$end" ]]; then
                    break
                fi
            done

            datasets=("${filtered_datasets[@]}")
        fi

        # Run the Python script for each dataset
        for ds in "${datasets[@]}"; do
            echo "Running on $ds ($mode)"
            python -m scripts.assoc_pp_model.main_pipeline.main \
                --mode "$mode" \
                --data_source "$ds" \
                --only_colors \
                --progressive_optim \
                $model_arg \
                $adaptive_arg
        done
    else
        echo "Directory $base_dir not found!"
    fi
}

if [[ -z "$only_type" || "$only_type" == "Classification" ]]; then
    run_datasets "Classification" "$classification_dir"
fi

if [[ -z "$only_type" || "$only_type" == "Regression" ]]; then
    run_datasets "Regression" "$regression_dir"
fi
