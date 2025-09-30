#!/bin/bash

only_type="$1"  # Can be "Classification", "Regression" or empty
shift

# Optional model names (up to 3)
model_names=()
dataset_names=()
adaptive_batch_flag=""

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
        --adaptive_batch_size)
            adaptive_batch_flag="--adaptive_batch_size"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Build model_names argument string if provided
model_arg=""
if [[ ${#model_names[@]} -gt 0 ]]; then
    model_arg="--model_names ${model_names[*]}"
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
            datasets=("${dataset_names[@]}")
        else
            for ds_path in "$base_dir"/*; do
                if [[ -d "$ds_path" ]]; then
                    datasets+=("$(basename "$ds_path")")
                fi
            done
        fi

        for ds in "${datasets[@]}"; do
            echo "Running on $ds ($mode)"
            python -m Scripts_python.assoc_pp_model.association_pp_model \
                --mode "$mode" \
                --data_source "$ds" \
                --only_colors \
                --progressive_optim \
                $model_arg \
                $adaptive_batch_flag
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
