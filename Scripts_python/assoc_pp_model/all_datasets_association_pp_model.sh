#!/bin/bash

only_type="$1"  # Can be "Classification", "Regression" or empty
shift

# Optional model names (up to 3)
model_names=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                model_names+=("$1")
                shift
            done
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

if [[ -z "$only_type" || "$only_type" == "Classification" ]]; then
    if [[ -d "$classification_dir" ]]; then
        for ds_path in "$classification_dir"/*; do
            if [[ -d "$ds_path" ]]; then
                ds=$(basename "$ds_path")
                echo "Running on $ds (Classification)"
                python -m Scripts_python.assoc_pp_model.association_pp_model \
                    --mode Classification \
                    --data_source "$ds" \
                    --only_colors \
                    --progressive_optim \
                    $model_arg
            fi
        done
    else
        echo "Directory $classification_dir not found!"
    fi
fi

if [[ -z "$only_type" || "$only_type" == "Regression" ]]; then
    if [[ -d "$regression_dir" ]]; then
        for ds_path in "$regression_dir"/*; do
            if [[ -d "$ds_path" ]]; then
                ds=$(basename "$ds_path")
                echo "Running on $ds (Regression)"
                python -m Scripts_python.assoc_pp_model.association_pp_model \
                    --mode Regression \
                    --data_source "$ds" \
                    --only_colors \
                    --progressive_optim \
                    $model_arg
            fi
        done
    else
        echo "Directory $regression_dir not found!"
    fi
fi
