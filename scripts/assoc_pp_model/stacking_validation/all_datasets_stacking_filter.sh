#!/bin/bash

# ============================================================
# Run stacking_from_pipelines.py over all datasets.
#
# Usage:
#   ./run_stacking_all_datasets.sh [Classification|Regression] \
#       [--subset_pipelines ...] \
#       [--dataset_names ...] \
#       [--between_datasets ds1 ds2]
#
# Behaviour:
#   - If first argument is empty → run Classification + Regression
#   - --subset_pipelines lets you choose among:
#         gatekeeping graph_pruning weakness_coverage no_filter all
#   - If --dataset_names is provided → restrict to specific datasets
#   - If --between_datasets A B → run only datasets lexically between A and B
# ============================================================

only_type="$1"   # "Classification", "Regression", or empty
shift

subset_pipelines=()      # Pipelines to pass to python
dataset_names=()         # Explicit list of datasets
between_datasets=()      # Start / end dataset boundaries

# ------------------------------------------------------------
# Parse all remaining arguments
# ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subset_pipelines)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                subset_pipelines+=("$1")
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
        *)
            shift
            ;;
    esac
done


# ------------------------------------------------------------
# Build arguments to forward to Python
# ------------------------------------------------------------

pipeline_arg=""
if [[ ${#subset_pipelines[@]} -gt 0 ]]; then
    pipeline_arg="--subset_pipelines ${subset_pipelines[*]}"
fi


# ------------------------------------------------------------
# Base dataset directories
# ------------------------------------------------------------
classification_dir="Data/Classification"
regression_dir="Data/Regression"


# ------------------------------------------------------------
# Function to iterate over datasets and run stacking script
# ------------------------------------------------------------
run_datasets() {
    local mode="$1"
    local base_dir="$2"

    if [[ ! -d "$base_dir" ]]; then
        echo "Directory $base_dir not found!"
        return
    fi

    # --- Collect datasets ---
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

    # Sort lexicographically for consistency
    IFS=$'\n' datasets=($(sort <<<"${datasets[*]}"))
    unset IFS

    # --- Apply between_datasets filter ---
    if [[ ${#between_datasets[@]} -eq 2 ]]; then
        local start="${between_datasets[0]}"
        local end="${between_datasets[1]}"
        local in_range=false
        local filtered=()

        for ds in "${datasets[@]}"; do
            if [[ "$ds" == "$start" ]]; then
                in_range=true
            fi
            if [[ "$in_range" == true ]]; then
                filtered+=("$ds")
            fi
            if [[ "$ds" == "$end" ]]; then
                break
            fi
        done

        datasets=("${filtered[@]}")
    fi

    # --- Run Python stacking on each dataset ---
    for ds in "${datasets[@]}"; do
        echo "-------------------------------------------------------"
        echo "Running stacking on dataset: $ds ($mode)"
        echo "-------------------------------------------------------"

        python -m scripts.assoc_pp_model.stacking_validation.stacking_from_pipelines \
            --mode "$mode" \
            --data_source "$ds" \
            $pipeline_arg
    done
}


# ------------------------------------------------------------
# Launch datasets according to selected type
# ------------------------------------------------------------

if [[ -z "$only_type" || "$only_type" == "Classification" ]]; then
    run_datasets "Classification" "$classification_dir"
fi

if [[ -z "$only_type" || "$only_type" == "Regression" ]]; then
    run_datasets "Regression" "$regression_dir"
fi