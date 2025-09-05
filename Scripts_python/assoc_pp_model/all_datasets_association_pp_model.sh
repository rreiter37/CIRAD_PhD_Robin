#!/bin/bash

only_type="$1"  # Can be "Classification", "Regression" or empty

# Datasets de classification
classification_datasets=("CoffeeSpecies" "YamMould" "WhiskyConcentration" "mDigest_custom3" "Malaria2024")

# Datasets de régression
regression_datasets=("BeerOriginalExtract" "YamProtein" "Digest_0.8")

if [[ -z "$only_type" || "$only_type" == "Classification" ]]; then
    for ds in "${classification_datasets[@]}"; do
        echo "Running on $ds (Classification)"
        python -m Scripts_python.assoc_pp_model.association_pp_model --mode Classification --data_source "$ds" --only_colors --progressive_optim 
    done
fi

if [[ -z "$only_type" || "$only_type" == "Regression" ]]; then
    for ds in "${regression_datasets[@]}"; do
        echo "Running on $ds (Regression)"
        python -m Scripts_python.assoc_pp_model.association_pp_model --mode Regression --data_source "$ds" --only_colors --progressive_optim
    done
fi