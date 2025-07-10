#!/bin/bash

for ds in YamMould CoffeeSpecies Malaria2024 WhiskyConcentration mDigest_custom3; do
    echo "Running on $ds"
    python association_pp_model.py --mode Classification --data_source $ds
done
for ds in BeerOriginalExtract YamProtein Digest_0.8; do
    echo "Running on $ds"
    python association_pp_model.py --mode Regression --data_source $ds
done