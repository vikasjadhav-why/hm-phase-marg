#!/bin/bash

export HDF5_USE_FILE_LOCKING=FALSE

INJECTION_FOLDER=
VARIABLE_PARAMS_FILE=
DATA_FILE=
MODEL_FILE=
MODE_ARRAY=
START=$1
END=$2
OUTPUT_FILE="output_${START}_${END}.hdf"

python /home/vjadhavy/hmphase/hm-phase-marg/helper_functions/run_hm_model.py \
    "$INJECTION_FOLDER" \
    "$VARIABLE_PARAMS_FILE" \
    "$DATA_FILE" \
    "$MODEL_FILE" \
    "$MODE_ARRAY" \
    "$OUTPUT_FILE" \
    "$START" \
    "$END"
